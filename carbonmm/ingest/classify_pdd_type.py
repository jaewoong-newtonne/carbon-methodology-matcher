"""Classify each PDD into a document type + semantic class.

This is the **top-tier criterion** for the methodology-recommendation pipeline:
the meaning of "Section A" (and therefore the validity of feeding it to the
recommender) depends on what kind of document the PDD is.

Five semantic classes:

  * **single**     — 1 project = 1 document. Section A is the project's
                     own description. Examples: GS-PDD standalone, VCS-PD,
                     CDM-PDD legacy, CDM-PDD-FORM, CDM-SSC-PDD.
  * **component**  — 1 of N projects under an umbrella PoA. Section A is
                     this component's description; methodology is inherited
                     from the umbrella but the same code label applies.
                     Examples: GS-VPA-DD, CDM-CPA-DD, VCS-CPA-component.
  * **umbrella**   — Programme framework, no specific project. Section A
                     is framework-level. Examples: GS-PoA-DD, VCS-Grouped.
  * **multi**      — Multiple distinct projects in one document. Section A
                     is collective. Example: VCS-Joint.
  * **excluded**   — Non-methodology document we shouldn't process. Example:
                     SD-VISta (Verra Sustainable Development Verified Impact).

Three-tier detection: filename signatures (most reliable when body extraction
fails) → body cover signatures → modern-Verra fallback for the 36% of PDDs
that lack a recognizable cover signature but follow the post-2020 VCS layout.

Output: a single manifest at `data/manifests/pdd-classification.json` mapping
globalId → {doc_type, semantic_class, classification_source, poa_id?}.

Usage:
    python3 -m carbonmm.ingest.classify_pdd_type
    python3 -m carbonmm.ingest.classify_pdd_type --code GS11044
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ─── Type system ──────────────────────────────────────────────────────────

# Map of doc_type → semantic_class. Single source of truth.
DOC_TYPE_TO_CLASS: dict[str, str] = {
    "GS-PDD standalone":   "single",
    "VCS-PD":              "single",
    "CDM-PDD legacy":      "single",
    "CDM-PDD-FORM":        "single",
    "CDM-SSC-PDD":         "single",
    "GS-VPA-DD":           "component",
    "CDM-CPA-DD":          "component",
    "VCS-CPA-component":   "component",
    "GS-PoA-DD":           "umbrella",
    "VCS-Grouped":         "umbrella",
    "VCS-Joint":           "multi",
    "SD-VISta":            "excluded",
    "unknown":             "unknown",
    "empty":               "unparseable",
}


@dataclass
class Classification:
    globalId: str
    registry: str
    doc_type: str
    semantic_class: str
    classification_source: str  # which tier matched
    source_pdf: str
    poa_id: Optional[str] = None


# ─── Filename signatures (tier 1) ─────────────────────────────────────────

# These match against the upper-cased source_pdf filename. Note: no `\b` at end
# of *_DD patterns because filenames often have `_V3` / `_NEW` suffixes where
# the underscore is a word character (no word boundary forms there).
FN_GS_VPA_DD = re.compile(r"(?:VPA[-_]DD|RVPA)")
FN_CDM_CPA_DD = re.compile(r"(?:CPA[-_]DD|CPA[-_]\d)")
FN_POA_DD = re.compile(r"(?:POA[-_]DD|POA[-_]\d)")
FN_SD_VISTA = re.compile(r"(?:SD[-_\s]?VISTA|VERRA\s+SD)")
FN_VCS_JOINT = re.compile(r"JOINT[-_\s]?PROJECT")
FN_CCB_STANDARD = re.compile(r"\bCCB[A-Z]?[-_\s]?(?:PDD|AR[-_]?PDD|STANDARD)")
FN_VCS_PD_MODERN = re.compile(
    r"(?:VCS[-_]PROJECT[-_]DESCRIPTION|"
    r"VCS[-_]PD[_\s-]|"
    r"\b\d{3,5}[-_]VCS[-_]PROJECT)"
)

# ─── Body cover signatures (tier 2) ───────────────────────────────────────
# All match against text[:5000].upper().

BODY_SIGNATURES = [
    # Order: most specific first.
    (re.compile(r"SD[\s-]?VISTA|VERRA\s+SD\s+VISTA"),                 "SD-VISta"),
    # CCB / CCBA = Climate, Community & Biodiversity standard — distinct from VCS methodology
    (re.compile(r"CLIMATE,?\s+COMMUNITY\s+AND\s+BIODIVERSITY"
                r"|\bCCB[A-Z]?[-\s]?PDD\b"
                r"|\bCCB[-\s]?STANDARD"),                             "SD-VISta"),  # treat same as SD-VISta: excluded
    # F-CDM-SSC-CPA-DD (small-scale CPA) — component of CDM SSC PoA
    (re.compile(r"F[-\s]?CDM[-\s]?SSC[-\s]?CPA[-\s]?DD"
                r"|SMALL[-\s]?SCALE\s+COMPONENT\s+PROJECT\s+ACTIVITY"),"CDM-CPA-DD"),
    (re.compile(r"COMPONENT\s+PROJECT\s+ACTIVITY\s+DESIGN\s+DOCUMENT"
                r"|CDM[-\s]?CPA[-\s]?DD"),                            "CDM-CPA-DD"),
    (re.compile(r"GROUPED\s+PROJECT(?:\s+DESCRIPTION)?"),             "VCS-Grouped"),
    (re.compile(r"JOINT\s+PROJECT(?:\s+DESCRIPTION)?"),               "VCS-Joint"),
    (re.compile(r"PROGRAMME\s+OF\s+ACTIVITIES"
                r"|POA[-\s]?DD\b"
                r"|POA\s+DESIGN"),                                    "GS-PoA-DD"),
    (re.compile(r"CDM[-\s]?SSC[-\s]?PDD"
                r"|SMALL[-\s]?SCALE\s+(?:CDM\s+)?PROJECT\s+ACTIVITY"),"CDM-SSC-PDD"),
    (re.compile(r"CDM[-\s]?PDD[-\s]?FORM"),                           "CDM-PDD-FORM"),
    (re.compile(r"CLEAN\s+DEVELOPMENT\s+MECHANISM\s+PROJECT\s+DESIGN"
                r"|CDM\s+PDD\b"
                r"|PROJECT\s+DESIGN\s+DOCUMENT\s+FORM\s*\(CDM[-\s]?PDD\)"),
                                                                      "CDM-PDD legacy"),
    # Modern VCS (with "Version") AND legacy VCS templates lacking "Version"
    (re.compile(r"PROJECT\s+DESCRIPTION:?\s+VCS\s+VERSION"
                r"|VCS\s+PROJECT\s+DESCRIPTION\s+TEMPLATE"
                r"|VOLUNTARY\s+CARBON\s+STANDARD\s+PROJECT\s+DESCRIPTION"
                r"|VCS\s+PROJECT\s+DESCRIPTION"),                     "VCS-PD"),
    (re.compile(r"VPA\s+DESIGN\s+DOCUMENT"
                r"|KEY\s+PROJECT\s+INFORMATION\s*&?\s*VPA"),          "GS-VPA-DD"),
    (re.compile(r"KEY\s+PROJECT\s+INFORMATION\s*&?\s*PROJECT\s+DESIG[N]?\s+DOCUMENT"
                r"|GOLD\s+STANDARD\s+FOR\s+THE\s+GLOBAL\s+GOALS"
                r"|PROJECT\s+DESIGN\s+DOCUMENT\s*\(PDD\)"),           "GS-PDD standalone"),
]

# ─── Modern Verra fallback (tier 3) ───────────────────────────────────────
# Matches the 2020+ Verra layout that lacks a "PROJECT DESCRIPTION: VCS Version" cover
# but starts with <uppercase title> + "Document Prepared by" within first 1k chars.

MODERN_VERRA_FALLBACK = re.compile(
    # Match "Document Prepared by" anywhere in the first ~1k chars; the uppercase-title
    # heuristic is too brittle (unicode chars, project names with mixed case break it).
    r"Document\s+Prepared\s+by",
    re.IGNORECASE,
)
# CPA component indicator in a Verra title (e.g., "CPA-W-015", "CPA 15", "CPA-002")
CPA_IN_TITLE = re.compile(r"CPA[-\s]?[A-Z]?[-\s]?\d{1,4}\b", re.IGNORECASE)


# ─── PoA id extraction for component dedupe ───────────────────────────────

# GS-VPA-DD: filename has "GS5658 VPA-37" or "GS11044_GS5658-VPA-DD" — extract first GS\d{4,5}.
POA_GS_RE = re.compile(r"\bGS[\s_]?(\d{4,5})\b")
# CDM-CPA-DD: body usually has "PoA NN" or filename has the parent PoA id.
POA_CDM_RE = re.compile(r"\bPo[Aa][-\s]?\d+(?:\.\d+)?\b")
# VCS-CPA-component: title has "CPA-W-015" → group "CPA-W"; or "CPA 15" → group "CPA"
VCS_CPA_GROUP_RE = re.compile(r"CPA[-\s]?([A-Z]?)[-\s]?\d{1,4}", re.IGNORECASE)


def extract_poa_id(doc_type: str, source_pdf: str, ft_head: str,
                   global_id: str) -> Optional[str]:
    pdf = source_pdf or ""
    if doc_type == "GS-VPA-DD":
        # Prefer filename PoA reference distinct from the VPA's own globalId
        for m in POA_GS_RE.finditer(pdf):
            candidate = f"GS{m.group(1)}"
            if candidate != global_id:
                return candidate
        # Body fallback
        for m in POA_GS_RE.finditer(ft_head):
            candidate = f"GS{m.group(1)}"
            if candidate != global_id:
                return candidate
        return None  # cannot derive; will be deduped alone
    if doc_type == "CDM-CPA-DD":
        m = POA_CDM_RE.search(ft_head) or POA_CDM_RE.search(pdf)
        return m.group(0).replace(" ", "").replace("_", "-") if m else None
    if doc_type == "VCS-CPA-component":
        m = VCS_CPA_GROUP_RE.search(pdf) or VCS_CPA_GROUP_RE.search(ft_head[:500])
        if m:
            suffix = m.group(1) or ""
            return f"CPA-{suffix}" if suffix else "CPA"
        return None
    return None


# ─── Classifier ───────────────────────────────────────────────────────────


def classify_pdd(global_id: str, registry: str,
                 source_pdf: str, full_text: str) -> Classification:
    if not full_text or not full_text.strip():
        return Classification(global_id, registry, "empty",
                              DOC_TYPE_TO_CLASS["empty"],
                              "empty_body", source_pdf or "")

    pdf_up = (source_pdf or "").upper()
    head_up = full_text[:5000].upper()
    head_first1k = full_text[:1000]

    # Tier 1 — filename signatures (high precision)
    if FN_SD_VISTA.search(pdf_up) or FN_CCB_STANDARD.search(pdf_up):
        return Classification(global_id, registry, "SD-VISta",
                              DOC_TYPE_TO_CLASS["SD-VISta"],
                              "filename:non-methodology-standard",
                              source_pdf or "")
    if FN_VCS_JOINT.search(pdf_up):
        return Classification(global_id, registry, "VCS-Joint",
                              DOC_TYPE_TO_CLASS["VCS-Joint"],
                              "filename:joint", source_pdf or "")
    if FN_POA_DD.search(pdf_up):
        return Classification(global_id, registry, "GS-PoA-DD",
                              DOC_TYPE_TO_CLASS["GS-PoA-DD"],
                              "filename:PoA-DD", source_pdf or "")
    if FN_CDM_CPA_DD.search(pdf_up):
        # Most CPA-DD filenames are CDM, but VCS uses "VCS-PD_CPA NN" pattern → check body
        body_doc_type = _body_match(head_up)
        if body_doc_type == "VCS-PD":
            doc_type = "VCS-CPA-component"
            src = "filename:CPA + body:VCS-PD"
        else:
            doc_type = "CDM-CPA-DD"
            src = "filename:CPA-DD"
        poa = extract_poa_id(doc_type, source_pdf or "", full_text[:5000], global_id)
        return Classification(global_id, registry, doc_type,
                              DOC_TYPE_TO_CLASS[doc_type], src,
                              source_pdf or "", poa)
    if FN_GS_VPA_DD.search(pdf_up):
        poa = extract_poa_id("GS-VPA-DD", source_pdf or "", full_text[:5000], global_id)
        return Classification(global_id, registry, "GS-VPA-DD",
                              DOC_TYPE_TO_CLASS["GS-VPA-DD"],
                              "filename:VPA-DD", source_pdf or "", poa)

    # Tier 2 — body cover signatures
    body_doc_type = _body_match(head_up)
    if body_doc_type:
        # For body-matched component docs, derive PoA id
        poa = extract_poa_id(body_doc_type, source_pdf or "",
                             full_text[:5000], global_id) \
            if DOC_TYPE_TO_CLASS.get(body_doc_type) == "component" else None
        return Classification(global_id, registry, body_doc_type,
                              DOC_TYPE_TO_CLASS[body_doc_type],
                              f"body:{body_doc_type}",
                              source_pdf or "", poa)

    # Tier 3 — modern Verra fallback
    if (FN_VCS_PD_MODERN.search(pdf_up) or
            MODERN_VERRA_FALLBACK.search(head_first1k)):
        # Distinguish CPA component within modern Verra
        title_chunk = full_text[:300]
        if CPA_IN_TITLE.search(title_chunk):
            doc_type = "VCS-CPA-component"
            poa = extract_poa_id(doc_type, source_pdf or "",
                                 full_text[:5000], global_id)
            return Classification(global_id, registry, doc_type,
                                  DOC_TYPE_TO_CLASS[doc_type],
                                  "tier3:VCS-modern+CPA",
                                  source_pdf or "", poa)
        return Classification(global_id, registry, "VCS-PD",
                              DOC_TYPE_TO_CLASS["VCS-PD"],
                              "tier3:VCS-modern", source_pdf or "")

    return Classification(global_id, registry, "unknown",
                          DOC_TYPE_TO_CLASS["unknown"],
                          "no_signature_matched", source_pdf or "")


def _body_match(head_up: str) -> Optional[str]:
    for pat, doc_type in BODY_SIGNATURES:
        if pat.search(head_up):
            return doc_type
    return None


# ─── CLI ──────────────────────────────────────────────────────────────────


def iter_body_files(eval_pdd_root: Path) -> Iterator[Path]:
    for reg in ("GS", "VCS"):
        reg_dir = eval_pdd_root / reg
        if not reg_dir.exists():
            continue
        for f in sorted(reg_dir.glob("*/*.body.json")):
            yield f


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    default_eval = repo_root / "data" / "eval-pdds"
    default_out = repo_root / "data" / "manifests" / "pdd-classification.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-pdd-root", type=Path, default=default_eval)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--code", type=str, default=None,
                    help="run only on a single PDD (smoke test)")
    args = ap.parse_args()

    if not args.eval_pdd_root.exists():
        sys.exit(f"ERROR: {args.eval_pdd_root} does not exist")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    classifications: list[Classification] = []
    type_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    src_counter: Counter[str] = Counter()
    by_registry: dict[str, Counter] = {"GS": Counter(), "VCS": Counter()}

    for body_path in iter_body_files(args.eval_pdd_root):
        try:
            body = json.loads(body_path.read_text())
        except Exception:
            continue
        gid = body["globalId"]
        if args.code and gid != args.code:
            continue
        result = classify_pdd(
            gid, body["registry"],
            body.get("source_pdf") or "",
            body.get("full_text") or "",
        )
        classifications.append(result)
        type_counter[result.doc_type] += 1
        class_counter[result.semantic_class] += 1
        src_counter[result.classification_source] += 1
        by_registry.setdefault(result.registry, Counter())[result.doc_type] += 1

    manifest = {
        "version": "1.0",
        "n_total": len(classifications),
        "by_doc_type": dict(type_counter),
        "by_semantic_class": dict(class_counter),
        "by_classification_source": dict(src_counter),
        "by_registry_doc_type": {r: dict(c) for r, c in by_registry.items()},
        "classifications": [asdict(c) for c in classifications],
    }
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"=== PDD classification — {manifest['n_total']} PDDs ===")
    print(f"\nBy doc_type:")
    for t, n in sorted(type_counter.items(), key=lambda x: -x[1]):
        sc = DOC_TYPE_TO_CLASS.get(t, "?")
        print(f"  {t:<22} {n:>5} ({sc})")
    print(f"\nBy semantic_class:")
    for c, n in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {c:<14} {n:>5} ({n / manifest['n_total']:.1%})")
    print(f"\nBy classification source (top 10):")
    for s, n in sorted(src_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"  {s:<35} {n:>5}")
    keep = class_counter.get("single", 0) + class_counter.get("component", 0)
    drop = manifest['n_total'] - keep
    print(f"\nInclusion preview (single + component): {keep} KEEP / {drop} DROP")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
