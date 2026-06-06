"""Guidance harvester — registry FRAMEWORK RULEBOOKS (not methodologies / PDDs).

Pulls registry-level rulebooks (CDM Project Standard, CDM-EB66-A23-GUID SSC
general guidelines, VCS Standard / Program Guide / Methodology Requirements,
GS4GG Principles & Requirements + key guidance) — the documents that encode
applicability / eligibility / threshold decision-logic.

Two phases (discover-then-fetch), mirroring scrape_gs.py. Run from repo root:

    # 1. Discover: crawl catalog pages → review manifest (all rows pending)
    python3 -m carbonmm.ingest.scrape_guidance --discover --registry gs

    # 2. REVIEW data/manifests/guidance-review.yaml — drop spurious/superseded
    #    rows, fix version/doc_type, flip the ones you want to: review_status: approved

    # 3. Fetch: download ONLY approved rows (SHA256 + skip-exists)
    python3 -m carbonmm.ingest.scrape_guidance --fetch --registry gs

Seed config: carbonmm/configs/guidance_index.yaml
Storage:     data/corpus/{REG}/_guidance/{doc_id}/{file}.pdf
             (--remote → $DATA_ROOT/methodology-corpus/{REG}/_guidance/...)

Engines (per registry, from seed config):
  playwright  headless Chromium (repo playwright_fetcher) — default automatable
  snapshot    parse a local data/snapshots/guidance-{reg}-YYYYMMDD.html

WAF-hard interactive discovery (e.g. CDM Incapsula): render the page in a real
browser, "Save Page As" → the snapshot path above, then run --discover
--engine snapshot.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import yaml

from .common import (
    MANIFEST_DIR,
    MethodologyEntry,
    RateLimiter,
    build_client,
    guidance_dirs,
    load_manifest,
    save_manifest,
    setup_logging,
    _utcnow,
)

logger = logging.getLogger(__name__)

SEED_YAML = Path(__file__).resolve().parents[1] / "configs" / "guidance_index.yaml"
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots"

# All standards/framework PDFs are anchored as .pdf links on the catalog pages.
PDF_ANCHOR_SELECTOR = "a[href$='.pdf'], a[href*='.pdf?']"
# VCS/Verra lists docs as /documents/{slug}/ pages that 302-redirect to the PDF.
DOCLINK_SELECTOR = "a[href*='/documents/']"


# ─── Seed config ─────────────────────────────────────────────────────────────
def load_seed() -> dict:
    if not SEED_YAML.exists():
        raise FileNotFoundError(f"Missing seed config: {SEED_YAML}")
    return yaml.safe_load(SEED_YAML.read_text())["registries"]


# ─── Filename / row parsing ──────────────────────────────────────────────────
def parse_standards_filename(href: str) -> dict:
    """Parse {code}_V{version}_{TAG}_{slug}.pdf (GS framework convention).

    Falls back to a filename-derived slug for non-matching patterns (other
    registries / irregular names). Returns code/version/title/filename.
    """
    name = href.rsplit("/", 1)[-1].split("?", 1)[0]
    m = re.match(r"^([A-Za-z0-9.\-]+?)_V([\d.]+)_(.+?)\.pdf$", name, re.IGNORECASE)
    if not m:
        stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
        return {"filename": name, "code": "", "version": "",
                "title": stem.replace("-", " ").replace("_", " ").strip()}
    code, version, rest = m.group(1), m.group(2), m.group(3)
    tag, slug = rest.split("_", 1) if "_" in rest else ("", rest)
    return {
        "filename": name,
        "code": code,
        "version": version,
        "tag": tag,
        "title": slug.replace("-", " ").replace("_", " ").strip(),
    }


# ── CDM row-based parsing (identity lives in row text, not the filename) ─────
# Modern CDM docs carry a code like CDM-EB66-A23-GUID / CDM-EB119-A04-AMEN;
# older ones carry "EBxx AnnexNN" / "EBxx ParaNN". Version is "VerXX.X".
_CDM_CODE_RE = re.compile(r"\b(CDM-EB\d+-A\d+-[A-Z]+)\b")
_CDM_VER_RE = re.compile(r"\bVer\s*([\d.]+)", re.IGNORECASE)


def parse_cdm_row(href: str, row_text: str) -> dict:
    """Extract {code, version, title} from a CDM catalog table row."""
    row = (row_text or "").strip()
    code_m = _CDM_CODE_RE.search(row)
    ver_m = _CDM_VER_RE.search(row)
    code = code_m.group(1) if code_m else ""
    version = ver_m.group(1) if ver_m else ""
    # Title = text up to the earliest of " Ver", " Note:", or the CDM code.
    cuts = [row.find(m) for m in (" Ver", " Note:", " CDM-EB") if row.find(m) > 0]
    title = row[:min(cuts)].strip() if cuts else row
    stem = href.rsplit("/", 1)[-1].split("?", 1)[0]
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    return {"code": code, "version": version, "title": title[:140], "stem": stem}


_SECTION_DOCTYPE = [("standard", "standard"), ("procedure", "procedure"),
                    ("guideline", "guideline"), ("clarif", "guideline")]


def doc_type_from_section(section: str) -> str:
    s = (section or "").lower()
    for needle, dt in _SECTION_DOCTYPE:
        if needle in s:
            return dt
    return ""


def doc_type_from_row(row_text: str) -> str:
    """Map a GS catalog row label → doc_type. '' if unknown."""
    t = (row_text or "").lower()
    if "optional" in t:
        return "optional"
    if "core document" in t:
        return "core"
    if "guideline" in t:
        return "guideline"
    if "procedure" in t:
        return "procedure"
    return ""


# ── VCS/Verra doclink parsing (link text "VCS Standard, v5.0" → base + version) ─
_DOCLINK_VER_RE = re.compile(r"^(.*?)[,\s]+v\.?\s*([\d.]+)\s*$", re.IGNORECASE)


def parse_doclink_text(text: str) -> dict:
    """Extract {title, version, slug} from a Verra /documents/ link label."""
    text = (text or "").strip()
    m = _DOCLINK_VER_RE.match(text)
    base, version = (m.group(1).strip(), m.group(2)) if m else (text, "")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-")
    return {"title": base[:140], "version": version, "slug": slug}


def _ver_key(v: str) -> tuple:
    """Sortable version key; ('') → (0,). Tolerates non-numeric parts."""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")  # DD.MM.YYYY (GS)


def iso_date_from_row(row_text: str) -> str:
    m = _DATE_RE.search(row_text or "")
    if not m:
        return ""
    d, mo, y = m.group(1), m.group(2), m.group(3)
    return f"{y}-{int(mo):02d}-{int(d):02d}"


# ─── Catalog extraction (engine-agnostic via a Playwright page) ──────────────
def _extract_anchors(page, selector: str = PDF_ANCHOR_SELECTOR) -> list[dict]:
    """Return [{href, text, row}] for every matching anchor on the current page."""
    return page.eval_on_selector_all(
        selector,
        """els => els.map(e => {
            const row = e.closest('tr,li,article,section,div');
            return {
                href: e.href,
                text: (e.innerText||'').replace(/\\s+/g,' ').trim().slice(0,160),
                row: row ? (row.innerText||'').replace(/\\s+/g,' ').trim().slice(0,180) : ''
            };
        })""",
    )


def _load_page(page, *, url: str | None, html: str | None, wait_seconds: float = 2.5) -> bool:
    """Point a Playwright page at a live URL or local snapshot HTML.

    Returns False if a WAF challenge page is detected (tiny + Incapsula marker).
    """
    if html is not None:
        page.set_content(html, wait_until="domcontentloaded")
        return True
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    time.sleep(wait_seconds)
    content = page.content()
    if len(content) < 2000 and ("Incapsula" in content or "incap_ses" in content):
        logger.warning("  WAF challenge suspected (%d bytes) at %s — try --engine snapshot "
                       "(render the page in a real browser, save HTML).", len(content), url)
        return False
    return True


def discover_registry(reg_key: str, reg_cfg: dict, *, snapshot_dir: Path) -> list[MethodologyEntry]:
    """Crawl one registry's catalog pages → pending MethodologyEntry rows."""
    from .playwright_fetcher import browser_context

    registry = reg_cfg.get("registry", reg_key.upper())
    authority = reg_cfg.get("authority", "")
    engine = reg_cfg.get("discover_engine", "playwright")
    parse_mode = reg_cfg.get("parse_mode", "filename")
    host_prefix = reg_cfg.get("pdf_host_prefix", "")
    catalogs = reg_cfg.get("catalogs", [])
    expected = {e["doc_id"] for e in reg_cfg.get("expected", [])}

    logger.info("── %s: discovering via engine=%s (%d catalog pages) ──", registry, engine, len(catalogs))

    entries: dict[str, MethodologyEntry] = {}  # doc_id → entry (dedupe across catalogs)
    rl = RateLimiter(requests_per_second=0.5)  # 2s between pages — polite

    with browser_context() as ctx:
        page = ctx.new_page()
        for cat in catalogs:
            url = cat["url"] if isinstance(cat, dict) else cat
            section = cat.get("section", "") if isinstance(cat, dict) else ""
            rl.wait()

            html = None
            if engine == "snapshot":
                snap = _newest_snapshot(snapshot_dir, reg_key)
                if not snap:
                    logger.error("  snapshot engine but no data/snapshots/guidance-%s-*.html found", reg_key)
                    continue
                html = snap.read_text(errors="ignore")
                logger.info("  [snapshot] %s (%d bytes)", snap.name, len(html))

            selector = DOCLINK_SELECTOR if parse_mode == "doclink" else PDF_ANCHOR_SELECTOR
            try:
                if not _load_page(page, url=None if html else url, html=html):
                    continue
                anchors = _extract_anchors(page, selector)
            except Exception as ex:
                logger.error("  %s → %s", url, ex)
                continue

            found_here = 0
            for a in anchors:
                href = a["href"]
                if host_prefix and host_prefix not in href:
                    continue  # skip nav/footer links outside the registry host
                row = a.get("row", "")
                if parse_mode == "doclink":
                    meta = parse_doclink_text(a.get("text", ""))
                    if not meta["slug"]:
                        continue
                    slug = meta["slug"]
                    doc_id = slug if slug.upper().startswith(registry.upper()) else f"{registry}-{slug}"
                    new = MethodologyEntry(
                        registry=registry, code=doc_id, title=meta["title"],
                        detail_url=url, pdf_url=href, doc_type=doc_type_from_section(section),
                        version=meta["version"], authority=authority,
                        review_status="pending",
                        notes="; ".join(p for p in [f"section={section}" if section else "",
                                                    "resolve=redirect"] if p),
                    )
                    prev = entries.get(doc_id)
                    if prev is None or _ver_key(meta["version"]) > _ver_key(prev.version):
                        entries[doc_id] = new          # keep latest version per doc
                        if prev is None:
                            found_here += 1
                    continue
                if parse_mode == "row":
                    meta = parse_cdm_row(href, row)
                    doc_id = meta["code"] or f"{registry}-{meta['stem']}"
                    doc_type = doc_type_from_section(section)
                    version = meta["version"]
                    title = meta["title"]
                    filename = meta["stem"] + ".pdf"
                else:  # filename mode (GS)
                    meta = parse_standards_filename(href)
                    code = meta["code"] or re.sub(r"\.pdf$", "", meta["filename"], flags=re.IGNORECASE)
                    doc_id = f"{registry}-{code}"
                    doc_type = doc_type_from_row(row)
                    version = meta["version"]
                    title = meta["title"]
                    filename = meta["filename"]
                if doc_id in entries:
                    continue
                entries[doc_id] = MethodologyEntry(
                    registry=registry,
                    code=doc_id,                       # doc_id lives in `code` (see common.py)
                    title=title,
                    detail_url=url,
                    pdf_url=href,
                    doc_type=doc_type,
                    version=version,
                    authority=authority,
                    effective_date=iso_date_from_row(row),
                    review_status="pending",
                    notes="; ".join(p for p in [f"section={section}" if section else "",
                                                 f"file={filename}"] if p),
                )
                found_here += 1
            logger.info("  %-46s → %d PDFs", section or url, found_here)

    found_ids = set(entries)
    missing = expected - found_ids
    logger.info("%s: discovered %d unique docs (expected ≥%d). %s",
                registry, len(entries), len(expected),
                f"MISSING expected: {sorted(missing)}" if missing else "all expected present ✓")
    return list(entries.values())


def _newest_snapshot(snapshot_dir: Path, reg_key: str) -> Path | None:
    cands = sorted(snapshot_dir.glob(f"guidance-{reg_key}-*.html"))
    return cands[-1] if cands else None


def discover(reg_keys: list[str], manifest_path: Path, snapshot_dir: Path) -> list[MethodologyEntry]:
    seed = load_seed()
    all_entries: list[MethodologyEntry] = []
    for rk in reg_keys:
        cfg = seed.get(rk.upper())
        if not cfg:
            logger.error("No seed config for registry %r (have: %s)", rk, list(seed))
            continue
        all_entries.extend(discover_registry(rk, cfg, snapshot_dir=snapshot_dir))
    return all_entries


# ─── Fetch (download approved rows) ──────────────────────────────────────────
def fetch_pdfs(
    entries: list[MethodologyEntry],
    *,
    remote: bool = False,
    require_approved: bool = True,
    max_n: int | None = None,
) -> list[MethodologyEntry]:
    """Download approved rows to {REG}/_guidance/{doc_id}/. Mirrors scrape_gs.download_pdfs."""
    import hashlib

    def eligible(e: MethodologyEntry) -> bool:
        if not e.pdf_url:
            return False
        if require_approved and e.review_status != "approved":
            return False
        return True

    targets = [e for e in entries if eligible(e)]
    if max_n:
        targets = targets[:max_n]
    n_pending = sum(1 for e in entries if e.pdf_url and e.review_status != "approved")
    logger.info("Fetching %d approved PDFs (%d non-approved skipped; require_approved=%s).",
                len(targets), n_pending if require_approved else 0, require_approved)

    rl = RateLimiter(requests_per_second=1.0)
    with build_client(timeout=60.0) as client:
        for i, e in enumerate(targets, 1):
            rl.wait()
            _, out_dir = guidance_dirs(e.registry, e.code, remote=remote)
            direct = e.pdf_url.rsplit("/", 1)[-1].split("?", 1)[0]
            is_pdf_name = direct.lower().endswith(".pdf")
            # skip-exists: direct → known filename; redirect → any cached *.pdf in the doc dir
            if is_pdf_name:
                p = out_dir / direct
                cached = p if (p.exists() and p.stat().st_size > 1024) else None
            else:
                existing = [p for p in out_dir.glob("*.pdf") if p.stat().st_size > 1024]
                cached = existing[0] if existing else None
            if cached:
                logger.info("  [%2d/%d] cached: %s", i, len(targets), cached.name)
                e.pdf_local_path = str(cached)
                continue
            try:
                r = client.get(e.pdf_url)  # build_client() follows redirects → final PDF
                r.raise_for_status()
                if r.headers.get("content-type", "").startswith("application/pdf") or len(r.content) > 10_000:
                    fname = direct if is_pdf_name else (str(r.url).rsplit("/", 1)[-1].split("?", 1)[0] or e.code)
                    if not fname.lower().endswith(".pdf"):
                        fname += ".pdf"
                    out_path = out_dir / fname
                    out_path.write_bytes(r.content)
                    e.pdf_local_path = str(out_path)
                    e.pdf_bytes = len(r.content)
                    e.pdf_sha256 = hashlib.sha256(r.content).hexdigest()
                    e.fetched_at = _utcnow()
                    logger.info("  [%2d/%d] saved: %s (%d KB)", i, len(targets), out_path.name, e.pdf_bytes // 1024)
                else:
                    logger.warning("  [%2d/%d] non-PDF or tiny response: %s", i, len(targets), e.pdf_url)
            except Exception as ex:
                logger.error("  [%2d/%d] %s → %s", i, len(targets), e.pdf_url, ex)
    return entries


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _resolve_registries(arg: str) -> list[str]:
    if arg == "all":
        return ["cdm", "vcs", "gs"]
    return [arg]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--discover", action="store_true", help="Crawl catalog pages → review manifest (pending)")
    p.add_argument("--fetch", action="store_true", help="Download approved rows from the manifest")
    p.add_argument("--registry", default="gs", choices=["gs", "cdm", "vcs", "all"])
    p.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "guidance-review.yaml")
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--remote", action="store_true", help="Store under the shared data root instead of the local data dir")
    p.add_argument("--require-approved", action=argparse.BooleanOptionalAction, default=True,
                   help="Only fetch rows with review_status: approved (default ON)")
    p.add_argument("--max-n", type=int, default=None, help="Cap downloads (smoke test)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    if not (args.discover or args.fetch):
        p.error("Pass --discover and/or --fetch")

    reg_keys = _resolve_registries(args.registry)

    if args.discover:
        entries = discover(reg_keys, args.manifest, args.snapshot_dir)
        save_manifest(entries, args.manifest)
        logger.info("Review manifest written: %s (%d rows, all pending)", args.manifest, len(entries))
        logger.info("NEXT: review %s, flip rows to `review_status: approved`, then --fetch.", args.manifest)
    else:
        entries = load_manifest(args.manifest)
        logger.info("Loaded %d rows from %s", len(entries), args.manifest)

    if args.fetch:
        entries = fetch_pdfs(entries, remote=args.remote,
                             require_approved=args.require_approved, max_n=args.max_n)
        save_manifest(entries, args.manifest)
        logger.info("Manifest updated with download metadata: %s", args.manifest)

    n_pdf = sum(1 for e in entries if e.pdf_url)
    n_appr = sum(1 for e in entries if e.review_status == "approved")
    n_dl = sum(1 for e in entries if e.pdf_local_path)
    logger.info("Done. %d rows | %d with PDF URL | %d approved | %d downloaded.",
                len(entries), n_pdf, n_appr, n_dl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
