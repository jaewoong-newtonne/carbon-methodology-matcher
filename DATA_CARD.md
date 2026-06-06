# Data Card

*Anonymized for triple-blind review.* This card documents the data shipped in `carbonmm/data/manifests/`,
its provenance, and what is deliberately excluded and why.

## Task and label space

The task is to rank a corpus of carbon-credit **methodologies** for a given Project Design Document
(PDD) and place the registry-recorded methodology as high as possible. The label space is itself a
corpus of regulatory documents:

- **Harvested catalog:** 594 published methodology records across ~20 registries (codes, titles,
  issuing registry, version, source URLs) — `carbonmm/data/manifests/genvision-methodology-catalog.json`.
- **Ranked label space:** 531 methodologies (those with parsed clauses available to the retriever),
  concentrated in the three major registries — CDM (≈269), Verra/VCS (≈48), Gold Standard (≈40), 357 in
  total from the majors — plus a long tail of smaller standards that act as distractors.

## Evaluation set (PDDs)

- **535 test PDDs** (155 Gold Standard, 380 Verra) + a disjoint **100-PDD validation** fold (fusion-
  weight selection only). Frozen with `seed 42`; IDs and ground-truth methodology codes are in
  `carbonmm/data/manifests/icdm2026-splits.json`.
- **Sampling/inclusion (code in `carbonmm/ingest/`):** drawn from the public project listings of the Gold
  Standard and Verra registries — CDM is excluded as a *query* source (its methodologies remain
  *candidates*) — under three jointly applied conditions: crediting-period start on/after 2020-01-01,
  **total issued credits > 0**, and successful layout parsing. A document-type classifier keeps only
  single-project ∪ component documents; a programme-of-activities dedupe retains one representative
  component per programme.
- **Why "issued credits > 0":** it drops projects whose methodology was later contested by a
  validation/verification body or that never issued credits through a methodology or MRV deficiency —
  cases whose registry-recorded methodology would be an unreliable label.
- **Labels** are the registry-recorded primary methodology codes, grounded in public issued-credit
  transaction records.

## Leakage and the clean-evaluation protocol

Because a PDD is a structured instantiation of its ground-truth methodology, the literal code leaks
into the raw text (17.2% of test PDDs; GS 43.2%, VCS 6.6%). The evaluation feeds each PDD's
project-description section (Section A) with the methodology-declaring region redacted (the redaction
code is in `carbonmm/redact/` and `carbonmm/eval/build_clean_eval.py`; audit: precision 100%, miss rate
2.9% on a stratified pilot). All reported numbers are on this redacted input.

## Included in `carbonmm/data/manifests/` (redistributable curated metadata)

| File | Contents |
|---|---|
| `icdm2026-splits.json` | validation/test PDD IDs + ground-truth methodology codes |
| `genvision-methodology-catalog.json` | harvested methodology catalog: codes, titles, registries, source URLs |
| `method-groups.json` | prediction-blind activity map (methodology code → mitigation activity group) |
| `cdm-ssc-thresholds.json` | curated small-scale (SSC) capacity thresholds + provenance |
| `g01-capacity-extracted.json` | extracted project capacity for energy-group PDDs (information-ceiling split) |
| `registry-attributes.json` | curated registry-attribute layer (region, status, …) |

## Excluded (registry Terms of Service) — and how to rebuild

Source registry documents and the text/embeddings derived from them are **not redistributed**:

- PDD and methodology source PDFs (registry-hosted);
- the full methodology **clause text** and the dense **clause embeddings**;
- PDD-level extracted text/features.

These are reconstructible from the public registries using the harvest scripts in `carbonmm/ingest/`
(registry listings + a public methodology-aggregation CDN; URLs and crawl metadata are recorded inside
each manifest and in the catalog). Rebuild order: harvest the catalog → download methodology/PDD
documents → parse to clauses → embed (`text-embedding-3-large`) → build the BM25 + dense index.

## Source registries

- Verra Registry (Verified Carbon Standard)
- Gold Standard Impact Registry
- Clean Development Mechanism (CDM / UNFCCC) — methodology candidates only

Crawl dates are recorded in each manifest's metadata. No personal data is collected; all labels are
public registry-issued facts.
