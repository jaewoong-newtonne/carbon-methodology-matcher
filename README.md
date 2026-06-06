# Selecting Carbon-Credit Methodologies from Project Documents — Code & Data Release

*Anonymized artifact for triple-blind review.* This bundle accompanies the paper
*"Selecting Carbon-Credit Methodologies from Project Documents: Retrieval-as-Classification and the
Role of a General-Purpose Language Model."* It contains the evaluation code, the retrieve→rerank
pipeline (naive RAG and GraphRAG), the six-model significance harness, and the curated metadata needed
to reproduce the analysis. Author and affiliation information is withheld for review.

## What this is

We frame carbon-credit **methodology selection** as *retrieval-as-classification*: a retriever proposes
candidates from a label space that is itself a corpus of regulatory documents (531 methodologies,
concentrated in the three major registries — CDM, Verra/VCS, Gold Standard — plus a long tail), and a
general-purpose language model decides among them. On a clean, leakage-free benchmark of 535 Project
Design Documents (155 GS, 380 VCS) and six models (three vendors × two capability tiers), we measure
what the engine contributes: a robust, vendor-agnostic **recall** gain and a **model-dependent top-1
conversion**.

## Layout

```
carbonmm/                  importable package — run modules as `python3 -m carbonmm.<pkg>.<mod>`
  eval/        evaluation harness, clean-eval (Section-A redaction), significance + decomposition,
               guidance-KG ablation, metrics
  graphrag/    retrieve → reciprocal-rank fusion → candidate filter → LLM reranker; the six
               reranker arms; the registry-rulebook knowledge-graph hooks
  baselines/   non-LLM reference points (random, frequency prior, BM25, dense-kNN) + naive RAG
  ingest/      registry harvest, document-type classification, Section-A extraction
  redact/      Section-A redaction pattern library + clean-eval pipeline
  configs/     curated registry index YAMLs used by the harvest scripts
  data/manifests/   curated/metadata manifests (see DATA_CARD.md)
DATA_CARD.md   data provenance, source URLs, and what is / is not redistributable
requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Reranker + embedding calls use commercial vendor APIs; set the keys you intend to use:
export OPENAI_API_KEY=...      # query embeddings (text-embedding-3-large) + OpenAI reranker arms
export CLAUDE_API_KEY=...      # Claude reranker arms  (Anthropic Messages API)
export GEMINI_API_KEY=...      # Gemini reranker arms  (Google Generative Language API)
```

Run all modules from the bundle root (the directory containing `carbonmm/`) so the package
resolves, e.g. `python3 -m carbonmm.eval.harness …`.

## Reproducing the results

The evaluation is **zero-shot** (no training). Each run is on the frozen `test` split (`seed 42`) over
the leakage-free, Section-A-redacted input (`EVAL_TEXT_SOURCE=clean`). The reranker model is selected by
environment variable; results serialize to `results/clean/<model>/`.

| Paper element | Command (representative) |
|---|---|
| Build the clean (redacted) eval input | `python3 -m carbonmm.eval.build_clean_eval` |
| Main table — a Claude/Gemini arm | `EVAL_TEXT_SOURCE=clean V281_API=gemini V281_SONNET_MODEL=gemini-3.5-flash python3 -m carbonmm.eval.harness --baseline graphrag-v281-sonnet --split test --seed 42` |
| Main table — an OpenAI arm | `EVAL_TEXT_SOURCE=clean V28_OPENAI_MODEL=gpt-5.4 V28_REASONING_EFFORT=low python3 -m carbonmm.eval.harness --baseline graphrag-v281-openai --split test --seed 42` |
| Naive-RAG arm (model-controlled baseline) | `… --baseline naive-rag-sonnet …` / `… --baseline naive-rag-openai …` |
| Significance (paired McNemar) + decomposition | `python3 -m carbonmm.eval.capability_significance` ; `python3 -m carbonmm.eval.capability_decomposition` |
| Guidance-KG ablation (provenance, not accuracy) | `python3 -m carbonmm.eval.run_guidance_ablation --baseline graphrag-v281-openai --conditions baseline,all` |

Run the six models in both arms to fill the main table; the reranker is run at concurrency one and any
empty/timeout completion is regenerated (not scored). Significance is continuity-corrected McNemar on
paired per-PDD top-k correctness.

## What is NOT included (and how to rebuild it)

Source registry documents (PDD PDFs and methodology PDFs) and the full clause text / dense embeddings
derived from them are **not redistributed** (registry Terms of Service). The harvest scripts in
`carbonmm/ingest/` plus the source URLs and crawl dates in **`DATA_CARD.md`** let you reconstruct the
clause corpus and the dense index from the public registries. The curated, redistributable metadata
(evaluation splits, activity map, scale thresholds, registry attributes, methodology catalog) ships in
`carbonmm/data/manifests/`.
