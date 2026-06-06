"""OpenAI text-embedding-3-large + PCA→1024d for methodology-corpus clauses.

Usage:
    OPENAI_API_KEY=... python3 -m carbonmm.graphrag.embed \\
        [--out-dir carbonmm/data/embeddings] \\
        [--batch 96] [--cost-cap-usd 15] [--pca-dim 1024]

Outputs:
    {out_dir}/clauses-meta.parquet         row-aligned metadata (one row per clause)
    {out_dir}/clauses-3072d.npy            raw text-embedding-3-large vectors
    {out_dir}/clauses-1024d.npy            PCA-reduced (default 1024d)
    {out_dir}/pca-3072to1024.joblib        fitted PCA model (for query-time projection)
    {out_dir}/embed-log.json               run metadata (model, count, tokens, $)

Uses OpenAI for embedding generation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openai import OpenAI

from .clause_io import iter_all_clauses, CORPUS_ROOT

logger = logging.getLogger(__name__)

MODEL = "text-embedding-3-large"
MODEL_DIM = 3072
COST_PER_MTOKEN_USD = 0.13  # text-embedding-3-large pricing as of 2026
MAX_TOKENS_PER_INPUT = 7_500  # under 8,192 hard cap, leaves headroom for sep tokens


def truncate_tokens(text: str, max_tokens: int, enc) -> str:
    """Truncate text to <= max_tokens by tiktoken encoding."""
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def load_openai_key() -> str:
    """Find an OpenAI key — environment variable first, then a local ``.env`` file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    candidates = [Path(".env"), Path("backend") / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "OPENAI_API_KEY not in env or .env. "
        "Set OPENAI_API_KEY before running."
    )


def chunked(seq, n):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def embed_batched(
    client: OpenAI,
    texts: list[str],
    batch_size: int,
    cost_cap_usd: float,
    max_retries: int = 5,
) -> tuple[np.ndarray, dict]:
    """Embed `texts` via OpenAI in batches. Returns (vectors, stats)."""
    n = len(texts)
    out = np.zeros((n, MODEL_DIM), dtype=np.float32)
    tokens_used = 0
    cost_usd = 0.0
    t0 = time.time()

    for batch_idx, batch_rows in enumerate(chunked(enumerate(texts), batch_size)):
        idxs = [i for i, _ in batch_rows]
        batch_texts = [t for _, t in batch_rows]

        # Budget guard
        if cost_usd >= cost_cap_usd:
            raise RuntimeError(
                f"cost cap ${cost_cap_usd:.2f} exceeded (used ${cost_usd:.2f}); aborting"
            )

        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=MODEL, input=batch_texts)
                break
            except Exception as e:
                wait = 2**attempt
                logger.warning(
                    "batch %d attempt %d failed: %s — retry in %ds", batch_idx, attempt + 1, e, wait
                )
                time.sleep(wait)
        else:
            raise RuntimeError(f"batch {batch_idx} exhausted retries")

        for j, item in enumerate(resp.data):
            out[idxs[j]] = item.embedding
        tokens_used += resp.usage.total_tokens
        cost_usd = tokens_used * COST_PER_MTOKEN_USD / 1_000_000

        if batch_idx % 5 == 0:
            elapsed = time.time() - t0
            logger.info(
                "batch %d: %d/%d clauses · %.0f tok · $%.4f · %.1fs",
                batch_idx,
                min((batch_idx + 1) * batch_size, n),
                n,
                tokens_used,
                cost_usd,
                elapsed,
            )

    return out, {
        "n_clauses": n,
        "tokens": tokens_used,
        "cost_usd": round(cost_usd, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "model": MODEL,
    }


def fit_pca(
    vectors: np.ndarray, target_dim: int, sample_size: int = 1000, seed: int = 42
) -> tuple[np.ndarray, "object"]:
    """Fit IncrementalPCA on a random sample, transform the full matrix.

    Note: n_samples must be >= n_components for IncrementalPCA to estimate
    that many components, so we auto-expand sample_size to at least
    target_dim + 64.
    """
    from sklearn.decomposition import IncrementalPCA

    n = vectors.shape[0]
    effective_sample = min(n, max(sample_size, target_dim + 64))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=effective_sample, replace=False)
    sample = vectors[sample_idx]

    # IncrementalPCA also requires batch_size >= n_components
    batch_size = min(sample.shape[0], max(target_dim + 16, 256))
    pca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    pca.fit(sample)

    reduced = np.zeros((n, target_dim), dtype=np.float32)
    chunk = 1024
    for start in range(0, n, chunk):
        reduced[start : start + chunk] = pca.transform(vectors[start : start + chunk])

    var_explained = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA %dd → %dd: var explained = %.3f", vectors.shape[1], target_dim, var_explained)
    return reduced, pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default="carbonmm/data/embeddings", type=Path
    )
    ap.add_argument("--corpus-root", default=CORPUS_ROOT, type=Path)
    ap.add_argument("--batch", default=96, type=int)
    ap.add_argument("--cost-cap-usd", default=15.0, type=float)
    ap.add_argument("--pca-dim", default=1024, type=int)
    ap.add_argument("--pca-sample-size", default=1000, type=int)
    ap.add_argument("--dry-run", action="store_true", help="extract clauses, skip embedding")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "clauses-meta.parquet"
    raw_path = args.out_dir / "clauses-3072d.npy"
    red_path = args.out_dir / f"clauses-{args.pca_dim}d.npy"
    pca_path = args.out_dir / f"pca-3072to{args.pca_dim}.joblib"
    log_path = args.out_dir / "embed-log.json"

    logger.info("loading clauses from %s …", args.corpus_root)
    clauses = list(iter_all_clauses(args.corpus_root))
    logger.info("loaded %d clauses across %d methodologies", len(clauses), len({c.code for c in clauses}))

    # Persist meta first (row-aligned with embeddings below)
    pd.DataFrame([c.to_dict() for c in clauses]).to_parquet(meta_path, index=False)
    logger.info("wrote meta: %s", meta_path)

    if args.dry_run:
        # Per-clause-type breakdown for inspection
        df = pd.DataFrame([c.to_dict() for c in clauses])
        logger.info("by clause_type:\n%s", df["clause_type"].value_counts().to_string())
        logger.info("by registry:\n%s", df["registry"].value_counts().to_string())
        return

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # used by text-embedding-3-*
    texts = []
    truncated = 0
    for c in clauses:
        t = truncate_tokens(c.clause_text, MAX_TOKENS_PER_INPUT, enc)
        if len(t) < len(c.clause_text):
            truncated += 1
        texts.append(t)
    if truncated:
        logger.info("truncated %d clauses to <=%d tokens", truncated, MAX_TOKENS_PER_INPUT)
    key = load_openai_key()
    client = OpenAI(api_key=key)

    vectors, stats = embed_batched(client, texts, args.batch, args.cost_cap_usd)
    np.save(raw_path, vectors)
    logger.info("wrote raw vectors: %s · shape=%s · $%.4f", raw_path, vectors.shape, stats["cost_usd"])

    reduced, pca = fit_pca(vectors, args.pca_dim, sample_size=args.pca_sample_size)
    np.save(red_path, reduced)
    import joblib
    joblib.dump(pca, pca_path)
    logger.info("wrote PCA: %s · shape=%s", red_path, reduced.shape)

    log = {
        **stats,
        "pca_dim": args.pca_dim,
        "pca_var_explained": float(np.sum(pca.explained_variance_ratio_)),
        "pca_sample_size": args.pca_sample_size,
    }
    log_path.write_text(json.dumps(log, indent=2))
    logger.info("done · log: %s", log_path)


if __name__ == "__main__":
    main()
