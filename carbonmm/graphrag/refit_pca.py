"""Refit PCA from existing raw 3072d embeddings without OpenAI calls.

Use when an sklearn version mismatch breaks the joblib pickle (e.g., an older
sklearn cannot safely unpickle an IncrementalPCA trained with a newer version).

Usage:
    python3 -m carbonmm.graphrag.refit_pca \
        --embed-dir carbonmm/data/embeddings

Config mirrors embed.fit_pca defaults used for the original fit
(sample_size=2000, seed=42, target_dim=1024).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

logger = logging.getLogger(__name__)


def refit(embed_dir: Path, target_dim: int = 1024, sample_size: int = 2000, seed: int = 42):
    raw_path = embed_dir / "clauses-3072d.npy"
    out_pca = embed_dir / f"pca-3072to{target_dim}.joblib"
    out_red = embed_dir / f"clauses-{target_dim}d.npy"

    logger.info("loading %s ...", raw_path)
    vectors = np.load(raw_path)
    n, dim = vectors.shape
    logger.info("n=%d dim=%d", n, dim)

    effective_sample = min(n, max(sample_size, target_dim + 64))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=effective_sample, replace=False)
    sample = vectors[sample_idx]
    batch_size = min(sample.shape[0], max(target_dim + 16, 256))

    pca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    pca.fit(sample)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA %dd → %dd: var explained = %.4f", dim, target_dim, var_explained)

    reduced = np.zeros((n, target_dim), dtype=np.float32)
    chunk = 1024
    for start in range(0, n, chunk):
        reduced[start : start + chunk] = pca.transform(vectors[start : start + chunk])

    joblib.dump(pca, out_pca)
    np.save(out_red, reduced)
    logger.info("wrote %s (%d bytes)", out_pca, out_pca.stat().st_size)
    logger.info("wrote %s (%d bytes)", out_red, out_red.stat().st_size)

    log_path = embed_dir / "embed-log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        log["pca_var_explained"] = var_explained
        log["pca_refit"] = True
        log_path.write_text(json.dumps(log, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path, required=True)
    ap.add_argument("--target-dim", type=int, default=1024)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    refit(args.embed_dir, args.target_dim, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
