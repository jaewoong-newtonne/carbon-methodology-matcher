"""Baseline 5: naive RAG — hybrid retrieve top-30 clauses, then Claude
daemon ranks top-5 methodology codes. NO graph-constraint filter (that's
the GraphRAG arm).

This is the most direct comparator for the GraphRAG paper: same retrieval
front-end as our system, same LLM generator, only the graph filter is
ablated away. The §G ablation table will isolate the graph-filter gain.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import httpx

from .base import Baseline, label_space

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 120.0
DEFAULT_MODEL = "claude-haiku-4-5"

# Regex matching valid methodology codes — same set as redact.patterns.
_CODE_RE = re.compile(
    r"\b("
    r"ACM\d{4}|AMS-[IVX]+(?:\.[A-Z]{1,3})?(?:\.[a-z]+)?\.|AM\d{4}|"
    r"AR-AM\d{4}|AR-AMS-[IVX]+(?:\.[A-Z])?|"
    r"VM\d{4}|VMD\d{4}|VMR\d{4}|"
    r"TPDDTEC|GS-[A-Z]{2}-\d{3}|GS\d{1,5}|\d{3}(?:-\d{1,2})?"
    r")\b"
)

PROMPT_TEMPLATE = """You are a carbon-credit methodology classifier.

A carbon-project's design document is described below. You are given the
text of the project plus the most-relevant clauses from a corpus of registered
carbon-credit methodologies (Gold Standard / CDM / Verra etc.). Your task:
rank the FIVE methodology codes that are most likely to apply to this project,
based on alignment between the project description and the methodology's
applicability and baseline clauses.

# Project text (truncated)

{project_text}

# Candidate methodology clauses (retrieved)

{candidate_clauses}

# Instructions

Output ONLY a JSON array of EXACTLY 5 objects with the schema:

  [
    {{"code": "ACM0002", "rationale": "<1 sentence>"}},
    {{"code": "AMS-I.D.", "rationale": "..."}}
  ]

Use codes EXACTLY as they appear in the candidate clauses above. Do not
invent codes that are not in the candidate list. Order from most to least
likely.
"""


def _build_candidate_block(hits: list[tuple[dict, float]], max_chars: int = 8_000) -> str:
    """Format retrieved clauses as compact text for the LLM prompt."""
    # Group by code, dedupe types, cap total length.
    by_code: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row, _score in hits:
        by_code[row["code"]].append((row["clause_type"], row["clause_text"]))

    lines = []
    used = 0
    for code, items in by_code.items():
        for clause_type, text in items[:2]:  # max 2 clauses per code
            snippet = text[:400].replace("\n", " ")
            line = f"- [{code}] ({clause_type}) {snippet}"
            if used + len(line) > max_chars:
                return "\n".join(lines) + "\n- ... (truncated)"
            lines.append(line)
            used += len(line)
    return "\n".join(lines)


def _extract_json_array(text: str) -> list[dict]:
    """Tolerate code fences + leading prose; return the first balanced JSON array."""
    # Try fenced block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Balanced-bracket scan
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return []
    return []


class NaiveRAGBaseline(Baseline):
    name = "naive-rag"

    def __init__(self, retriever=None, model: str = DEFAULT_MODEL, daemon_url: str = DAEMON_URL):
        if retriever is None:
            import importlib
            mod = importlib.import_module("carbonmm.graphrag.retrieve")
            retriever = mod.HybridRetriever.load_default()
        self.R = retriever
        self.model = model
        self.daemon_url = daemon_url
        self.valid_codes = set(label_space())
        self.client = httpx.Client(timeout=DAEMON_TIMEOUT)

    def _call_daemon(self, prompt: str) -> str:
        resp = self.client.post(
            self.daemon_url,
            json={
                "prompt": prompt,
                "model": self.model,
                "timeout_s": min(DAEMON_TIMEOUT - 5.0, 90.0),
                "use_cache": True,
            },
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            hits = self.R.query(pdd_text, top_k=30)
        except Exception as e:
            logger.warning("retriever failed: %s", e)
            return []
        if not hits:
            return []

        candidate_block = _build_candidate_block(hits)
        project_snippet = pdd_text[:6_000]
        prompt = PROMPT_TEMPLATE.format(
            project_text=project_snippet, candidate_clauses=candidate_block
        )

        try:
            text = self._call_daemon(prompt)
        except Exception as e:
            logger.warning("daemon call failed: %s", e)
            return []

        parsed = _extract_json_array(text)
        out: list[tuple[str, float]] = []
        seen = set()
        for rank, item in enumerate(parsed, 1):
            code = (item or {}).get("code")
            if not code or code in seen:
                continue
            if code not in self.valid_codes:
                logger.debug("naive-rag hallucinated code: %s", code)
                continue
            score = 1.0 / rank  # higher = better
            out.append((code, score))
            seen.add(code)
            if len(out) >= top_k:
                break

        # If response was mangled / empty, fall back to top hybrid hits' codes.
        if not out:
            logger.info("naive-rag: empty/invalid LLM response, falling back to retrieval order")
            by_code_score: dict[str, float] = {}
            for row, score in hits:
                by_code_score[row["code"]] = max(by_code_score.get(row["code"], 0), score)
            for code, score in sorted(by_code_score.items(), key=lambda x: -x[1])[:top_k]:
                out.append((code, score))
        return out
