"""Methodology corpus ingestion — scrapers + section extractor + KG loader.

LLM inference (if needed for section identification or title canonicalization)
routes through the configured inference transport.

Scraped PDFs are stored under a local data root at
`$DATA_ROOT/methodology-corpus/{registry}/{code}/`.
"""
