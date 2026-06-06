"""Playwright-based HTML fetcher for WAF-protected registry sites.

UNFCCC (Incapsula), Verra, and (some) Gold Standard pages fingerprint stock
HTTP clients (httpx/curl) at the TLS layer. A real browser bypasses this.

Lazy-imports Playwright so the module is importable in dev envs without it.
Install: `pip install playwright && playwright install chromium`.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


_STEALTH_INIT = """
// Minimal anti-fingerprinting patches to evade common WAF bot checks.
// (Imperva/Incapsula, Cloudflare, etc. probe these in JS.)
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        { name: 'Chrome PDF Plugin' },
        { name: 'Chrome PDF Viewer' },
        { name: 'Native Client' },
    ],
});
window.chrome = { runtime: {} };
// Permissions query override (some bot-detectors call this)
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters)
);
"""


@contextmanager
def browser_context(*, headless: bool = True, slow_mo_ms: int = 0) -> Iterator:
    """Yield a Playwright browser context with minimal stealth patches.

    Usage:
        with browser_context() as ctx:
            page = ctx.new_page()
            page.goto(url)
            html = page.content()
    """
    from playwright.sync_api import sync_playwright

    p = sync_playwright().start()
    try:
        browser = p.chromium.launch(
            headless=headless,
            slow_mo=slow_mo_ms,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        try:
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1280, "height": 900},
            )
            ctx.add_init_script(_STEALTH_INIT)
            try:
                yield ctx
            finally:
                ctx.close()
        finally:
            browser.close()
    finally:
        p.stop()


def fetch_html(
    url: str,
    *,
    wait_selector: Optional[str] = None,
    wait_seconds: float = 3.0,
    wait_until: str = "domcontentloaded",
    headless: bool = True,
    timeout_ms: int = 60_000,
) -> str:
    """Fetch a URL via headless Chromium and return rendered HTML.

    `wait_selector`: CSS selector to wait for before returning (for JS-rendered
        content). If None, just waits `wait_seconds` after `wait_until` event.
    `wait_until`: 'load' | 'domcontentloaded' | 'networkidle'. Prefer
        'domcontentloaded' for sites with WAF JS-challenges (e.g. Incapsula)
        that keep the network non-idle.

    Auto-detects Incapsula challenge pages (size <2KB + 'Incapsula' marker)
    and logs a warning.
    """
    logger.info("Playwright fetch: %s (wait_until=%s)", url, wait_until)
    with browser_context(headless=headless) as ctx:
        page = ctx.new_page()
        page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        if wait_selector:
            try:
                page.wait_for_selector(wait_selector, timeout=30_000)
            except Exception as e:
                logger.warning("wait_for_selector %r failed: %s", wait_selector, e)
        else:
            time.sleep(wait_seconds)
        html = page.content()
    logger.info("  → %d bytes", len(html))
    if len(html) < 2000 and ("Incapsula" in html or "incap_ses" in html):
        logger.warning("Likely Incapsula challenge page (WAF). Try: increase wait, run headed, or use snapshot.")
    return html


def fetch_many(
    urls: list[str],
    *,
    wait_selector: Optional[str] = None,
    wait_until: str = "domcontentloaded",
    polite_seconds: float = 1.0,
    wait_seconds: float = 3.0,
) -> dict[str, str]:
    """Fetch multiple URLs sequentially in one browser session (cheaper)."""
    results: dict[str, str] = {}
    with browser_context() as ctx:
        page = ctx.new_page()
        for url in urls:
            try:
                page.goto(url, wait_until=wait_until, timeout=60_000)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=30_000)
                    except Exception:
                        time.sleep(wait_seconds)
                else:
                    time.sleep(wait_seconds)
                results[url] = page.content()
                logger.info("  %s → %d bytes", url, len(results[url]))
            except Exception as e:
                logger.error("  %s → ERROR: %s", url, e)
                results[url] = ""
            time.sleep(polite_seconds)
    return results
