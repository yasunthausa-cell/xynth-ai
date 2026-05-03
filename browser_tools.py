"""browser_tools.py — Web search and scraping tools for Xynth AI.

- web_search : DuckDuckGo instant API (~200ms, no browser)
- scrape_page: Playwright full browser for specific URL scraping
"""
import asyncio
import re
from typing import Optional

# ── Fast Search via DDGS (no browser needed) ─────────────────────────────────
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# ── Playwright (only for URL scraping) ───────────────────────────────────────
try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def _clean_text(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()[:max_chars]


def web_search(query: str, num_results: int = 5) -> str:
    """
    Instant web search using DuckDuckGo API — no browser launch.
    Returns top results as formatted text in ~200ms.
    """
    if not DDGS_AVAILABLE:
        return "[Web search unavailable: duckduckgo_search not installed]"
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                title   = r.get("title", "")
                body    = r.get("body", "")
                href    = r.get("href", "")
                results.append(f"**{title}**\n{href}\n{body}")
        if not results:
            return "[No results found]"
        return "\n\n".join(results)
    except Exception as e:
        return f"[Search error: {e}]"


async def _async_scrape(url: str) -> str:
    """Navigate to a URL and extract clean readable text."""
    if not PLAYWRIGHT_AVAILABLE:
        return "[Web browsing unavailable: playwright not installed]"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        page = await browser.new_page()
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"
        })
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(1500)  # Let JS render

            # Try to get article/main content, fall back to body
            for selector in ["article", "main", "[role='main']", "body"]:
                el = await page.query_selector(selector)
                if el:
                    text = await el.inner_text()
                    if len(text.strip()) > 200:
                        return _clean_text(text)
            return "[Could not extract readable text from page]"

        except PWTimeout:
            return "[Page load timed out]"
        except Exception as e:
            return f"[Scrape error: {e}]"
        finally:
            await browser.close()


# ── Synchronous wrapper for scraping only ────────────────────────────────────
def scrape_page(url: str) -> str:
    """Visit a URL with a real browser and extract its readable text."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_async_scrape(url))
        loop.close()
        return result
    except Exception as e:
        return f"[Scrape failed: {e}]"


# ── Intent detection ──────────────────────────────────────────────────────────
_SEARCH_KEYWORDS = [
    "latest", "current", "now", "today", "news", "price", "stock", "weather",
    "search", "find", "look up", "who is", "what is", "when did", "where is",
    "how much", "how many", "trending", "recent", "2024", "2025", "2026",
    "website", "visit", "browse", "open", "go to", "check",
]

def needs_web_search(message: str) -> bool:
    """Heuristic: returns True if the message likely needs live web data."""
    msg = message.lower()
    return any(kw in msg for kw in _SEARCH_KEYWORDS)


def needs_scrape(message: str) -> Optional[str]:
    """If message contains a URL, returns it. Otherwise None."""
    url_match = re.search(r'https?://[^\s]+', message)
    return url_match.group(0) if url_match else None
