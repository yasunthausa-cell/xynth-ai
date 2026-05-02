"""browser_tools.py — Playwright-based web search and scraping for Xynth AI.

Provides two main capabilities:
1. search(query) — DuckDuckGo search, returns top results as text snippets
2. scrape(url)   — Full page scrape, returns cleaned readable text
"""
import asyncio
import re
from typing import Optional

# ── Playwright async API ──────────────────────────────────────────────────────
try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def _clean_text(text: str, max_chars: int = 3000) -> str:
    """Remove excess whitespace and truncate to max_chars."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()[:max_chars]


async def _async_search(query: str, num_results: int = 4) -> str:
    """Search DuckDuckGo and return top result snippets."""
    if not PLAYWRIGHT_AVAILABLE:
        return "[Web search unavailable: playwright not installed]"

    results = []
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
            encoded = query.replace(" ", "+")
            await page.goto(
                f"https://duckduckgo.com/html/?q={encoded}",
                wait_until="domcontentloaded",
                timeout=20000
            )
            await page.wait_for_selector(".result", timeout=10000)
            items = await page.query_selector_all(".result")

            for item in items[:num_results]:
                try:
                    title_el = await item.query_selector(".result__title")
                    snippet_el = await item.query_selector(".result__snippet")
                    url_el = await item.query_selector(".result__url")

                    title = (await title_el.inner_text()).strip() if title_el else ""
                    snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""
                    url = (await url_el.inner_text()).strip() if url_el else ""

                    if title or snippet:
                        results.append(f"**{title}**\n{url}\n{snippet}")
                except Exception:
                    continue
        except PWTimeout:
            results.append("[Search timed out]")
        except Exception as e:
            results.append(f"[Search error: {e}]")
        finally:
            await browser.close()

    if not results:
        return "[No results found]"
    return "\n\n".join(results)


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


# ── Synchronous wrappers (for use in non-async Flask/Python code) ─────────────
def web_search(query: str, num_results: int = 4) -> str:
    """Synchronous wrapper for _async_search. Safe to call from any thread."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_async_search(query, num_results))
        loop.close()
        return result
    except Exception as e:
        return f"[Search failed: {e}]"


def scrape_page(url: str) -> str:
    """Synchronous wrapper for _async_scrape. Safe to call from any thread."""
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
