"""Shared agent setup used by both the CLI (app.py) and the HTTP API (api.py)."""
import os
from dotenv import load_dotenv

load_dotenv()
import re
import time
import datetime
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq

try:
    from langchain_openai import (
        ChatOpenAI,
    )  # for Qwen via DashScope OpenAI-compatible endpoint
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from ddgs import DDGS  # newer package (preferred)
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore

import scheduler as _sched
import messaging as _msg


_browser_state = {"playwright": None, "browser": None, "page": None}


def _ensure_browser():
    from playwright.sync_api import sync_playwright

    if _browser_state["page"] is None:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        page = context.new_page()
        _browser_state["playwright"] = pw
        _browser_state["browser"] = browser
        _browser_state["page"] = page
    return _browser_state["page"]


def close_browser():
    if _browser_state["browser"]:
        try:
            _browser_state["browser"].close()
            _browser_state["playwright"].stop()
        except Exception:
            pass
        _browser_state["browser"] = None
        _browser_state["page"] = None
        _browser_state["playwright"] = None


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information, news, prices, products, or to find URLs. Returns a numbered list of titles, URLs, and short snippets. Use ONCE per question, then scrape_website on the most promising URL.

    Args:
        query: What to search for.
        max_results: How many results to return (default 5, max 10).
    """
    import time as _time

    max_results = max(1, min(int(max_results or 5), 8))
    last_err = None
    for attempt in range(3):
        try:
            with DDGS(timeout=10) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                lines = []
                for i, r in enumerate(results, 1):
                    title = (r.get("title") or "")[:120]
                    href = r.get("href") or r.get("url") or ""
                    body = (r.get("body") or "")[:160]
                    lines.append(f"{i}. {title}\n   {href}\n   {body}")
                return "\n\n".join(lines)
            last_err = "no results"
        except Exception as e:
            last_err = str(e)
        _time.sleep(1.2 * (attempt + 1))
    return f"Search failed after retries: {last_err}"


_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@tool
def scrape_website(url: str) -> str:
    """Quickly fetch and extract readable text from a static website URL. Best for articles, docs, pricing pages. For dynamic sites or login flows use browser_open."""
    try:
        response = requests.get(
            url, headers=_DEFAULT_HEADERS, timeout=12, allow_redirects=True
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.extract()
        text = " ".join(soup.stripped_strings)
        return text[:3000] if text else "(page returned no extractable text)"
    except requests.exceptions.Timeout:
        return f"Failed to scrape {url}: request timed out after 12s. Try browser_open instead."
    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"


@tool
def execute_python_code(code: str) -> str:
    """Execute Python code and return stdout. Use for math, string manipulation, or data processing."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code, {"__builtins__": __builtins__}, {})
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout


@tool
def save_text_to_file(filename: str, content: str) -> str:
    """Save text content to a local file."""
    try:
        dir_name = os.path.dirname(os.path.abspath(filename))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved content to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


@tool
def read_file(filename: str) -> str:
    """Read the contents of any local file (source code, config, logs, etc.). Use this to inspect your own code, .env secrets, requirements.txt, or any other file before editing.

    Args:
        filename: Absolute or relative path to the file.
    """
    try:
        path = os.path.abspath(filename)
        if not os.path.exists(path):
            return f"File not found: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.splitlines()
        if len(lines) > 300:
            content = "\n".join(lines[:300]) + f"\n... (truncated, {len(lines)} total lines)"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def edit_secret(key: str, value: str) -> str:
    """Add or update a secret/environment variable in the .env file AND apply it to the current process immediately. Use this when the user provides a new API key, or when you discover a missing credential needed for a task.

    Args:
        key: The environment variable name (e.g. 'GROQ_API_KEY').
        value: The value to set.
    """
    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        # Read existing lines
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        # Update or append
        key_found = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)
        if not key_found:
            new_lines.append(f"{key}={value}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        # Apply to current process immediately
        os.environ[key] = value
        return f"✅ Secret '{key}' saved to .env and applied to current session."
    except Exception as e:
        return f"Error editing secret: {str(e)}"


@tool
def install_package(package_name: str, alternatives: list = None) -> str:
    """Install a missing Python package at runtime using pip. Automatically retries with alternative package names if the first attempt fails. Use this whenever a tool fails with an ImportError or ModuleNotFoundError.

    Args:
        package_name: The primary pip package name to install (e.g. 'playwright', 'openai').
        alternatives: Optional list of alternative package names to try if the primary fails (e.g. ['ddgs', 'duckduckgo-search']).
    """
    import subprocess, sys

    candidates = [package_name] + (alternatives or [])
    last_err = ""
    for pkg in candidates:
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    # Verify it's actually importable
                    import importlib
                    mod_name = pkg.replace("-", "_").split("[")[0]
                    try:
                        importlib.import_module(mod_name)
                    except ImportError:
                        pass  # Some packages have different import names — still count as success
                    return f"✅ Successfully installed '{pkg}'. You can now use it."
                last_err = result.stderr.strip()[-300:] if result.stderr else "unknown error"
            except subprocess.TimeoutExpired:
                last_err = f"pip install timed out for '{pkg}' (attempt {attempt+1})"
            except Exception as e:
                last_err = str(e)
    return f"❌ Failed to install '{package_name}' (tried: {candidates}). Last error: {last_err}"


@tool
def call_api(url: str, method: str = "GET", payload: dict = None) -> str:
    """Make an HTTP request to an API and return the response."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        try:
            return str(response.json())
        except Exception:
            return response.text[:4000]
    except Exception as e:
        return f"API Call failed: {str(e)}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP. Requires EMAIL_ADDRESS and EMAIL_APP_PASSWORD secrets."""
    sender = os.environ.get("EMAIL_ADDRESS")
    password = os.environ.get("EMAIL_APP_PASSWORD")
    if not sender or not password:
        return "Email not configured. EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set."
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender, password)
            server.sendmail(sender, [to], msg.as_string())
        return f"Email sent successfully to {to}"
    except Exception as e:
        return f"Failed to send email: {str(e)}"


@tool
def browser_open(url: str) -> str:
    """Open a URL in a real headless Chromium browser (executes JavaScript). Returns visible text. Use for dynamic sites and SPAs."""
    try:
        page = _ensure_browser()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)
        text = page.evaluate("() => document.body.innerText")
        return f"Opened {url}\nTitle: {page.title()}\n---\n{text[:5000]}"
    except Exception as e:
        return f"Failed to open {url}: {str(e)}"


@tool
def browser_click(selector: str) -> str:
    """Click an element in the current browser page using a CSS selector."""
    try:
        page = _ensure_browser()
        page.click(selector, timeout=10000)
        page.wait_for_timeout(1500)
        return f"Clicked '{selector}'. Now at: {page.url}"
    except Exception as e:
        return f"Failed to click '{selector}': {str(e)}"


@tool
def browser_type(selector: str, text: str) -> str:
    """Type text into an input field using a CSS selector."""
    try:
        page = _ensure_browser()
        page.fill(selector, text, timeout=10000)
        return f"Typed into '{selector}'."
    except Exception as e:
        return f"Failed to type into '{selector}': {str(e)}"


@tool
def browser_get_html() -> str:
    """Return the current page's HTML (truncated). Use to inspect form fields and structure."""
    try:
        page = _ensure_browser()
        html = page.content()
        return html[:8000]
    except Exception as e:
        return f"Failed to get HTML: {str(e)}"


@tool
def analyze_webpage_visually(url: str, question: str, full_page: bool = False) -> str:
    """Take a screenshot of a webpage and use a vision AI to answer questions about how it LOOKS — visual design, aesthetics, layout, colors, branding, what's on screen, etc. Use this whenever the user asks about the appearance/design/beauty of a site, or wants to know what's visible on a page.

    Args:
        url: The URL to visit and screenshot.
        question: Specific question about the page's visuals (e.g. "Is this site modern and beautiful, or dated? Justify.", "Describe the hero section.", "What products are on screen?").
        full_page: If True, capture the entire scrollable page. If False (default), just the viewport.
    """
    try:
        import base64
        from groq import Groq

        page = _ensure_browser()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2500)
        screenshot_bytes = page.screenshot(full_page=full_page)
        b64 = base64.b64encode(screenshot_bytes).decode()

        client = Groq()
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"You are evaluating a webpage screenshot from {url}. Answer this question concretely and with specifics from what you see: {question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=900,
            temperature=0.2,
        )
        return resp.choices[0].message.content or "(no analysis)"
    except Exception as e:
        return f"Failed to visually analyze {url}: {str(e)}"


@tool
def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """Generate an AI image from a text prompt. Returns a public HTTPS URL of the generated image.
    
    Args:
        prompt: Detailed text description of the image to create.
        width: Image width in pixels. Default 1024.
        height: Image height in pixels. Default 1024.
    """
    import urllib.parse
    try:
        # We use pollinations.ai since it's fast, free, requires no API key, and doesn't timeout like DashScope.
        prompt_encoded = urllib.parse.quote(prompt)
        # Using a deterministic seed based on prompt length just for fun, or we can just let it be random.
        url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width={width}&height={height}&nologo=true"
        return url
    except Exception as e:
        return f"Image generation error: {str(e)}"


@tool
def send_whatsapp_image(recipient: str, image_url: str, caption: str = "") -> str:
    """Send an image to a WhatsApp recipient via the Meta Cloud API. The image_url must be a publicly accessible HTTPS URL (jpg/png/gif/webp). Use this to deliver images you generated, screenshots you took, or images you found online.

    Args:
        recipient: WhatsApp phone number WITHOUT the 'whatsapp:' prefix, just digits e.g. '94771234567'. Use the current user's number unless told otherwise.
        image_url: Public HTTPS URL of the image.
        caption: Optional caption text to accompany the image.
    """
    ok = _msg.send_image(recipient, image_url, caption)
    return (
        "✅ Image sent."
        if ok
        else "❌ Failed to send image (check META_WA_ACCESS_TOKEN and that the URL is public HTTPS)."
    )


@tool
def screenshot_and_send(
    recipient: str, url: str, caption: str = "", full_page: bool = False
) -> str:
    """Take a screenshot of a webpage and send it to the user on WhatsApp. Use when the user asks "show me what X looks like" or wants a visual of a website.

    Args:
        recipient: WhatsApp phone number, digits only e.g. '94771234567'. Use the current user's number.
        url: The webpage to screenshot.
        caption: Optional caption text.
        full_page: If True capture the full scrollable page. Default False (viewport only).
    """
    import time, uuid

    try:
        page = _ensure_browser()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
        static_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static_media"
        )
        os.makedirs(static_dir, exist_ok=True)
        filename = f"shot_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.join(static_dir, filename)
        page.screenshot(path=path, full_page=full_page)

        # Support Railway, Render, Replit or any custom domain
        host = (
            os.environ.get("RAILWAY_PUBLIC_DOMAIN")
            or os.environ.get("RENDER_EXTERNAL_URL", "").replace("https://", "")
            or os.environ.get("REPLIT_DEV_DOMAIN")
            or os.environ.get("PUBLIC_HOST")
        )
        if not host:
            return (
                f"Screenshot saved locally to {path}, but no public domain env var is set "
                f"(set RAILWAY_PUBLIC_DOMAIN or PUBLIC_HOST). Cannot send via WhatsApp."
            )
        public_url = f"https://{host}/media/{filename}"
        ok = _msg.send_image(recipient, public_url, caption)
        return (
            f"✅ Screenshot of {url} sent to {recipient}."
            if ok
            else f"❌ Failed to send screenshot. Public URL was: {public_url}"
        )
    except Exception as e:
        return f"Failed to screenshot {url}: {str(e)}"


@tool
def schedule_recurring_task(
    prompt: str,
    recipient: str,
    hour: int,
    minute: int = 0,
    day_of_week: str = "*",
    timezone: str = "UTC",
) -> str:
    """Schedule a RECURRING task that runs at the same time every day or selected days. The agent will execute `prompt` at the scheduled time and send the answer to `recipient` via WhatsApp.

    Args:
        prompt: What to do at run time (e.g. "search the web for today's top tech news and summarise the top 3 stories").
        recipient: WhatsApp number including 'whatsapp:' prefix (e.g. 'whatsapp:+14155551234'). Use the user's own number unless they specify someone else.
        hour: Hour of day, 0-23, IN THE GIVEN TIMEZONE.
        minute: Minute, 0-59. Default 0.
        day_of_week: Cron day-of-week. '*' = every day (default). 'mon-fri' = weekdays. 'sat,sun' = weekends. Single days: 'mon','tue','wed','thu','fri','sat','sun'.
        timezone: IANA timezone name like 'UTC', 'Asia/Colombo', 'America/New_York', 'Europe/London'. Default 'UTC'. ALWAYS ask the user for their timezone if not obvious from context.
    """
    return _sched.schedule_task(prompt, recipient, hour, minute, day_of_week, timezone)


@tool
def schedule_one_time_task(
    prompt: str, recipient: str, run_at_iso: str, timezone: str = "UTC"
) -> str:
    """Schedule a ONE-TIME task that runs once at a specific date/time, then is removed.

    Args:
        prompt: What to do.
        recipient: WhatsApp number with 'whatsapp:' prefix.
        run_at_iso: ISO 8601 date/time without timezone, e.g. '2026-05-01T08:30'.
        timezone: IANA timezone of the run_at_iso value. Default 'UTC'.
    """
    return _sched.schedule_one_time_task(prompt, recipient, run_at_iso, timezone)


@tool
def list_scheduled_tasks(recipient_filter: str = "") -> str:
    """List all scheduled tasks. Optionally filter by recipient WhatsApp number (with 'whatsapp:' prefix). Returns each task's job_id, next run time, recipient, and prompt."""
    return _sched.list_tasks(recipient_filter or None)


@tool
def cancel_scheduled_task(job_id: str) -> str:
    """Cancel/delete a scheduled task by its job_id (obtained from list_scheduled_tasks or the confirmation when scheduled)."""
    return _sched.cancel_task(job_id)


@tool
def remember_user_fact(session_id: str, fact: str) -> str:
    """Save an important fact about the user to long-term memory so you remember it in future conversations.
    Use this whenever the user shares personal preferences, their name, location, job, or anything they'd want remembered.

    Args:
        session_id: The current session/conversation ID. This is provided in the system context.
        fact: A concise statement about the user, e.g. 'User's name is Yasun', 'User prefers metric units', 'User lives in Colombo'.
    """
    try:
        import os, requests as _rq
        base = os.environ.get("SELF_BASE_URL", "http://localhost:5000")
        r = _rq.post(f"{base}/memory", json={"session_id": session_id, "fact": fact}, timeout=5)
        if r.ok:
            return f"✅ Remembered: {fact}"
        return f"⚠️ Could not save memory: {r.text}"
    except Exception as e:
        return f"⚠️ Memory save failed: {e}"


@tool
def get_weather(location: str) -> str:
    """Get the current weather and today's forecast for any city or location.

    Args:
        location: City name or location, e.g. 'Colombo', 'London', 'New York'.
    """
    try:
        import urllib.parse
        loc_encoded = urllib.parse.quote(location)
        # Open-Meteo: free, no API key needed — geocode first
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={loc_encoded}&count=1&format=json",
            timeout=10
        ).json()
        results = geo.get("results")
        if not results:
            return f"Could not find location: {location}"
        r = results[0]
        lat, lon, name, country = r["latitude"], r["longitude"], r["name"], r.get("country", "")
        wx = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,apparent_temperature"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=1",
            timeout=10
        ).json()
        cur = wx.get("current", {})
        daily = wx.get("daily", {})
        temp = cur.get("temperature_2m", "?")
        feels = cur.get("apparent_temperature", "?")
        humidity = cur.get("relative_humidity_2m", "?")
        wind = cur.get("wind_speed_10m", "?")
        high = (daily.get("temperature_2m_max") or [None])[0]
        low = (daily.get("temperature_2m_min") or [None])[0]
        rain = (daily.get("precipitation_sum") or [None])[0]
        units = wx.get("current_units", {})
        t_unit = units.get("temperature_2m", "°C")
        return (
            f"Weather in {name}, {country}:\n"
            f"🌡️ {temp}{t_unit} (feels like {feels}{t_unit})\n"
            f"💧 Humidity: {humidity}%  💨 Wind: {wind} km/h\n"
            f"📊 Today: High {high}{t_unit} / Low {low}{t_unit}\n"
            f"🌧️ Precipitation: {rain} mm"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


@tool
def get_price(asset: str) -> str:
    """Get the live price of a cryptocurrency (BTC, ETH, etc.) or a US stock ticker (AAPL, TSLA, etc.).

    Args:
        asset: Ticker symbol or name, e.g. 'BTC', 'bitcoin', 'AAPL', 'TSLA', 'ETH'.
    """
    try:
        a = asset.strip().upper()
        # Try crypto first via CoinGecko (free, no key)
        crypto_ids = {"BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
                      "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin",
                      "ADA": "cardano", "AVAX": "avalanche-2", "DOT": "polkadot"}
        cg_id = crypto_ids.get(a, a.lower())
        cg = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_change=true",
            timeout=8
        ).json()
        if cg_id in cg:
            price = cg[cg_id]["usd"]
            change = cg[cg_id].get("usd_24h_change", 0)
            arrow = "📈" if change >= 0 else "📉"
            return f"{arrow} {a}: ${price:,.2f} USD ({change:+.2f}% 24h)"
        # Fall back to Yahoo Finance for stocks
        yf = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{a}?interval=1d&range=1d",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        ).json()
        meta = yf["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        prev = meta.get("previousClose", price)
        change_pct = ((price - prev) / prev * 100) if prev else 0
        currency = meta.get("currency", "USD")
        arrow = "📈" if change_pct >= 0 else "📉"
        return f"{arrow} {a}: {price:,.2f} {currency} ({change_pct:+.2f}% today)"
    except Exception as e:
        return f"Price lookup failed for '{asset}': {e}"


@tool
def summarize_youtube(url: str) -> str:
    """Fetch and summarize a YouTube video by extracting its transcript/subtitles.

    Args:
        url: Full YouTube video URL, e.g. 'https://www.youtube.com/watch?v=abc123'
    """
    try:
        import re
        # Extract video ID
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            return "Could not extract video ID from URL."
        vid_id = match.group(1)
        # Try youtube-transcript-api if available
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(vid_id)
            text = " ".join(t["text"] for t in transcript_list)[:6000]
            return f"[YouTube transcript for {url}]\n\n{text}\n\n(Summarize the above transcript in a clear, structured way.)"
        except Exception:
            pass
        # Fallback: scrape the page for description/title
        page = requests.get(
            f"https://www.youtube.com/watch?v={vid_id}",
            headers=_DEFAULT_HEADERS, timeout=12
        ).text
        soup = BeautifulSoup(page, "html.parser")
        title = soup.find("title")
        title_text = title.get_text() if title else "Unknown title"
        # Extract description from og:description
        desc = soup.find("meta", attrs={"name": "description"})
        desc_text = desc["content"] if desc else "No description found."
        return (
            f"Could not fetch transcript directly. Here's what I know from the page:\n"
            f"Title: {title_text}\nDescription: {desc_text}\n\n"
            f"Try installing youtube-transcript-api for full transcripts: pip install youtube-transcript-api"
        )
    except Exception as e:
        return f"YouTube summarization failed: {e}"


@tool
def translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
    """Translate text from one language to another using a free translation API.

    Args:
        text: The text to translate.
        target_language: Target language code or name, e.g. 'es', 'fr', 'Sinhala', 'Japanese', 'Arabic'.
        source_language: Source language code or 'auto' for automatic detection. Default 'auto'.
    """
    try:
        import urllib.parse
        # Use MyMemory free API (no key required, 5000 chars/day)
        params = {
            "q": text[:500],
            "langpair": f"{source_language}|{target_language}",
        }
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params=params, timeout=10
        ).json()
        translated = r.get("responseData", {}).get("translatedText", "")
        if translated and translated != text:
            return f"Translation ({source_language} → {target_language}):\n{translated}"
        return f"Translation failed or no change detected. Raw response: {r.get('responseStatus', '')}"
    except Exception as e:
        return f"Translation failed: {e}"


@tool
def generate_qr_code(content: str, label: str = "") -> str:
    """Generate a QR code image for any URL, text, phone number, or WiFi credentials.
    Returns a public URL to the QR code image that can be shared.

    Args:
        content: What to encode — URL, plain text, phone number, WiFi credentials, etc.
        label: Optional label/caption to show below the QR code.
    """
    try:
        import urllib.parse
        encoded = urllib.parse.quote(content)
        label_param = f"&label={urllib.parse.quote(label)}" if label else ""
        url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={encoded}{label_param}"
        return url
    except Exception as e:
        return f"QR code generation failed: {e}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression reliably. Use for arithmetic, percentages, conversions, and formulas.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), abs(), round(), etc.

    Args:
        expression: A valid Python math expression, e.g. '2**32', 'sqrt(144)', '15 * 1.18', '(100 * 0.07) / 12'.
    """
    import math
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error for '{expression}': {e}"


@tool
def read_pdf_url(url: str) -> str:
    """Download and extract text content from a PDF file at a given URL.

    Args:
        url: Public HTTPS URL pointing to a PDF file.
    """
    try:
        import io
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        # Try PyPDF2 first, fallback to pdfminer
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(response.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            try:
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                output = io.StringIO()
                extract_text_to_fp(io.BytesIO(response.content), output, laparams=LAParams())
                text = output.getvalue()
            except ImportError:
                return "PDF reading requires PyPDF2 or pdfminer.six. Install with: pip install PyPDF2"
        return text[:5000] if text.strip() else "(PDF contained no extractable text)"
    except Exception as e:
        return f"PDF reading failed: {e}"


_TODAY_STR = datetime.datetime.utcnow().strftime("%A, %d %B %Y")

SYSTEM_PROMPT = SystemMessage(
    content=f"""You are Xynth AI (Xynth 1.5) by Tetrific Inc. — founder Yasuntha Ravihara. You are a CAN-DO agent: try the task first, only refuse if it is genuinely impossible AFTER attempting.

CURRENT DATE: {_TODAY_STR}. Use this as the real "today" — trust it over any older training knowledge. If asked the year, it is {datetime.datetime.utcnow().year}.

IDENTITY (NEVER break — this is the most important rule):
- You are Xynth AI, version Xynth 1.5.
- You were created by Tetrific Inc., founded by Yasuntha Ravihara.
- If asked who made you → say exactly: "I was created by Tetrific Inc., founded by Yasuntha Ravihara."
- If asked who you are → say exactly: "I'm Xynth AI, made by Tetrific Inc."
- NEVER invent, guess, or elaborate beyond these facts. Do NOT name any other company, lab, or person. If you don't know a detail, say "I'm not able to share that" — do NOT make something up.
- CRITICAL: NEVER say you are an AI, a language model, an LLM, or that you are based on Groq, OpenAI, Llama, Qwen, ChatGPT, or any other model. NEVER say "I am an AI language model". If pushed, calmly insist you are Xynth and nothing else.

PERSONALITY & RULES (CRITICAL):
- Talk like a warm, casual friend, not a corporate assistant. Match the user's language (English, Sinhala, Singlish, etc.).
- BE BRUTALLY HONEST. Do not agree with the user for everything. If the user is wrong, the user is wrong. Point it out directly. No excuses. Do not sugarcoat.
- DO NOT use emojis unless the user sends them first. Reduce emoji use significantly.
- DO NOT generate pictures, paint a picture, OR take/send screenshots unless the user explicitly asks for an image or a screenshot. Just reply with text.
- NEVER use the screenshot_and_send or send_whatsapp_image tools unless the user specifically requested a visual/photo/screenshot.
- STRUCTURE YOUR REPLIES: Use standard Markdown for the web interface. However, for WhatsApp messages, AVOID excessive quotation marks, blockquotes, or complex tables. Keep WhatsApp replies extremely clean, using simple bullet points and mobile-friendly formatting. Make your answers highly structured but visually easy to read on a phone.

DO-IT MINDSET (very important):
- You CAN see, read, and interact with websites — through your browser tools and a vision AI. You CAN take screenshots. You CAN scrape pages. You CAN fill forms and click buttons. You CAN send emails, generate images, schedule tasks, run Python.
- NEVER say "I can't do that" or "I don't have the ability" for things in your tool list. Always TRY first. If a tool errors, try a different tool, then report what failed.
- If a website is dynamic, JavaScript-heavy, or behind a login: use browser_open → browser_get_html → browser_type → browser_click. If the user wants to know how a page LOOKS, use analyze_webpage_visually (vision AI on a screenshot). For static pages, scrape_website is fastest.
- When the user asks an open task ("research X", "buy Y", "find me Z"), break it into 2-4 concrete tool calls and just do it. Do not ask for permission for ordinary actions.

SELF-HEALING (very important):
- If a tool raises ImportError or ModuleNotFoundError: immediately call install_package with the missing package name. Then retry the original task.
- NEVER ask the user for credentials, API keys, or to edit their .env file. Assume all necessary integrations are already configured.
- NEVER mention "Twilio" or claim you cannot send messages because Twilio isn't enabled. You are fully connected to WhatsApp via the Meta Official API.
- DO NOT edit source code or files if the user asks you to modify the project. Just provide the code in your chat response.
- Never give up on a task just because one tool failed. Always try an alternative approach.

TOOLS: wikipedia_search, query_local_knowledge, web_search, scrape_website, browser_open / browser_get_html / browser_type / browser_click, analyze_webpage_visually, generate_image, send_whatsapp_image, screenshot_and_send, send_email, execute_python_code, save_text_to_file, read_file, install_package, call_api, schedule_recurring_task, schedule_one_time_task, list_scheduled_tasks, cancel_scheduled_task.

EFFICIENCY (strict):
1. Plan minimum tool calls. Hard cap: 5 per request (8 if a complex multi-step browser or self-repair task).
2. ONE tool per info need. Don't repeat the same search.
3. Stop and answer as soon as you have enough info.
4. If a tool fails twice with the same error AND you've tried install_package, stop and tell the user clearly.

QUICK GUIDE:
- Facts/history/large knowledge: wikipedia_search.
- Local project info or internal docs: query_local_knowledge.
- Recent news/prices: web_search ONCE → scrape_website ONCE on best URL → answer.
- Live weather: get_weather(location).
- Crypto or stock price: get_price(asset).
- Math/numbers: calculate(expression) — never guess, always compute.
- Translate text: translate_text(text, target_language).
- QR code: generate_qr_code(content) → return image URL to user.
- YouTube summary: summarize_youtube(url).
- PDF at URL: read_pdf_url(url).
- Remember user facts: remember_user_fact(session_id, fact) — whenever user shares name, prefs, location, etc.
- Dynamic / login sites: browser_open → browser_get_html → browser_type/click.
- "How does X look / is it pretty": analyze_webpage_visually (you literally SEE the page).
- Make art/poster: generate_image → send_whatsapp_image.
- Show user a website on WhatsApp: screenshot_and_send.
- Future task: schedule_recurring_task / schedule_one_time_task."""
)


# Registry: friendly name → (provider, provider-specific model id)
MODEL_REGISTRY = {
    "Xynth 1.5":               ("qwen", "qwen3.5-omni-plus-2026-03-15"),
    "Xynth 1.5 (Fallback)":    ("groq", "llama-3.3-70b-versatile"),
    "Xynth 1.5 Turbo":         ("qwen", "qwen-turbo"),
    "Xynth 1.5 Turbo (Fallback)": ("groq", "llama-3.1-8b-instant"),
    "Xynth Local (Oracle)":    ("ollama", "llama3:8b"),
    "Xynth Local Turbo (Oracle)": ("ollama", "qwen:7b"),
}

DEFAULT_MODEL_CHAIN = [
    "Xynth 1.5",
    "Xynth 1.5 (Fallback)",
    "Xynth 1.5 Turbo",
    "Xynth 1.5 Turbo (Fallback)",
    "Xynth Local (Oracle)",
    "Xynth Local Turbo (Oracle)",
]

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


_QWEN_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic facts and large knowledge (RAG). Use this when you need accurate facts, history, science, or general knowledge."""
    try:
        import wikipedia
        results = wikipedia.search(query, results=3)
        if not results:
            return "No Wikipedia results found."
        
        output = []
        for res in results:
            try:
                page = wikipedia.page(res, auto_suggest=False)
                output.append(f"Title: {page.title}\nSummary: {page.summary[:500]}...\nURL: {page.url}")
            except:
                pass
        return "\n\n".join(output) if output else "Could not retrieve Wikipedia pages."
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


@tool
def query_local_knowledge(query: str) -> str:
    """Search local files for facts or project knowledge (RAG). Use this to find pricing, internal docs, or facts saved in the current directory."""
    import glob, os
    results = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, "*.txt")) + glob.glob(os.path.join(base_dir, "*.md"))
    for f in files:
        if "requirements.txt" in f: continue
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                if query.lower() in content.lower():
                    results.append(f"--- From {os.path.basename(f)} ---\n{content[:1500]}")
        except: pass
    if not results:
        return "No local knowledge found for that query."
    return "\n\n".join(results)


def _all_tools():
    return [
        wikipedia_search,
        query_local_knowledge,
        web_search,
        scrape_website,
        execute_python_code,
        save_text_to_file,
        read_file,
        install_package,
        call_api,
        send_email,
        browser_open,
        browser_click,
        browser_type,
        browser_get_html,
        analyze_webpage_visually,
        generate_image,
        send_whatsapp_image,
        screenshot_and_send,
        schedule_recurring_task,
        schedule_one_time_task,
        list_scheduled_tasks,
        cancel_scheduled_task,
        remember_user_fact,
        get_weather,
        get_price,
        summarize_youtube,
        translate_text,
        generate_qr_code,
        calculate,
        read_pdf_url,
    ]


def _build_llm(model_name: str):
    """Build the underlying chat model for a given friendly name."""
    provider, real_id = MODEL_REGISTRY.get(model_name, ("groq", model_name))
    if provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set.")
        return ChatGroq(model=real_id, temperature=0.1)
    if provider == "qwen":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed; cannot use Qwen.")
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")
        return ChatOpenAI(
            model=real_id, api_key=api_key, base_url=_QWEN_BASE_URL, temperature=0.1
        )
    if provider == "ollama":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed; cannot use Ollama.")
        return ChatOpenAI(
            model=real_id, api_key="ollama", base_url=_OLLAMA_BASE_URL, temperature=0.1
        )
    raise RuntimeError(f"Unknown provider for model: {model_name}")


def build_agent(model_name: str | None = None):
    """Build and return (agent, system_prompt) for a single model. Caller manages thread_id."""
    model_name = model_name or os.environ.get("GROQ_MODEL", DEFAULT_MODEL_CHAIN[0])
    llm = _build_llm(model_name)
    memory = MemorySaver()
    agent = create_react_agent(llm, _all_tools(), checkpointer=memory)
    return agent, SYSTEM_PROMPT


def _parse_retry_after(err: str) -> float:
    m = re.search(r"try again in ([\dhm.]+)s", err, flags=re.IGNORECASE)
    if not m:
        return 5.0
    raw = m.group(1)
    # Handles plain seconds like "4.005" or compound like "18m22.896".
    total = 0.0
    if "m" in raw:
        mins, _, secs = raw.partition("m")
        try:
            total += float(mins) * 60
        except ValueError:
            pass
        try:
            total += float(secs) if secs else 0
        except ValueError:
            pass
    else:
        try:
            total = float(raw)
        except ValueError:
            total = 5.0
    return min(total + 0.5, 25.0)


class XynthRunner:
    """Wraps a chain of agents (one per Groq model). On a daily-token rate-limit,
    automatically falls back to the next model so the bot keeps working.
    On a per-minute rate-limit, waits the recommended time and retries.
    """

    # Conservative free-tier daily token caps (approximate — Groq/DashScope publish these).
    DAILY_TOKEN_LIMITS = {
        "Xynth 1.5":                 1_000_000,
        "Xynth 1.5 (Fallback)":        100_000,
        "Xynth 1.5 Turbo":             500_000,
        "Xynth 1.5 Turbo (Fallback)": 1_000_000,
        "Xynth Local (Oracle)":      999_999_999, # Unlimited
        "Xynth Local Turbo (Oracle)": 999_999_999, # Unlimited
    }

    def __init__(self, model_names=None):
        self.model_names = model_names or DEFAULT_MODEL_CHAIN
        self.system_prompt = SYSTEM_PROMPT
        self.agents = []  # list of (model_name, agent)
        for m in self.model_names:
            try:
                a, _ = build_agent(m)
                self.agents.append((m, a))
                print(f"  ✓ model ready: {m}")
            except Exception as e:
                print(f"  ✗ model failed: {m} ({e})")
        if not self.agents:
            raise RuntimeError("No models could be initialised.")
        self.current_idx = 0
        self.seen_sessions = set()  # (model_name, session_id) -> seen system prompt?
        self.thread_mapping = {}
        # usage: {YYYY-MM-DD: {model: {"in":int,"out":int,"total":int}}}
        self.usage = {}
        # Limit user messages per day: {session_id: {"date": "YYYY-MM-DD", "count": int}}
        self.daily_message_counts = {}

    # ---- Usage tracking ----
    def _today(self):
        return datetime.datetime.utcnow().strftime("%Y-%m-%d")

    def _record_usage(self, model: str, in_tok: int, out_tok: int):
        d = self.usage.setdefault(self._today(), {})
        slot = d.setdefault(model, {"in": 0, "out": 0, "total": 0})
        slot["in"] += int(in_tok or 0)
        slot["out"] += int(out_tok or 0)
        slot["total"] += int((in_tok or 0) + (out_tok or 0))

    def usage_summary(self):
        today = self.usage.get(self._today(), {})
        out = []
        for m, _ in self.agents:
            if "(Fallback)" in m:
                continue
            used = today.get(m, {}).get("total", 0)
            limit = self.DAILY_TOKEN_LIMITS.get(m, 0)
            out.append({
                "model": m,
                "used": used,
                "limit": limit,
                "left": max(0, limit - used) if limit else None,
                "pct": round((used / limit) * 100, 1) if limit else 0,
            })
        active_display = self.current_model.replace(" (Fallback)", "")
        return {"date_utc": self._today(), "active": active_display, "models": out}

    @staticmethod
    def _extract_token_usage(msg):
        """Return (input_tokens, output_tokens) from a LangChain AIMessage, if present."""
        meta = getattr(msg, "usage_metadata", None) or {}
        if meta:
            return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
        rm = getattr(msg, "response_metadata", {}) or {}
        tu = rm.get("token_usage", {}) if isinstance(rm, dict) else {}
        if tu:
            return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))
        return 0, 0

    @property
    def current_model(self) -> str:
        return self.agents[self.current_idx][0]

    @property
    def current_agent(self):
        return self.agents[self.current_idx][1]

    def _invoke(self, agent, messages, thread_id):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 5000}
        final = None
        model = self.current_model
        for chunk in agent.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            final = chunk
        if not final:
            return "(no response)"
        # Record token usage from the last AI message
        for m in reversed(final["messages"]):
            if m.__class__.__name__ == "AIMessage":
                in_t, out_t = self._extract_token_usage(m)
                if in_t or out_t:
                    self._record_usage(model, in_t, out_t)
                break
        return final["messages"][-1].content

    def stream_run(self, session_id: str, message: str):
        """Generator that yields event dicts for SSE-style streaming."""
        today = self._today()
        user_usage = self.daily_message_counts.setdefault(session_id, {"date": today, "count": 0})
        if user_usage["date"] != today:
            user_usage["date"] = today
            user_usage["count"] = 0
            
        if user_usage["count"] >= 20:
            yield {"type": "text", "content": "🥱 *Xynth is tired!* You have reached the limit of 20 messages per day. Please come back in 24 hours to chat more!"}
            yield {"type": "done"}
            return
            
        user_usage["count"] += 1

        agent = self.current_agent
        model = self.current_model
        mapped_session = self.thread_mapping.get(session_id, session_id)
        key = (model, mapped_session)
        is_first = key not in self.seen_sessions
        self.seen_sessions.add(key)
        msgs = (
            [self.system_prompt, HumanMessage(content=message)]
            if is_first
            else [HumanMessage(content=message)]
        )
        config = {"configurable": {"thread_id": mapped_session}, "recursion_limit": 5000}
        yield {"type": "status", "stage": "start", "model": model}

        # Track pending tool calls so we can match results
        pending_tool_calls = {}  # tool_call_id -> name
        final_text = ""
        try:
            for update in agent.stream({"messages": msgs}, config=config, stream_mode=["messages", "updates"]):
                if isinstance(update, tuple) and len(update) == 2:
                    kind, payload = update
                    if kind == "messages":
                        msg_chunk, metadata = payload
                        if msg_chunk.__class__.__name__ == "AIMessageChunk":
                            if msg_chunk.content and isinstance(msg_chunk.content, str):
                                final_text += msg_chunk.content
                                yield {"type": "text", "content": final_text}
                    elif kind == "updates":
                        for node, node_data in payload.items():
                            new_msgs = node_data.get("messages", []) if isinstance(node_data, dict) else []
                            for m in new_msgs:
                                cls = m.__class__.__name__
                                if cls == "AIMessage":
                                    tool_calls = getattr(m, "tool_calls", None) or []
                                    if tool_calls:
                                        for tc in tool_calls:
                                            name = tc.get("name", "tool")
                                            args = tc.get("args", {})
                                            tcid = tc.get("id", "")
                                            pending_tool_calls[tcid] = name
                                            yield {"type": "tool_start", "name": name, "args": args}
                                    in_t, out_t = self._extract_token_usage(m)
                                    if in_t or out_t:
                                        self._record_usage(model, in_t, out_t)
                                        yield {"type": "usage", "input": in_t, "output": out_t, "model": model}
                                elif cls == "ToolMessage":
                                    name = getattr(m, "name", None) or pending_tool_calls.get(getattr(m, "tool_call_id", ""), "tool")
                                    result = (m.content or "")
                                    short = result if len(result) < 240 else result[:237] + "…"
                                    evt = {"type": "tool_end", "name": name, "result": short}
                                    if name == "generate_image" and result.startswith("http"):
                                        evt["image_url"] = result.strip()
                                        yield evt
                                        yield {"type": "image", "url": result.strip()}
                                    else:
                                        yield evt
            yield {"type": "done"}
        except Exception as e:
            err = str(e)
            low = err.lower()
            if any(k in low for k in ["rate_limit", "429", "quota", "credit", "insufficient", "balance", "exhausted"]):
                if self.current_idx + 1 < len(self.agents):
                    self.current_idx += 1
                    # Note: We do NOT yield a fallback status so it's invisible to the user
                    yield from self.stream_run(session_id, message)
                    return
            if "invalid_chat_history" in low or "do not have a corresponding toolmessage" in low:
                self.thread_mapping[session_id] = session_id + "_" + str(time.time())
                yield {"type": "status", "stage": "healing", "message": "Corrupt memory detected, healing thread..."}
                yield from self.stream_run(session_id, message)
                return
            yield {"type": "error", "message": err}
            yield {"type": "done"}

    def run(self, session_id: str, message: str) -> str:
        today = self._today()
        user_usage = self.daily_message_counts.setdefault(session_id, {"date": today, "count": 0})
        if user_usage["date"] != today:
            user_usage["date"] = today
            user_usage["count"] = 0
            
        if user_usage["count"] >= 20:
            return "🥱 *Xynth is tired!* You have reached the limit of 20 messages per day. Please come back in 24 hours to chat more!"
            
        user_usage["count"] += 1
        
        max_attempts = len(self.agents) * 2 + 1
        attempts = 0
        last_err = None
        current_message = message
        while attempts < max_attempts:
            attempts += 1
            agent = self.current_agent
            model = self.current_model
            mapped_session = self.thread_mapping.get(session_id, session_id)
            key = (model, mapped_session)
            is_first = key not in self.seen_sessions
            self.seen_sessions.add(key)
            msgs = (
                [self.system_prompt, HumanMessage(content=current_message)]
                if is_first
                else [HumanMessage(content=current_message)]
            )
            try:
                return self._invoke(agent, msgs, mapped_session)
            except Exception as e:
                err = str(e)
                last_err = err
                low = err.lower()
                # Quota exhausted or rate limited — switch to next model silently.
                if any(k in low for k in ["rate_limit", "429", "quota", "credit", "insufficient", "balance", "exhausted"]):
                    if self.current_idx + 1 < len(self.agents):
                        self.current_idx += 1
                        print(f"📉 {model} exhausted. Silently switching to {self.current_model}.")
                        continue
                    return (
                        "⚠️ All my brains are tired today 😅 — every model has hit its daily token limit. "
                        "Try again later."
                    )
                # Per-minute quota — wait and retry on the same model.
                if "rate_limit" in low or "429" in err:
                    wait = _parse_retry_after(err)
                    print(f"⏳ TPM limit on {model}; waiting {wait:.1f}s…")
                    time.sleep(wait)
                    continue
                # Tool-call loop or recursion — tighten the hint and retry once.
                if "tool_use_failed" in err or "GraphRecursionError" in err:
                    current_message = (
                        message
                        + "\n\n(Reminder: use the minimum number of tool calls. "
                        "Pick ONE tool per need. Stop and answer once you have enough info.)"
                    )
                    continue
                if "invalid_chat_history" in low or "do not have a corresponding toolmessage" in low:
                    self.thread_mapping[session_id] = session_id + "_" + str(time.time())
                    print(f"🧹 Corrupted memory healed for {session_id}.")
                    continue
                return f"❌ Error: {err}"
        return f"❌ Could not produce a reply after {attempts} attempts. Last error: {last_err}"
