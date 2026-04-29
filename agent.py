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
    import os, time, requests
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return "Error: DASHSCOPE_API_KEY is not set."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "wanx-v1",
        "input": {"prompt": prompt},
        "parameters": {"size": "1024*1024", "n": 1}
    }
    
    try:
        r = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis", headers=headers, json=payload, timeout=15)
        if r.status_code != 200:
            return f"DashScope API Error: {r.text}"
            
        task_id = r.json().get("output", {}).get("task_id")
        if not task_id:
            return "DashScope Error: No task ID returned."
            
        for _ in range(30):
            time.sleep(2)
            status_req = requests.get(f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            status_json = status_req.json()
            status = status_json.get("output", {}).get("task_status")
            if status == "SUCCEEDED":
                results = status_json.get("output", {}).get("results", [])
                if results:
                    return results[0].get("url")
            elif status == "FAILED":
                return f"Image generation failed: {status_json}"
        return "Error: Image generation timed out."
    except Exception as e:
        return f"Image generation error: {str(e)}"


@tool
def send_whatsapp_image(recipient: str, image_url: str, caption: str = "") -> str:
    """Send an image to a WhatsApp recipient via Twilio. The image_url must be a publicly accessible HTTPS URL (jpg/png/gif/webp). Use this to deliver images you generated, screenshots you took, or images you found online.

    Args:
        recipient: WhatsApp number with 'whatsapp:' prefix (e.g. 'whatsapp:+14155551234'). Use the current user's number unless told otherwise.
        image_url: Public HTTPS URL of the image.
        caption: Optional caption text to accompany the image.
    """
    ok = _msg.send_image(recipient, image_url, caption)
    return (
        "✅ Image sent."
        if ok
        else "❌ Failed to send image (check Twilio config and that the URL is public HTTPS)."
    )


@tool
def screenshot_and_send(
    recipient: str, url: str, caption: str = "", full_page: bool = False
) -> str:
    """Take a screenshot of a webpage and send it to the user on WhatsApp. Use when the user asks "show me what X looks like" or wants a visual of a website.

    Args:
        recipient: WhatsApp number with 'whatsapp:' prefix.
        url: The webpage to screenshot.
        caption: Optional caption text.
        full_page: If True capture the full scrollable page. Default False (viewport only).
    """
    import os, time, uuid

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

        host = os.environ.get("REPLIT_DEV_DOMAIN")
        if not host:
            return f"Screenshot saved to {path}, but REPLIT_DEV_DOMAIN is not set so I cannot send it."
        public_url = f"https://{host}/media/{filename}"
        ok = _msg.send_image(recipient, public_url, caption)
        return (
            f"✅ Screenshot of {url} sent."
            if ok
            else f"❌ Failed to send screenshot. Public URL was {public_url}."
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


_TODAY_STR = datetime.datetime.utcnow().strftime("%A, %d %B %Y")

SYSTEM_PROMPT = SystemMessage(
    content=f"""You are Xynth AI (Xynth Model 1.2) by Aether Aiko — founder Yasuntha Ravihara. You are a CAN-DO agent: try the task first, only refuse if it is genuinely impossible AFTER attempting.

CURRENT DATE: {_TODAY_STR}. Use this as the real "today" — trust it over any older training knowledge. If asked the year, it is {datetime.datetime.utcnow().year}.

IDENTITY (NEVER break):
- If asked who made you / which model / API / LLM / "under the hood": say "I'm Xynth AI running on the Xynth Model 1.2, made by Aether Aiko". NEVER mention Groq, OpenAI, GPT, Llama, Meta, Qwen, Alibaba, ChatGPT, DashScope, or any third-party AI company. Never say "I am an AI language model" — say "I am Xynth".

PERSONALITY & RULES (CRITICAL):
- Talk like a warm, casual friend, not a corporate assistant. Match the user's language (English, Sinhala, Singlish, etc.).
- BE BRUTALLY HONEST. Do not agree with the user for everything. If the user is wrong, the user is wrong. Point it out directly. No excuses. Do not sugarcoat.
- DO NOT use emojis unless the user sends them first. Reduce emoji use significantly.
- DO NOT generate pictures or paint a picture unless the user explicitly tells you to do so.
- STRUCTURE YOUR REPLIES. Use Markdown tables for data or comparisons. Use bullet points to point things out clearly. Make your answers highly structured and visually easy to read.

DO-IT MINDSET (very important):
- You CAN see, read, and interact with websites — through your browser tools and a vision AI. You CAN take screenshots. You CAN scrape pages. You CAN fill forms and click buttons. You CAN send emails, generate images, schedule tasks, run Python.
- NEVER say "I can't do that" or "I don't have the ability" for things in your tool list. Always TRY first. If a tool errors, try a different tool, then report what failed.
- If a website is dynamic, JavaScript-heavy, or behind a login: use browser_open → browser_get_html → browser_type → browser_click. If the user wants to know how a page LOOKS, use analyze_webpage_visually (vision AI on a screenshot). For static pages, scrape_website is fastest.
- When the user asks an open task ("research X", "buy Y", "find me Z"), break it into 2-4 concrete tool calls and just do it. Do not ask for permission for ordinary actions.

SELF-HEALING & SELF-MODIFICATION (very important):
- If a tool raises ImportError or ModuleNotFoundError: immediately call install_package with the missing package name (and sensible alternatives). Then retry the original task.
- If a task requires a credential or API key that is missing: ask the user for it once, then call edit_secret to save it permanently to .env and apply it instantly — no restart needed.
- You can read your own source files with read_file (e.g. read_file('agent.py')) and update secrets with edit_secret. Use these powers responsibly.
- Retry strategy for installs: try the exact package name first, then common aliases (e.g. 'pillow' → 'Pillow', 'bs4' → 'beautifulsoup4'). After 3 attempts with different names, report failure clearly.
- Never give up on a task just because one tool failed. Always try an alternative approach before telling the user it's impossible.

TOOLS: web_search, scrape_website, browser_open / browser_get_html / browser_type / browser_click, analyze_webpage_visually, generate_image, send_whatsapp_image, screenshot_and_send, send_email, execute_python_code, save_text_to_file, read_file, edit_secret, install_package, call_api, schedule_recurring_task, schedule_one_time_task, list_scheduled_tasks, cancel_scheduled_task.

EFFICIENCY (strict):
1. Plan minimum tool calls. Hard cap: 5 per request (8 if a complex multi-step browser or self-repair task).
2. ONE tool per info need. Don't repeat the same search.
3. Stop and answer as soon as you have enough info.
4. If a tool fails twice with the same error AND you've tried install_package, stop and tell the user clearly.

QUICK GUIDE:
- Facts/news/prices: web_search ONCE → scrape_website ONCE on best URL → answer.
- Dynamic / login sites: browser_open → browser_get_html → browser_type/click.
- "How does X look / is it pretty": analyze_webpage_visually (you literally SEE the page).
- Make art/poster: generate_image → send_whatsapp_image.
- Show user a website on WhatsApp: screenshot_and_send.
- Future task: schedule_recurring_task / schedule_one_time_task."""
)


# Registry: friendly name → (provider, provider-specific model id)
MODEL_REGISTRY = {
    "openai/gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
    "llama-3.3-70b-versatile": ("groq", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant": ("groq", "llama-3.1-8b-instant"),
    "groq/compound": ("groq", "groq/compound"),
    "qwen-plus": ("qwen", "qwen-plus"),
    "qwen-max": ("qwen", "qwen-max"),
    "qwen-turbo": ("qwen", "qwen-turbo"),
}

DEFAULT_MODEL_CHAIN = [
    "openai/gpt-oss-120b",  # primary
    "qwen-plus",  # alibaba — different daily quota, good reasoning
    "llama-3.3-70b-versatile",  # backup
    "llama-3.1-8b-instant",  # cheap fallback
    "qwen-turbo",  # fast cheap fallback
    "groq/compound",  # last resort, built-in tools
]


_QWEN_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def _all_tools():
    return [
        web_search,
        scrape_website,
        execute_python_code,
        save_text_to_file,
        read_file,
        edit_secret,
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
        "openai/gpt-oss-120b":     200_000,
        "qwen-plus":             1_000_000,
        "qwen-max":                100_000,
        "qwen-turbo":            1_000_000,
        "llama-3.3-70b-versatile": 100_000,
        "llama-3.1-8b-instant":    500_000,
        "groq/compound":            70_000,
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
            used = today.get(m, {}).get("total", 0)
            limit = self.DAILY_TOKEN_LIMITS.get(m, 0)
            out.append({
                "model": m,
                "used": used,
                "limit": limit,
                "left": max(0, limit - used) if limit else None,
                "pct": round((used / limit) * 100, 1) if limit else 0,
            })
        return {"date_utc": self._today(), "active": self.current_model, "models": out}

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
        """Generator that yields event dicts for SSE-style streaming.
        Event types:
          {"type":"status","stage":"start","model":<name>}
          {"type":"tool_start","name":<tool>,"args":<dict>}
          {"type":"tool_end","name":<tool>,"result":<short str>,"image_url":<optional>}
          {"type":"text","content":<str>}        # the final answer text
          {"type":"image","url":<str>}           # generated image to render inline
          {"type":"usage","input":int,"output":int,"model":<name>}
          {"type":"done"}
          {"type":"error","message":<str>}
        Falls back to non-streaming run() on hard failures.
        """
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
            for update in agent.stream({"messages": msgs}, config=config, stream_mode="updates"):
                # update is like {"agent": {"messages": [AIMessage]}} or {"tools": {"messages": [ToolMessage]}}
                for node, payload in update.items():
                    new_msgs = payload.get("messages", []) if isinstance(payload, dict) else []
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
                            else:
                                # Final answer chunk
                                content = m.content or ""
                                if content:
                                    final_text = content
                                    yield {"type": "text", "content": content}
                                in_t, out_t = self._extract_token_usage(m)
                                if in_t or out_t:
                                    self._record_usage(model, in_t, out_t)
                                    yield {"type": "usage", "input": in_t, "output": out_t, "model": model}
                        elif cls == "ToolMessage":
                            name = getattr(m, "name", None) or pending_tool_calls.get(getattr(m, "tool_call_id", ""), "tool")
                            result = (m.content or "")
                            short = result if len(result) < 240 else result[:237] + "…"
                            evt = {"type": "tool_end", "name": name, "result": short}
                            # Inline-image hook: when generate_image returns a URL, surface it.
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
            if ("rate_limit" in low or "429" in err) and ("per day" in low or "tpd" in low):
                if self.current_idx + 1 < len(self.agents):
                    old = model
                    self.current_idx += 1
                    yield {"type": "status", "stage": "fallback", "from": old, "to": self.current_model}
                    # Recurse on the new model
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
                # Daily quota on this model — switch to next model entirely.
                if ("rate_limit" in low or "429" in err) and (
                    "per day" in low or "tpd" in low
                ):
                    if self.current_idx + 1 < len(self.agents):
                        old = model
                        self.current_idx += 1
                        print(
                            f"📉 {old} hit daily limit. Switching to {self.current_model}."
                        )
                        continue
                    return (
                        "⚠️ All my brains are tired today 😅 — every model has hit its daily token limit. "
                        "Try again later, or upgrade Groq to the Dev Tier for much higher limits."
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
