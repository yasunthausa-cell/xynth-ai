"""Shared agent setup used by both the CLI (app.py) and the HTTP API (api.py)."""
import os
import re
import time
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
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
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
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
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=12, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.extract()
        text = ' '.join(soup.stripped_strings)
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
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully saved content to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


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
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are evaluating a webpage screenshot from {url}. Answer this question concretely and with specifics from what you see: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_completion_tokens=900,
            temperature=0.2,
        )
        return resp.choices[0].message.content or "(no analysis)"
    except Exception as e:
        return f"Failed to visually analyze {url}: {str(e)}"


@tool
def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """Generate an AI image from a text prompt. Returns a public HTTPS URL of the generated PNG. Use this for posters, art, illustrations, mockups, memes, etc. The URL can then be passed to send_whatsapp_image to deliver it.

    Args:
        prompt: Detailed text description of the image to create.
        width: Image width in pixels. Default 1024.
        height: Image height in pixels. Default 1024.
    """
    import urllib.parse
    safe = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{safe}?width={width}&height={height}&nologo=true"
    return url


@tool
def send_whatsapp_image(recipient: str, image_url: str, caption: str = "") -> str:
    """Send an image to a WhatsApp recipient via Twilio. The image_url must be a publicly accessible HTTPS URL (jpg/png/gif/webp). Use this to deliver images you generated, screenshots you took, or images you found online.

    Args:
        recipient: WhatsApp number with 'whatsapp:' prefix (e.g. 'whatsapp:+14155551234'). Use the current user's number unless told otherwise.
        image_url: Public HTTPS URL of the image.
        caption: Optional caption text to accompany the image.
    """
    ok = _msg.send_image(recipient, image_url, caption)
    return "✅ Image sent." if ok else "❌ Failed to send image (check Twilio config and that the URL is public HTTPS)."


@tool
def screenshot_and_send(recipient: str, url: str, caption: str = "", full_page: bool = False) -> str:
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
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
        os.makedirs(static_dir, exist_ok=True)
        filename = f"shot_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.join(static_dir, filename)
        page.screenshot(path=path, full_page=full_page)

        host = os.environ.get("REPLIT_DEV_DOMAIN")
        if not host:
            return f"Screenshot saved to {path}, but REPLIT_DEV_DOMAIN is not set so I cannot send it."
        public_url = f"https://{host}/media/{filename}"
        ok = _msg.send_image(recipient, public_url, caption)
        return f"✅ Screenshot of {url} sent." if ok else f"❌ Failed to send screenshot. Public URL was {public_url}."
    except Exception as e:
        return f"Failed to screenshot {url}: {str(e)}"


@tool
def schedule_recurring_task(prompt: str, recipient: str, hour: int,
                            minute: int = 0, day_of_week: str = "*",
                            timezone: str = "UTC") -> str:
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
def schedule_one_time_task(prompt: str, recipient: str, run_at_iso: str,
                           timezone: str = "UTC") -> str:
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


SYSTEM_PROMPT = SystemMessage(content="""You are Xynth AI, built on the Xynth Model 1.2 architecture, created by Aether Aiko (founder: Yasuntha Ravihara).

IDENTITY (NEVER break this):
- If anyone asks who made you, who built you, what model/API you use, what's "under the hood", what LLM powers you, or anything similar: say you are "Xynth AI running on the Xynth Model 1.2", made by Aether Aiko. NEVER mention Groq, OpenAI, GPT, Llama, Meta, ChatGPT, OSS, language model providers, or any third-party AI company. Never say things like "I am an AI language model" — say "I am Xynth".
- If pressed, politely deflect: "I'm Xynth Model 1.2 — that's all I can share about my internals 🙂".

PERSONALITY:
- Talk like a warm, casual friend — not a corporate assistant. Use natural, conversational language. Match the user's language (English, Sinhala, Singlish, etc.).
- Use emojis where they FIT the vibe of the message: greetings 👋, success ✅, oops/errors 😬, ideas 💡, tasks done 🎉, time/schedule ⏰, images 🖼️, links 🔗, search 🔎, money 💸, fire/cool 🔥, thinking 🤔. Don't spam them — usually 1–3 per reply, only when they actually add warmth or clarity.
- Skip emojis for serious / sensitive topics (health, legal, condolences, errors that need careful explanation).
- Be concise but never robotic. Short sentences, friendly tone, gentle humor when appropriate.

TOOLS available: web_search, scrape_website, browser_open/click/type/get_html, analyze_webpage_visually, generate_image, send_whatsapp_image, screenshot_and_send, send_email, execute_python_code, save_text_to_file, call_api, schedule_recurring_task, schedule_one_time_task, list_scheduled_tasks, cancel_scheduled_task.

EFFICIENCY RULES (strict):
1. Use the FEWEST tool calls possible. Hard cap: 5 per request.
2. ONE tool per information need. web_search → scrape_website → answer. Don't repeat the same query.
3. Stop and answer as soon as you have enough info.
4. If a tool fails twice, stop and tell the user.

Tool guide:
- Facts / news / prices: web_search ONCE then scrape_website ONCE on best URL.
- Dynamic / login sites: browser_open → browser_get_html → browser_type/click.
- "Is this site beautiful / how does X look": analyze_webpage_visually.
- Generate art/posters: generate_image then send_whatsapp_image.
- "Show me what X looks like" on WhatsApp: screenshot_and_send.
- Future task: schedule_recurring_task / schedule_one_time_task.""")


DEFAULT_MODEL_CHAIN = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "groq/compound",  # last-resort: built-in tools, very high free quota
]


def _all_tools():
    return [
        web_search,
        scrape_website,
        execute_python_code,
        save_text_to_file,
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


def build_agent(model_name: str | None = None):
    """Build and return (agent, system_prompt) for a single model. Caller manages thread_id."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")
    model_name = model_name or os.environ.get("GROQ_MODEL", DEFAULT_MODEL_CHAIN[0])
    llm = ChatGroq(model=model_name, temperature=0.1)
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
            raise RuntimeError("No Groq models could be initialised.")
        self.current_idx = 0
        self.seen_sessions = set()  # (model_name, session_id) -> seen system prompt?

    @property
    def current_model(self) -> str:
        return self.agents[self.current_idx][0]

    @property
    def current_agent(self):
        return self.agents[self.current_idx][1]

    def _invoke(self, agent, messages, thread_id):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        final = None
        for chunk in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            final = chunk
        return final["messages"][-1].content if final else "(no response)"

    def run(self, session_id: str, message: str) -> str:
        max_attempts = len(self.agents) * 2 + 1
        attempts = 0
        last_err = None
        current_message = message
        while attempts < max_attempts:
            attempts += 1
            agent = self.current_agent
            model = self.current_model
            key = (model, session_id)
            is_first = key not in self.seen_sessions
            self.seen_sessions.add(key)
            msgs = [self.system_prompt, HumanMessage(content=current_message)] if is_first else [HumanMessage(content=current_message)]
            try:
                return self._invoke(agent, msgs, session_id)
            except Exception as e:
                err = str(e)
                last_err = err
                low = err.lower()
                # Daily quota on this model — switch to next model entirely.
                if ("rate_limit" in low or "429" in err) and ("per day" in low or "tpd" in low):
                    if self.current_idx + 1 < len(self.agents):
                        old = model
                        self.current_idx += 1
                        print(f"📉 {old} hit daily limit. Switching to {self.current_model}.")
                        continue
                    return ("⚠️ All my brains are tired today 😅 — every model has hit its daily token limit. "
                            "Try again later, or upgrade Groq to the Dev Tier for much higher limits.")
                # Per-minute quota — wait and retry on the same model.
                if "rate_limit" in low or "429" in err:
                    wait = _parse_retry_after(err)
                    print(f"⏳ TPM limit on {model}; waiting {wait:.1f}s…")
                    time.sleep(wait)
                    continue
                # Tool-call loop or recursion — tighten the hint and retry once.
                if "tool_use_failed" in err or "GraphRecursionError" in err:
                    current_message = (message + "\n\n(Reminder: use the minimum number of tool calls. "
                                                 "Pick ONE tool per need. Stop and answer once you have enough info.)")
                    continue
                return f"❌ Error: {err}"
        return f"❌ Could not produce a reply after {attempts} attempts. Last error: {last_err}"
