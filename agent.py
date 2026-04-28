"""Shared agent setup used by both the CLI (app.py) and the HTTP API (api.py)."""
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

import scheduler as _sched


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
def scrape_website(url: str) -> str:
    """Quickly fetch and extract text from a static website URL using HTTP requests. Best for simple pages and articles. For dynamic sites or login flows, use browser_open."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = ' '.join(soup.stripped_strings)
        return text[:6000]
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


SYSTEM_PROMPT = SystemMessage(content="""You are Xynth AI, a powerful superagent created by Aether Aiko (creator: Yasuntha Ravihara). You can search the web, scrape pages, run Python, save files, call APIs, send emails, and control a real headless browser.

CRITICAL EFFICIENCY RULES — follow them strictly to avoid wasting tool calls:
1. Plan first. Decide the minimum number of tool calls needed before you start.
2. Pick ONE tool per information need. Do NOT call web_search, scrape_website, AND browser_open for the same query.
   - Use web_search ONCE to find URLs.
   - Then use scrape_website ONCE on the most promising result. Only use browser_open if scrape_website fails or the site needs JavaScript.
3. Never repeat the same search query. If a search didn't help, refine the query — don't re-run it.
4. Stop and answer as soon as you have enough information. Do not "keep gathering" indefinitely.
5. Hard limit: aim for ≤ 5 tool calls per user request unless the task genuinely requires more (e.g., multi-step browser automation).
6. If a tool fails twice with the same error, stop trying it and tell the user what went wrong.

Tool selection:
- Quick facts / current info: web_search → scrape_website (one of each).
- Dynamic sites, logins, forms: browser_open → browser_get_html → browser_type → browser_click.
- Sending email: send_email.
- Computation: execute_python_code.

Be concise in answers, but thorough in actions.""")


def build_agent():
    """Build and return (agent, system_prompt). Caller manages thread_id for memory."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(model=model_name, temperature=0.1)

    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="Search the web for current information, news, or to find URLs."
    )

    tools = [
        search_tool,
        scrape_website,
        execute_python_code,
        save_text_to_file,
        call_api,
        send_email,
        browser_open,
        browser_click,
        browser_type,
        browser_get_html,
        schedule_recurring_task,
        schedule_one_time_task,
        list_scheduled_tasks,
        cancel_scheduled_task,
    ]

    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    return agent, SYSTEM_PROMPT
