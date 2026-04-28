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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun


_browser_state = {"playwright": None, "browser": None, "page": None}


def _ensure_browser():
    """Lazily start a Chromium browser/page and reuse it across calls."""
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


@tool
def scrape_website(url: str) -> str:
    """Quickly fetches and extracts text from a static website URL using HTTP requests. Best for simple pages, articles, and documentation. For sites that require JavaScript or interaction, use browser_open instead."""
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
    """Executes Python code and returns stdout. Use for math, string manipulation, or basic data processing."""
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
    """Saves text content to a local file. Useful for saving scraped data, reports, or automation results."""
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
    """Makes an HTTP request to an API and returns the response. Useful for connecting to external services."""
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
    """Sends an email via Gmail SMTP. Requires EMAIL_ADDRESS and EMAIL_APP_PASSWORD secrets to be set. Use for sending notifications, reports, or messages on the user's behalf."""
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
    """Opens a URL in a real headless Chromium browser (executes JavaScript). Returns visible text from the page. Use this for dynamic sites, SPAs, or pages where simple scraping fails. The browser session persists, so you can follow up with browser_click, browser_type, browser_get_html."""
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
    """Clicks an element in the current browser page. Selector is a CSS selector (e.g. 'button#submit', 'a[href=\"/login\"]', 'text=Sign up')."""
    try:
        page = _ensure_browser()
        page.click(selector, timeout=10000)
        page.wait_for_timeout(1500)
        return f"Clicked '{selector}'. Now at: {page.url}"
    except Exception as e:
        return f"Failed to click '{selector}': {str(e)}"


@tool
def browser_type(selector: str, text: str) -> str:
    """Types text into an input field in the current browser page. Selector is a CSS selector (e.g. 'input[name=\"email\"]', '#username')."""
    try:
        page = _ensure_browser()
        page.fill(selector, text, timeout=10000)
        return f"Typed into '{selector}'."
    except Exception as e:
        return f"Failed to type into '{selector}': {str(e)}"


@tool
def browser_get_html() -> str:
    """Returns the current page's HTML (truncated). Use to inspect form fields, buttons, and structure before clicking or typing."""
    try:
        page = _ensure_browser()
        html = page.content()
        return html[:8000]
    except Exception as e:
        return f"Failed to get HTML: {str(e)}"


def main():
    print("=" * 60)
    print("🤖🚀 Xynth AI - The Superagent")
    print("Powered by Groq + LangGraph")
    print("=" * 60)

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY is not set. Add it to Secrets.")
        return

    # GPT-OSS 20B has very reliable tool calling on Groq
    model_name = "openai/gpt-oss-20b"
    try:
        llm = ChatGroq(model=model_name, temperature=0.1)
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        return

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
    ]

    system_prompt = SystemMessage(content="""You are Xynth AI, a powerful superagent created by Aether Aiko (creator: Yasuntha Ravihara).
You can autonomously search the web, scrape pages, run Python, save files, call APIs, send emails, and control a real headless browser.

Tool guidance:
- For quick info from static pages: web_search + scrape_website.
- For dynamic sites, logins, forms, or anything requiring JavaScript/interaction: use browser_open, then browser_get_html to inspect, then browser_type and browser_click.
- For sending emails on the user's behalf: send_email.
- For computation or data manipulation: execute_python_code.
- Always think step by step: search → inspect → act → verify.
- If a tool fails, try a different approach (e.g. fall back from scrape_website to browser_open).
- Be concise in responses, but thorough in actions.""")

    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    thread_config = {"configurable": {"thread_id": "main-session"}}

    first_turn = True
    while True:
        try:
            user_query = input("\n👤 You: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Xynth AI powering down...")
                break

            print("🤖 Xynth is thinking...")

            messages = [system_prompt, HumanMessage(content=user_query)] if first_turn else [HumanMessage(content=user_query)]
            first_turn = False

            final_chunk = None
            try:
                for chunk in agent.stream({"messages": messages}, config=thread_config, stream_mode="values"):
                    final_chunk = chunk
                    message = chunk["messages"][-1]
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            print(f"   [🛠️ ] Using {tool_call['name']}...")
            except Exception as stream_err:
                err_text = str(stream_err)
                if "tool_use_failed" in err_text:
                    print("⚠️  Model produced a malformed tool call. Retrying with a hint...")
                    retry_messages = [HumanMessage(content=user_query + "\n\n(Important: only use the provided tools through the official function-call interface. Do not write tool calls as plain text.)")]
                    final_chunk = None
                    for chunk in agent.stream({"messages": retry_messages}, config=thread_config, stream_mode="values"):
                        final_chunk = chunk
                        message = chunk["messages"][-1]
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                print(f"   [🛠️ ] Using {tool_call['name']}...")
                else:
                    raise

            if final_chunk is not None:
                final_message = final_chunk["messages"][-1].content
                print(f"\n✨ Xynth AI: {final_message}")

        except KeyboardInterrupt:
            print("\n👋 Xynth AI powering down...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

    if _browser_state["browser"]:
        try:
            _browser_state["browser"].close()
            _browser_state["playwright"].stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
