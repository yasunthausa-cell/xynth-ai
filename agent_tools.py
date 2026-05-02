"""agent_tools.py — Full tool registry for Xynth Superagent.

All tools Xynth can use:
- run_python     : Execute Python code in a sandbox (auto-installs missing packages)
- web_search     : DuckDuckGo search via Playwright
- scrape_page    : Full page scraping via Playwright
- generate_image : AI image generation via Cloudflare Flux
- send_email     : Send emails via SMTP
- calculator     : Safe math evaluation
- memory_read    : Read long-term memory from Supabase
- memory_write   : Write long-term memory to Supabase
- wikipedia      : Fetch Wikipedia summary
"""
import os
import re
import sys
import subprocess
import math
import traceback
import requests
import base64
from io import StringIO

# ── Image Generation ──────────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN", "")
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static_media")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def generate_image(prompt: str) -> str:
    """Generate an AI image using Cloudflare Flux. Returns a URL to the image."""
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return "[Image generation failed: Cloudflare credentials not set]"
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "num_steps": 4}

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            # CF returns base64 image
            b64 = result.get("result", {}).get("image", "")
            if b64:
                import uuid, time
                filename = f"xynth_img_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(b64))
                return f"/static_media/{filename}"
        return f"[Image generation failed: HTTP {resp.status_code} — {resp.text[:200]}]"
    except Exception as e:
        return f"[Image generation error: {e}]"


# ── Python Sandbox with Auto-Healing ─────────────────────────────────────────
def run_python(code: str, max_retries: int = 3) -> str:
    """Execute Python code safely. Auto-installs missing packages on ImportError."""
    for attempt in range(max_retries):
        try:
            result = _exec_sandboxed(code)
            return result
        except ModuleNotFoundError as e:
            # Extract the missing module name
            match = re.search(r"No module named '([^']+)'", str(e))
            if match:
                pkg = match.group(1).split(".")[0]
                print(f"[Sandbox] Auto-installing missing package: {pkg}")
                install_result = _pip_install(pkg)
                if "Error" in install_result:
                    return f"[Auto-install failed for '{pkg}']: {install_result}\n\nOriginal code:\n{code}"
                # Retry loop
                continue
            return f"[Import Error] {e}"
        except Exception as e:
            return f"[Execution Error]\n{traceback.format_exc()}"
    return "[Max retries reached — could not install required packages]"


def _exec_sandboxed(code: str) -> str:
    """Execute code in a subprocess with a 30s timeout."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            # Check if it's an import error
            if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                match = re.search(r"No module named '([^']+)'", result.stderr)
                if match:
                    raise ModuleNotFoundError(f"No module named '{match.group(1)}'")
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(code ran with no output)"
    except subprocess.TimeoutExpired:
        return "[Timeout] Code execution exceeded 30 seconds."


def _pip_install(package: str) -> str:
    """Install a Python package at runtime."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return f"Successfully installed {package}"
        return f"Error: {result.stderr[:300]}"
    except Exception as e:
        return f"Error: {e}"


# ── Calculator ────────────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"[Calculator error: {e}]"


# ── Email ─────────────────────────────────────────────────────────────────────
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using SMTP credentials from environment variables."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        if not smtp_user or not smtp_pass:
            return "[Email not configured: set SMTP_USER and SMTP_PASS env vars]"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return f"Email sent to {to} successfully."
    except Exception as e:
        return f"[Email error: {e}]"


# ── Wikipedia ─────────────────────────────────────────────────────────────────
def wikipedia_search(query: str) -> str:
    """Get a short Wikipedia summary of a topic."""
    try:
        import wikipedia as wiki
        results = wiki.search(query, results=3)
        if not results:
            return "[No Wikipedia results found]"
        page = wiki.page(results[0], auto_suggest=False)
        return f"**{page.title}**\n{page.summary[:2000]}\n\nSource: {page.url}"
    except Exception as e:
        return f"[Wikipedia error: {e}]"


# ── Memory (Supabase) ─────────────────────────────────────────────────────────
def memory_write(user_id: str, key: str, value: str, sb=None) -> str:
    """Write a piece of long-term memory to Supabase."""
    if not sb:
        return "[Memory not available: Supabase not connected]"
    try:
        sb.table("memories").upsert({"user_id": user_id, "key": key, "value": value}).execute()
        return f"Memory saved: {key} = {value}"
    except Exception as e:
        return f"[Memory write error: {e}]"


def memory_read(user_id: str, key: str = None, sb=None) -> str:
    """Read long-term memory from Supabase."""
    if not sb:
        return "[Memory not available: Supabase not connected]"
    try:
        q = sb.table("memories").select("key,value").eq("user_id", user_id)
        if key:
            q = q.eq("key", key)
        res = q.execute()
        if not res.data:
            return "[No memory found]"
        return "\n".join(f"{m['key']}: {m['value']}" for m in res.data)
    except Exception as e:
        return f"[Memory read error: {e}]"


# ── Tool Registry ─────────────────────────────────────────────────────────────
# This is what gets sent to Cloudflare as function definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information, news, prices, weather, or anything time-sensitive. Use this for any question about real-world, current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Visit and read the full content of a specific webpage URL. Use when the user shares a link or you need to read a specific page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to visit and read"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code. Use for data analysis, math, file processing, automation, or any computation. Missing packages will be auto-installed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an AI image from a text description. Use when the user asks to create, draw, generate, or visualize something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed description of the image to generate"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform precise mathematical calculations. Use for any arithmetic, algebra, or math expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to calculate, e.g. '2 ** 32 + sqrt(144)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Look up factual information about a topic on Wikipedia. Good for historical facts, science, people, places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to look up on Wikipedia"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email on behalf of the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save something to long-term memory for a user. Use to remember preferences, names, or important facts the user shares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "A short label for the memory"},
                    "value": {"type": "string", "description": "The information to remember"}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Recall previously saved long-term memories about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Optional specific memory key to read. If not given, reads all memories."}
                },
                "required": []
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict, user_id: str = None, sb=None) -> str:
    """Execute a tool by name with given arguments. Returns string result."""
    from browser_tools import web_search, scrape_page

    try:
        if tool_name == "web_search":
            return web_search(tool_args.get("query", ""))
        elif tool_name == "scrape_page":
            return scrape_page(tool_args.get("url", ""))
        elif tool_name == "run_python":
            return run_python(tool_args.get("code", ""))
        elif tool_name == "generate_image":
            return generate_image(tool_args.get("prompt", ""))
        elif tool_name == "calculator":
            return calculator(tool_args.get("expression", ""))
        elif tool_name == "wikipedia_search":
            return wikipedia_search(tool_args.get("query", ""))
        elif tool_name == "send_email":
            return send_email(tool_args.get("to", ""), tool_args.get("subject", ""), tool_args.get("body", ""))
        elif tool_name == "memory_write":
            return memory_write(user_id or "anon", tool_args.get("key", ""), tool_args.get("value", ""), sb)
        elif tool_name == "memory_read":
            return memory_read(user_id or "anon", tool_args.get("key"), sb)
        else:
            return f"[Unknown tool: {tool_name}]"
    except Exception as e:
        return f"[Tool '{tool_name}' error: {e}]"
