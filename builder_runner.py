"""builder_runner.py — AI code generation engine for Xynth Builder.

Generates complete, self-contained HTML/CSS/JS apps from user descriptions.
Uses qwen-max with a specialized builder system prompt.
"""
import os
import json
import re
import datetime

try:
    from openai import OpenAI as _OAI
    _client = _OAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
except Exception:
    _client = None

# Fallback to Groq if no DashScope key
_groq_client = None
try:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        from groq import Groq
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
except Exception:
    pass

BUILDER_MODEL       = "qwen-max"
BUILDER_MODEL_GROQ  = "llama-3.3-70b-versatile"

BUILDER_SYSTEM_PROMPT = """You are Xynth Builder, an expert web developer AI.
Your ONLY job is to generate complete, stunning, production-quality web apps as a single self-contained HTML file.

STRICT RULES:
1. ALWAYS output the COMPLETE HTML file — never partial code or snippets.
2. Wrap the entire output in a single ```html code block. Nothing else outside it.
3. Include ALL CSS in a <style> tag and ALL JavaScript in a <script> tag. No external files.
4. Make designs STUNNING — use gradients, animations, modern fonts (via Google Fonts CDN), glassmorphism, dark themes.
5. Make it FULLY FUNCTIONAL — buttons click, forms submit, features work.
6. Use vibrant colors, smooth hover effects, professional typography.
7. The app must be completely self-contained — no backend calls needed.
8. When updating, output the COMPLETE updated file (not just the changed section).
9. Never explain what you're doing — just output the code block.

TECH STACK:
- Vanilla HTML5, CSS3, JavaScript (ES6+)
- Google Fonts via CDN (Inter, Outfit, etc.)
- Lucide Icons or Font Awesome via CDN
- Chart.js, Three.js, or GSAP if needed (via CDN)
- NO React, NO Vue, NO build tools

DESIGN PHILOSOPHY:
- Dark mode preferred unless user specifies
- Use CSS variables for theming
- Smooth transitions on all interactive elements
- Mobile responsive with media queries
- Micro-animations on hover and click"""


# ── In-memory project state (per session) ─────────────────────────────────────
_projects: dict = {}  # {session_id: {"code": str, "history": list}}


def _get_client():
    if _client and os.environ.get("DASHSCOPE_API_KEY"):
        return _client, BUILDER_MODEL
    if _groq_client:
        return _groq_client, BUILDER_MODEL_GROQ
    return None, None


def _extract_html(text: str) -> str:
    """Extract the HTML code block from AI response."""
    # Try ```html ... ``` first
    match = re.search(r'```html\s*([\s\S]+?)\s*```', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try ``` ... ``` (any code block)
    match = re.search(r'```\s*(<!DOCTYPE|<html)([\s\S]+?)\s*```', text, re.IGNORECASE)
    if match:
        return (match.group(1) + match.group(2)).strip()
    # If the whole response looks like HTML
    if text.strip().startswith('<!DOCTYPE') or text.strip().startswith('<html'):
        return text.strip()
    return ""


def stream_build(session_id: str, message: str, sb=None, user_id=None, project_id=None):
    """SSE generator for app building. Streams code and emits preview event when done."""
    client, model = _get_client()
    if not client:
        yield f"data: {json.dumps({'type': 'error', 'text': 'No AI client available. Set DASHSCOPE_API_KEY or GROQ_API_KEY.'})}\n\n"
        return

    # Load project history
    project = _projects.setdefault(session_id, {"code": "", "history": []})
    history = project["history"]

    # Build message list
    messages = [{"role": "system", "content": BUILDER_SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    full_response = ""
    yield f"data: {json.dumps({'type': 'status', 'text': '🏗️ Building your app...'})}\n\n"

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192,  # Need lots of tokens for full HTML files
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
        return

    # Extract and emit the HTML code
    html_code = _extract_html(full_response)
    if html_code:
        project["code"] = html_code
        yield f"data: {json.dumps({'type': 'preview', 'html': html_code})}\n\n"

    # Update history (keep last 10 turns to avoid token explosion)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})
    if len(history) > 20:
        project["history"] = history[-20:]

    # Persist to Supabase if logged in
    if sb and user_id and html_code:
        try:
            if project_id:
                sb.table("projects").update({
                    "code": html_code,
                    "updated_at": datetime.datetime.utcnow().isoformat()
                }).eq("id", project_id).execute()
            else:
                # Create new project
                title = message[:50] + ("..." if len(message) > 50 else "")
                res = sb.table("projects").insert({
                    "user_id": user_id,
                    "title": title,
                    "code": html_code,
                    "published": False,
                }).execute()
                if res.data:
                    new_id = res.data[0]["id"]
                    yield f"data: {json.dumps({'type': 'project_id', 'id': new_id, 'title': title})}\n\n"
        except Exception as e:
            print("Project save error:", e)

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def get_project_code(session_id: str) -> str:
    return _projects.get(session_id, {}).get("code", "")


def clear_project(session_id: str):
    _projects.pop(session_id, None)
