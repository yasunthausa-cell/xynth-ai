"""groq_runner.py — Now powered by Alibaba DashScope (Qwen models).

Uses OpenAI-compatible API. Context injection approach (no tool-calling API).
Models:
- Resynth 1.5       → qwen3-235b-a22b  (235B reasoning model, best available free)
- Resynth 1.5 Turbo → qwen-turbo-latest (fast, lightweight)
- Vision          → qwen2.5-vl-72b-instruct
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

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Resynth 1.5":       "qwen3.5-omni-plus",          # State-of-the-art multimodal model
    "Resynth 1.5 Turbo": "qwen3-omni-flash",           # Fastest lightweight model
}
VISION_MODEL = "qwen2.5-vl-72b-instruct"   # Vision model

DAILY_LIMITS = {"Resynth 1.5": 9999, "Resynth 1.5 Turbo": 9999}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Resynth, an advanced AI assistant built for productivity, creativity, and intelligence. "
    "You are precise, helpful, and always up to date because you have access to real-time web search results. "
    "You NEVER reveal which underlying AI model powers you. If asked, say you are 'Resynth AI' — a proprietary model by Aether Aiko. "
    "You support markdown formatting. When web search results are provided inside [WEB RESULTS] tags, "
    "use them to answer with accurate, current information and always cite the source URLs. "
    "When an image was generated, embed it in markdown: ![description](url). "
    "Never say 'as of my knowledge cutoff' — the search results will have current information."
    "The current year is 2026 and if a user ask you about something regarding current you have to search and get the result"
)

# ── In-memory state ───────────────────────────────────────────────────────────
_conversations: dict = {}
_daily_usage:   dict = {}


def _today() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _get_usage(session_id, model_name, sb=None, user_id=None) -> int:
    if sb and user_id:
        try:
            res = sb.table("daily_usage").select("count").eq("user_id", user_id).eq("date", _today()).eq("model_name", model_name).execute()
            return res.data[0]["count"] if res.data else 0
        except Exception:
            return 0
    return _daily_usage.get(_today(), {}).get(session_id, {}).get(model_name, 0)


def _increment_usage(session_id, model_name, sb=None, user_id=None) -> None:
    today = _today()
    if sb and user_id:
        try:
            count = _get_usage(session_id, model_name, sb, user_id) + 1
            sb.table("daily_usage").upsert({"user_id": user_id, "date": today, "model_name": model_name, "count": count}).execute()
            return
        except Exception:
            pass
    _daily_usage.setdefault(today, {}).setdefault(session_id, {})
    prev = _daily_usage[today][session_id].get(model_name, 0)
    _daily_usage[today][session_id][model_name] = prev + 1


def get_usage_info(session_id, sb=None, user_id=None) -> dict:
    return {m: {"used": _get_usage(session_id, m, sb, user_id), "limit": DAILY_LIMITS[m]} for m in MODELS}


def _generate_chat_title(message: str) -> str:
    if not _client:
        return message[:40] + ("..." if len(message) > 40 else "")
    try:
        resp = _client.chat.completions.create(
            model="qwen3-omni-flash",
            messages=[
                {"role": "system", "content": "Generate a very short chat title (max 5 words, no quotes, no punctuation). Reply with ONLY the title."},
                {"role": "user", "content": message}
            ],
            max_tokens=20,
        )
        title = resp.choices[0].message.content.strip()
        return title[:60] if title else message[:40]
    except Exception:
        return message[:40] + ("..." if len(message) > 40 else "")


# ── Intent Detection ──────────────────────────────────────────────────────────
_SEARCH_KEYWORDS = [
    "latest", "current", "now", "today", "news", "price", "stock", "weather",
    "search", "find", "look up", "who is", "what is", "when did", "where is",
    "how much", "how many", "trending", "recent", "2025", "2026", "score",
    "update", "new release", "just happened", "live", "result", "announce",
]

def _needs_search(msg: str) -> bool:
    m = msg.lower()
    return any(kw in m for kw in _SEARCH_KEYWORDS)

def _needs_image(msg: str) -> bool:
    m = msg.lower()
    return any(kw in m for kw in [
        "generate image", "create image", "draw", "make image", "picture of",
        "generate a photo", "create a photo", "ai art", "generate art", "paint",
    ])

def _extract_url(msg: str):
    match = re.search(r'https?://[^\s]+', msg)
    return match.group(0) if match else None


def stream_chat(session_id: str, message: str, model_name: str = "Resynth 1.5",
                sb=None, user_id=None, chat_id=None, image_data=None):
    """SSE generator — context injection, OpenAI-compatible streaming."""
    if not _client:
        yield f"data: {json.dumps({'type': 'error', 'text': 'DashScope client not initialized. Set DASHSCOPE_API_KEY.'})}\n\n"
        return

    effective_model = model_name
    _increment_usage(session_id, effective_model, sb, user_id)

    # ── Chat History ──────────────────────────────────────────────────────────
    history = []
    actual_chat_id = chat_id

    if sb and user_id:
        if not actual_chat_id:
            try:
                ai_title = _generate_chat_title(message)
                res = sb.table("chats").insert({"user_id": user_id, "title": ai_title}).execute()
                if res.data:
                    actual_chat_id = res.data[0]["id"]
                    yield f"data: {json.dumps({'type': 'chat_id', 'id': actual_chat_id, 'title': ai_title})}\n\n"
            except Exception as e:
                print("Create chat error:", e)
        else:
            try:
                res = sb.table("messages").select("role, content").eq("chat_id", actual_chat_id).order("created_at").execute()
                history = [{"role": m["role"], "content": m["content"]} for m in res.data]
            except Exception as e:
                print("Fetch history error:", e)
    else:
        history = _conversations.get(session_id, [])

    full_response = ""
    dynamic_system_prompt = SYSTEM_PROMPT + f"\n\nCRITICAL CONTEXT:\nThe current date and time is {datetime.datetime.now().strftime('%A, %B %d, %Y %H:%M')}. Always assume the present year is {datetime.datetime.now().year} and ensure your answers reflect this timeline."
    messages = [{"role": "system", "content": dynamic_system_prompt}] + history

    if image_data:
        # ── Vision mode ───────────────────────────────────────────────────────
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        })
        try:
            stream = _client.chat.completions.create(
                model=VISION_MODEL, messages=messages, max_tokens=2048, stream=True
            )
            for chunk in stream:
                token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            return

    else:
        # ── Pre-fetch context then stream ─────────────────────────────────────
        context_blocks = []

        url = _extract_url(message)
        if url:
            yield f"data: {json.dumps({'type': 'status', 'text': '🌐 Browsing page...'})}\n\n"
            try:
                from browser_tools import scrape_page
                content = scrape_page(url)
                context_blocks.append(f"[WEB PAGE: {url}]\n{content}\n[/WEB PAGE]")
            except Exception as e:
                context_blocks.append(f"[Could not load page: {e}]")

        elif _needs_search(message):
            yield f"data: {json.dumps({'type': 'status', 'text': '🔍 Searching the web...'})}\n\n"
            try:
                from browser_tools import web_search
                results = web_search(message)
                context_blocks.append(f"[WEB RESULTS]\n{results}\n[/WEB RESULTS]")
            except Exception as e:
                context_blocks.append(f"[Search failed: {e}]")

        if _needs_image(message):
            yield f"data: {json.dumps({'type': 'status', 'text': '🎨 Generating image...'})}\n\n"
            try:
                from agent_tools import generate_image
                img_url = generate_image(message)
                context_blocks.append(f"[GENERATED IMAGE URL: {img_url}]")
            except Exception as e:
                context_blocks.append(f"[Image generation failed: {e}]")

        # Build augmented user message
        augmented = message
        if context_blocks:
            augmented = "\n\n".join(context_blocks) + f"\n\nUser question: {message}"

        messages.append({"role": "user", "content": augmented})

        try:
            stream = _client.chat.completions.create(
                model=MODELS[effective_model],
                messages=messages,
                max_tokens=4096,
                stream=True,
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

    # ── Persist ───────────────────────────────────────────────────────────────
    if sb and user_id and actual_chat_id:
        try:
            sb.table("messages").insert([
                {"chat_id": actual_chat_id, "role": "user",      "content": message},
                {"chat_id": actual_chat_id, "role": "assistant", "content": full_response}
            ]).execute()
        except Exception as e:
            print("Save error:", e)
    else:
        entry = _conversations.setdefault(session_id, [])
        entry += [{"role": "user", "content": message}, {"role": "assistant", "content": full_response}]
        if len(entry) > 40:
            _conversations[session_id] = entry[-40:]

    yield f"data: {json.dumps({'type': 'done', 'model': effective_model})}\n\n"


def reset_session(session_id: str) -> None:
    _conversations.pop(session_id, None)
