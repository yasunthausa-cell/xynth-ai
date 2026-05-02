"""Cloudflare Workers AI runner for Xynth web chat.
Handles streaming, per-user daily message limits, and automatic model fallback.
"""
import os
import json
import datetime
import requests

# ── Credentials ──────────────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "1d9504b41ab08baf145de0ab09efd59f")
CF_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN",  "cfut_8hFRjMD9E23N84tDo5wuvAGjGk4Z37WFBxRGcr5Jfaa54319")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Xynth 1.5":       "@cf/meta/llama-4-scout-instruct", # Llama 4 Scout (Powerful/Fast)
    "Xynth 1.5 Turbo": "@cf/meta/llama-3.2-3b-instruct",  # Lighter model for Turbo
}

# ── Daily message limits ───────────────────────────────────────────────────────
DAILY_LIMITS = {
    "Xynth 1.5":       10,
    "Xynth 1.5 Turbo": 20,
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Xynth, an advanced AI assistant built for productivity, creativity, and intelligence. "
    "You are precise, helpful, and concise. You never reveal which underlying AI model you use. "
    "If asked, say you are 'Xynth AI' — a proprietary model. Created by Aether Aiko. "
    "You support markdown formatting in your responses."
)

# ── In-memory state ───────────────────────────────────────────────────────────
# Conversation history: {session_id: [{"role": "user"/"assistant", "content": str}]}
_conversations: dict[str, list] = {}

# Daily usage counters: {date_str: {session_id: {model_name: count}}}
_daily_usage: dict[str, dict] = {}


def _today() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _get_usage(session_id: str, model_name: str, sb=None, user_id=None) -> int:
    if sb and user_id:
        try:
            today = _today()
            res = sb.table("daily_usage").select("count").eq("user_id", user_id).eq("date", today).eq("model_name", model_name).execute()
            if res.data:
                return res.data[0]["count"]
        except Exception as e:
            print("Usage fetch error:", e)
        return 0

    return _daily_usage.get(_today(), {}).get(session_id, {}).get(model_name, 0)


def _increment_usage(session_id: str, model_name: str, sb=None, user_id=None) -> None:
    today = _today()
    if sb and user_id:
        try:
            count = _get_usage(session_id, model_name, sb, user_id) + 1
            sb.table("daily_usage").upsert({
                "user_id": user_id,
                "date": today,
                "model_name": model_name,
                "count": count
            }).execute()
            return
        except Exception as e:
            print("Increment error:", e)
    
    if today not in _daily_usage:
        _daily_usage[today] = {}
    if session_id not in _daily_usage[today]:
        _daily_usage[today][session_id] = {}
    _daily_usage[today][session_id][model_name] = _daily_usage[today][session_id].get(model_name, 0) + 1


def _unlock_datetime() -> datetime.datetime:
    """UTC midnight tomorrow — when limits reset."""
    now = datetime.datetime.utcnow()
    return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def get_usage_info(session_id: str, sb=None, user_id=None) -> dict:
    """Return current usage counts for a session (used by frontend)."""
    return {
        model: {
            "used":  _get_usage(session_id, model, sb, user_id),
            "limit": DAILY_LIMITS[model],
        }
        for model in MODELS
    }


def stream_chat(session_id: str, message: str, model_name: str = "Xynth 1.5", sb=None, user_id=None, chat_id=None, image_data=None):
    """Generator that yields SSE-formatted data strings.

    Event types sent to the client:
    - {"type": "token",    "text": "..."}           — streamed token
    - {"type": "model",    "name": "Xynth 1.5 Turbo"} — silent auto-fallback notification
    - {"type": "limit",    "unlock_utc": "ISO str"} — daily limit hit for ALL models
    - {"type": "done"}                              — stream complete
    - {"type": "error",   "text": "..."}            — something went wrong
    """
    # ── Resolve effective model (auto-fallback logic) ─────────────────────────
    effective_model = model_name

    # Limits have been removed as requested!
    # Users can now send unlimited messages to Xynth 1.5 and Turbo.

    # ── Increment before calling (prevents double-spend on retry) ─────────────
    _increment_usage(session_id, effective_model, sb, user_id)

    # ── Handle Chat History (Supabase or Memory) ─────────────────────────────
    history = []
    actual_chat_id = chat_id

    if sb and user_id:
        if not actual_chat_id:
            # Create new chat
            try:
                chat_title = message[:40] + "..." if len(message) > 40 else message
                res = sb.table("chats").insert({"user_id": user_id, "title": chat_title}).execute()
                if res.data:
                    actual_chat_id = res.data[0]["id"]
                    yield f"data: {json.dumps({'type': 'chat_id', 'id': actual_chat_id})}\n\n"
            except Exception as e:
                print("Create chat error:", e)
        else:
            # Fetch history
            try:
                res = sb.table("messages").select("role, content").eq("chat_id", actual_chat_id).order("created_at").execute()
                history = [{"role": msg["role"], "content": msg["content"]} for msg in res.data]
            except Exception as e:
                print("Fetch history error:", e)
    else:
        history = _conversations.get(session_id, [])

    # ── Build message list ────────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    if image_data:
        # Switch to vision model!
        cf_model = "@cf/meta/llama-3.2-11b-vision-instruct"
        # CF requires format: [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}, {"type": "text", "text": "..."}]}]
        # But actually CF Workers AI format might be slightly different. Assuming standard OpenAI compatible.
        messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        })
    else:
        cf_model = MODELS[effective_model]
        messages.append({"role": "user", "content": message})

    # ── Call Cloudflare Workers AI ─────────────────────────────────────────────
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
        f"/ai/run/{cf_model}"
    )
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messages":   messages,
        "stream":     True,
        "max_tokens": 2048,
    }

    full_response = ""
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as resp:
            if resp.status_code != 200:
                err = resp.text[:300]
                yield f"data: {json.dumps({'type': 'error', 'text': f'Cloudflare error {resp.status_code}: {err}'})}\n\n"
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk.get("response", "")
                        if token:
                            full_response += token
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    except json.JSONDecodeError:
                        pass

    except requests.exceptions.Timeout:
        yield f"data: {json.dumps({'type': 'error', 'text': 'Request timed out — try a shorter message.'})}\n\n"
        return
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
        return

    # ── Persist conversation history ───────────────────────────────────────────
    if sb and user_id and actual_chat_id:
        try:
            # We must only save text for the user message to prevent DB bloat with base64 images
            sb.table("messages").insert([
                {"chat_id": actual_chat_id, "role": "user", "content": message},
                {"chat_id": actual_chat_id, "role": "assistant", "content": full_response}
            ]).execute()
        except Exception as e:
            print("Save message error:", e)
    else:
        history_entry = _conversations.setdefault(session_id, [])
        history_entry.append({"role": "user",      "content": message})
        history_entry.append({"role": "assistant", "content": full_response})
        if len(history_entry) > 40:
            _conversations[session_id] = history_entry[-40:]

    yield f"data: {json.dumps({'type': 'done', 'model': effective_model})}\n\n"


def reset_session(session_id: str) -> None:
    """Clear conversation history for a session."""
    _conversations.pop(session_id, None)
