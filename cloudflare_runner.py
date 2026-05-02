"""Cloudflare Workers AI runner for Xynth Superagent.
Handles streaming, tool-calling ReAct loop, image generation, and memory.
"""
import os
import json
import datetime
import requests
from agent_tools import TOOL_DEFINITIONS, execute_tool

# ── Credentials ──────────────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "1d9504b41ab08baf145de0ab09efd59f")
CF_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN",  "cfut_8hFRjMD9E23N84tDo5wuvAGjGk4Z37WFBxRGcr5Jfaa54319")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Xynth 1.5":       "@cf/meta/llama-4-scout-17b-16e-instruct",
    "Xynth 1.5 Turbo": "@cf/meta/llama-3.1-8b-instruct",
}

# ── Daily message limits ───────────────────────────────────────────────────────
DAILY_LIMITS = {
    "Xynth 1.5":       10,
    "Xynth 1.5 Turbo": 20,
}

# ── System Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Xynth, an advanced autonomous AI assistant with access to powerful tools. "
    "You are built for productivity, creativity, and intelligence. "
    "You NEVER reveal which underlying AI model you use. If asked, say you are 'Xynth AI' — a proprietary model. Created by Aether Aiko. "
    "You support markdown formatting in your responses. "
    "When you need to search the web, run code, generate images, or use any tool, USE IT — don't just describe what you would do. "
    "When generating an image, always embed the returned image URL in your response using markdown: ![description](url). "
    "Always be precise and cite your sources when using web data. "
    "CRITICAL RULE: If you are not 100% certain about a fact — especially anything involving current events, prices, people, statistics, or recent news — "
    "DO NOT guess or make up an answer. Instead, use the web_search tool to verify it first. "
    "It is better to search and be right than to answer from memory and be wrong. "
    "Never say 'as of my knowledge cutoff' — just search for the real answer instead."
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



def _generate_chat_title(message: str) -> str:
    """Use Cloudflare Llama to generate a short, smart chat title from the first message."""
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{MODELS['Xynth 1.5 Turbo']}"
        headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": "Generate a very short chat title (max 5 words, no quotes, no punctuation) that summarizes the user's message. Reply with ONLY the title, nothing else."},
                {"role": "user", "content": message}
            ],
            "stream": False,
            "max_tokens": 20,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            title = resp.json().get("result", {}).get("response", "").strip()
            if title:
                return title[:60]  # Cap at 60 chars just in case
    except Exception as e:
        print("Title generation error:", e)
    # Fallback: use first 40 chars
    return message[:40] + ("..." if len(message) > 40 else "")


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
            # Create new chat with AI-generated title
            try:
                ai_title = _generate_chat_title(message)
                res = sb.table("chats").insert({"user_id": user_id, "title": ai_title}).execute()
                if res.data:
                    actual_chat_id = res.data[0]["id"]
                    yield f"data: {json.dumps({'type': 'chat_id', 'id': actual_chat_id, 'title': ai_title})}\n\n"
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

    # ── Build base message list ────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    if image_data:
        # Vision mode — no tools, just direct vision call with streaming
        cf_model = "@cf/meta/llama-3.2-11b-vision-instruct"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        })
        full_response = ""
        cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{cf_model}"
        cf_headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
        try:
            with requests.post(cf_url, headers=cf_headers, json={"messages": messages, "stream": True, "max_tokens": 2048}, stream=True, timeout=90) as resp:
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Cloudflare error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                    return
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = str(chunk.get("response", ""))
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                        except json.JSONDecodeError:
                            pass
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            return
    else:
        # ── Full ReAct Agent Loop ──────────────────────────────────────────────
        # Xynth 1.5 Turbo doesn't support tool calling — fall back to direct call
        cf_model = MODELS[effective_model]
        turbo_mode = (effective_model == "Xynth 1.5 Turbo")

        messages.append({"role": "user", "content": message})
        cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{cf_model}"
        cf_headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}

        full_response = ""
        MAX_TOOL_ROUNDS = 5  # Prevent infinite loops

        for tool_round in range(MAX_TOOL_ROUNDS):
            # First: non-streaming call to detect tool use
            payload = {
                "messages": messages,
                "stream": False,
                "max_tokens": 2048,
            }
            if not turbo_mode:
                payload["tools"] = TOOL_DEFINITIONS

            try:
                resp = requests.post(cf_url, headers=cf_headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Cloudflare error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                    return

                result = resp.json().get("result", {})
                assistant_msg = result.get("response") or ""
                tool_calls = result.get("tool_calls") or []

                if not tool_calls:
                    # No tool calls — stream the final answer
                    break

                # ── Execute all tool calls ─────────────────────────────────
                messages.append({"role": "assistant", "content": assistant_msg or "", "tool_calls": tool_calls})

                for tc in tool_calls:
                    tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                    raw_args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                    tool_args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
                    tool_id = tc.get("id", tool_name)

                    # Emit status to user
                    status_labels = {
                        "web_search": "🔍 Searching the web...",
                        "scrape_page": "🌐 Browsing page...",
                        "run_python": "🐍 Running code...",
                        "generate_image": "🎨 Generating image...",
                        "calculator": "🧮 Calculating...",
                        "wikipedia_search": "📖 Checking Wikipedia...",
                        "send_email": "📧 Sending email...",
                        "memory_read": "🧠 Reading memory...",
                        "memory_write": "🧠 Saving memory...",
                    }
                    status_text = status_labels.get(tool_name, f"⚙️ Running {tool_name}...")
                    yield f"data: {json.dumps({'type': 'status', 'text': status_text})}\n\n"

                    print(f"[Agent] Tool call: {tool_name}({tool_args})")
                    tool_result = execute_tool(tool_name, tool_args, user_id=user_id, sb=sb)
                    print(f"[Agent] Tool result ({tool_name}): {str(tool_result)[:200]}")

                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_id,
                        "name": tool_name,
                    })

            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
                return

        # ── Stream final answer ────────────────────────────────────────────────
        # Make a fresh streaming call with all tool results in context
        final_payload = {"messages": messages, "stream": True, "max_tokens": 2048}
        try:
            with requests.post(cf_url, headers=cf_headers, json=final_payload, stream=True, timeout=90) as resp:
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Cloudflare error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                    return
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = str(chunk.get("response", ""))
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                        except json.JSONDecodeError:
                            pass
        except requests.exceptions.Timeout:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Request timed out.'})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            return

    # ── Persist conversation history ───────────────────────────────────────────
    if sb and user_id and actual_chat_id:
        try:
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
