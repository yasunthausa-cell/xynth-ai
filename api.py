"""HTTP API for Xynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded). Replies asynchronously.
- GET  /health
"""
import os
import io
import json
import base64
import random
import datetime
import threading
import traceback
import concurrent.futures
import requests
from flask import Flask, request, jsonify, Response, send_from_directory, render_template, stream_with_context
from langchain_core.messages import HumanMessage

from agent import XynthRunner
import scheduler as _sched
import messaging as _msg

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Supabase (optional – gracefully disabled if keys missing) ────────────────
try:
    from supabase import create_client as _sb_create
    _SB_URL = os.environ.get("SUPABASE_URL", "")
    _SB_KEY = os.environ.get("SUPABASE_KEY", "")
    _sb = _sb_create(_SB_URL, _SB_KEY) if _SB_URL and _SB_KEY else None
except Exception:
    _sb = None

app = Flask(__name__)
print("🚀 Initialising Xynth model chain…")
runner = XynthRunner()
print(f"Meta WA configured: {_msg.configured()}")


def _run_agent(session_id: str, message: str) -> str:
    return runner.run(session_id, message)


def _chunk_for_whatsapp(text: str, limit: int = 1500):
    return _msg.chunk_text(text, limit)


@app.route("/")
def landing():
    """Serve the landing page."""
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    """Serve the main chat interface."""
    return render_template("chat.html")


@app.route("/home", methods=["GET"])
def home():
    """Serve the web chat UI."""
    return render_template("chat.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "meta_wa_configured": _msg.configured()})


@app.route("/models", methods=["GET", "POST"])
def models_endpoint():
    """GET → list of models + active. POST {model:'name'} → switch active."""
    if request.method == "GET":
        names = [m for m, _ in runner.agents]
        return jsonify({"models": names, "active": runner.current_model})
    data = request.get_json(silent=True) or {}
    name = (data.get("model") or "").strip()
    for i, (m, _) in enumerate(runner.agents):
        if m == name or name.lower() in m.lower():
            runner.current_idx = i
            return jsonify({"ok": True, "active": runner.current_model})
    return jsonify({"ok": False, "error": f"No model matching '{name}'"}), 400


@app.route("/reset", methods=["POST"])
def reset_session():
    """Wipe a session's memory across all models."""
    data = request.get_json(silent=True) or {}
    sid = (data.get("session_id") or "").strip()
    if not sid:
        return jsonify({"ok": False, "error": "session_id required"}), 400
    runner.seen_sessions = {k for k in runner.seen_sessions if k[1] != sid}
    return jsonify({"ok": True})


@app.route("/media/<path:filename>", methods=["GET"])
def serve_media(filename):
    """Serve generated media files (screenshots, etc.) so Twilio can fetch them."""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/manifest.json", methods=["GET"])
def serve_manifest():
    """Serve the PWA manifest."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return send_from_directory(static_dir, "manifest.json")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "(empty message)"}), 400
    augmented = _augment_with_context(None, message)
    return jsonify({"response": _run_agent(session_id, augmented)})


@app.route("/chat/stream", methods=["GET"])
def chat_stream():
    """Server-Sent Events stream of the agent's progress for the web UI."""
    session_id = str(request.args.get("session_id", "default"))
    message = (request.args.get("message") or "").strip()
    if not message:
        return Response("data: {\"type\":\"error\",\"message\":\"empty\"}\n\n", mimetype="text/event-stream")
    augmented = _augment_with_context(None, message)

    def generate():
        try:
            for evt in runner.stream_run(session_id, augmented):
                yield f"data: {json.dumps(evt)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(generate(), headers=headers)


@app.route("/usage", methods=["GET"])
def usage_endpoint():
    """Return today's per-model token usage and remaining budget."""
    return jsonify(runner.usage_summary())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Accept an audio file from the browser and return transcribed text.
    Uses fast Whisper via Groq (free tier, supports English + Sinhala + Tamil + 90+ langs).
    """
    import requests as _rq
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400
    audio = request.files["audio"]
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "transcription unavailable"}), 500
    try:
        r = _rq.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio.filename or "voice.webm", audio.stream, audio.mimetype or "audio/webm")},
            data={"model": "whisper-large-v3-turbo", "response_format": "json"},
            timeout=30,
        )
        if r.status_code != 200:
            return jsonify({"error": f"transcription failed: {r.text[:200]}"}), 500
        return jsonify({"text": (r.json().get("text") or "").strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _send_whatsapp(to_number: str, text: str):
    _msg.send_text(to_number, text)


# ── Long-term memory ──────────────────────────────────────────────────────────
def _load_memories(session_id: str) -> str:
    if not _sb:
        return ""
    try:
        res = _sb.table("memories").select("fact").eq("session_id", session_id).execute()
        facts = [r["fact"] for r in (res.data or [])]
        if facts:
            return "[Long-term memory — facts about this user:\n" + "\n".join(f"• {f}" for f in facts) + "]\n\n"
    except Exception:
        pass
    return ""


def _save_memory(session_id: str, fact: str):
    if not _sb or not fact:
        return
    try:
        _sb.table("memories").insert({"session_id": session_id, "fact": fact[:500]}).execute()
    except Exception:
        pass


# ── WhatsApp media download ───────────────────────────────────────────────────
def _download_wa_media(media_id: str) -> bytes | None:
    token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    if not token:
        return None
    try:
        meta = requests.get(
            f"https://graph.facebook.com/v19.0/{media_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        ).json()
        url = meta.get("url")
        if not url:
            return None
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        return resp.content
    except Exception as e:
        print(f"⚠️  Media download failed: {e}")
        return None


def _augment_with_context(from_number: str | None, body: str, session_id: str | None = None) -> str:
    """Prepend date, user identity, and long-term memories into the message."""
    now_utc = datetime.datetime.utcnow()
    pretty = now_utc.strftime("%A, %d %B %Y, %H:%M UTC")
    memory_block = _load_memories(session_id) if session_id else ""
    if from_number:
        ctx = (f"[Context — TODAY IS {pretty}. Current user WhatsApp: {from_number}. "
               f"When scheduling tasks for the user, use this number. "
               f"Trust this date over any internal knowledge.]")
    else:
        ctx = f"[Context — TODAY IS {pretty}. Trust this date over any internal knowledge.]"
    return f"{memory_block}{ctx}\n\nUser: {body}"


_AGENT_TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "90"))


def _process_whatsapp_async(from_number: str, body: str):
    """Run the agent in a background thread. Sends instant ACK first, then full reply."""
    session_id = f"wa-{from_number}"

    # Instant acknowledgement so user knows bot received the message
    acks = ["⏳ On it! Give me a moment…", "🔍 Let me look into that…",
            "🧠 Thinking… back in a sec!", "⚡ Working on your request…"]
    _send_whatsapp(from_number, random.choice(acks))

    augmented = _augment_with_context(from_number, body, session_id)
    reply = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_agent, session_id, augmented)
            try:
                reply = fut.result(timeout=_AGENT_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                reply = (f"⏱️ Sorry, that took longer than {_AGENT_TIMEOUT_SECONDS}s. "
                         f"Try a simpler request or break it into steps.")
                print(f"⚠️  Agent timeout for {from_number}")
    except Exception as e:
        traceback.print_exc()
        reply = f"❌ Internal error: {e}"

    if not reply:
        reply = "(no response)"
    try:
        _send_whatsapp(from_number, reply)
        print(f"✅ Reply sent to {from_number} ({len(reply)} chars)")
    except Exception:
        traceback.print_exc()


# Initialise the scheduler now that _run_agent and _send_whatsapp exist.
_sched.init_scheduler(run_agent_fn=_run_agent, send_whatsapp_fn=_send_whatsapp)


@app.route("/whatsapp", methods=["GET", "POST"])
def whatsapp_webhook():
    """Meta WhatsApp Cloud API webhook — handles text, voice, and image messages."""
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == os.environ.get("META_WA_VERIFY_TOKEN", ""):
            return challenge, 200
        return "Forbidden", 403

    data = request.get_json(silent=True) or {}
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return "OK", 200

        message = messages[0]
        from_number = message.get("from", "")
        msg_type = message.get("type", "")
        body = ""

        if msg_type == "text":
            body = message.get("text", {}).get("body", "").strip()

        elif msg_type in ("audio", "voice"):
            # ── Voice note → transcribe via Whisper ──────────────────────────
            media_id = (message.get("audio") or message.get("voice") or {}).get("id")
            audio_bytes = _download_wa_media(media_id) if media_id else None
            if audio_bytes:
                groq_key = os.environ.get("GROQ_API_KEY", "")
                try:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files={"file": ("voice.ogg", io.BytesIO(audio_bytes), "audio/ogg")},
                        data={"model": "whisper-large-v3-turbo", "response_format": "json"},
                        timeout=30,
                    )
                    body = r.json().get("text", "").strip()
                    print(f"🎤 Transcribed: {body[:80]}")
                except Exception as e:
                    body = "(voice message — transcription failed)"
                    print(f"⚠️ Transcription error: {e}")
            else:
                body = "(voice message — could not download audio)"

        elif msg_type == "image":
            # ── Image → base64 → vision analysis ────────────────────────────
            media_id = message.get("image", {}).get("id")
            caption = message.get("image", {}).get("caption", "").strip()
            img_bytes = _download_wa_media(media_id) if media_id else None
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode()
                body = (
                    f"[User sent an image. Analyse it and respond.]\n"
                    f"data:image/jpeg;base64,{b64}\n"
                    f"User caption: {caption or '(no caption)'}"
                )
                print(f"🖼️ Image received ({len(img_bytes)} bytes)")
            else:
                body = "(image message — could not download)"

        else:
            body = f"(User sent a '{msg_type}' message)"

        print(f"💬 WA from {from_number} [{msg_type}]: {body[:80]}")
        if body:
            threading.Thread(
                target=_process_whatsapp_async,
                args=(from_number, body),
                daemon=True,
            ).start()

    except Exception as e:
        print(f"WhatsApp webhook error: {e}")

    return "OK", 200


@app.route("/memory", methods=["POST"])
def save_memory_endpoint():
    """Web UI or external caller can POST {session_id, fact} to save a memory."""
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    fact = (data.get("fact") or "").strip()
    if session_id and fact:
        _save_memory(session_id, fact)
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400


@app.route('/proxy/image')
def proxy_image():
    url = request.args.get('url')
    if not url:
        return "No url provided", 400
    try:
        r = requests.get(url, stream=True, timeout=20)
        return Response(
            stream_with_context(r.iter_content(chunk_size=4096)),
            content_type=r.headers.get('Content-Type', 'image/jpeg')
        )
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("AGENT_API_PORT") or "5000")
    app.run(host="0.0.0.0", port=port, debug=False)
