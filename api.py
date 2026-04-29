"""HTTP API for Xynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded). Replies asynchronously.
- GET  /health
"""
import os
import json
import datetime
import threading
import traceback
import concurrent.futures
from flask import Flask, request, jsonify, Response, send_from_directory, render_template
from langchain_core.messages import HumanMessage

from agent import XynthRunner
import scheduler as _sched
import messaging as _msg

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)
print("🚀 Initialising Xynth model chain…")
runner = XynthRunner()
print(f"Meta WA configured: {_msg.configured()}")


def _run_agent(session_id: str, message: str) -> str:
    return runner.run(session_id, message)


def _chunk_for_whatsapp(text: str, limit: int = 1500):
    return _msg.chunk_text(text, limit)


@app.route("/", methods=["GET"])
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


def _augment_with_context(from_number: str | None, body: str) -> str:
    """Prepend lightweight context so the agent knows the current date and (if WhatsApp) who.
    The LLMs' training data is older, so we MUST inject the real current date every turn.
    """
    now_utc = datetime.datetime.utcnow()
    pretty = now_utc.strftime("%A, %d %B %Y, %H:%M UTC")
    if from_number:
        ctx = (f"[Context — TODAY IS {pretty}. Current user WhatsApp: {from_number}. "
               f"When scheduling tasks for the user, use this number as the recipient unless they say otherwise. "
               f"Trust this date over any internal knowledge.]")
    else:
        ctx = (f"[Context — TODAY IS {pretty}. Trust this date over any internal knowledge.]")
    return f"{ctx}\n\nUser: {body}"


_AGENT_TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "90"))


def _process_whatsapp_async(from_number: str, body: str):
    """Run the agent in a background thread and push the reply via Twilio REST.
    Hard timeout so a hanging tool can never silently swallow the reply.
    """
    session_id = f"wa-{from_number}"
    augmented = _augment_with_context(from_number, body)
    reply = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_agent, session_id, augmented)
            try:
                reply = fut.result(timeout=_AGENT_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                reply = (f"⏱️ Sorry, that took longer than {_AGENT_TIMEOUT_SECONDS}s and I had to stop. "
                         f"Try a simpler request, or break it into smaller steps.")
                print(f"⚠️  Agent timeout for {from_number} on message: {body[:80]!r}")
    except Exception as e:
        traceback.print_exc()
        reply = f"❌ Internal error: {e}"

    if not reply:
        reply = "(no response)"
    try:
        _send_whatsapp(from_number, reply)
        print(f"✅ Async reply sent to {from_number} ({len(reply)} chars)")
    except Exception:
        traceback.print_exc()


# Initialise the scheduler now that _run_agent and _send_whatsapp exist.
_sched.init_scheduler(run_agent_fn=_run_agent, send_whatsapp_fn=_send_whatsapp)


@app.route("/whatsapp", methods=["GET", "POST"])
def whatsapp_webhook():
    """Meta WhatsApp Cloud API webhook."""
    if request.method == "GET":
        # Meta webhook verification
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        
        verify_token = os.environ.get("META_WA_VERIFY_TOKEN", "")
        if mode == "subscribe" and token == verify_token:
            return challenge, 200
        return "Forbidden", 403

    # Handle POST (incoming messages)
    data = request.get_json(silent=True) or {}
    
    # Extract message details
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        
        if not messages:
            # Maybe a status update, just acknowledge
            return "OK", 200
            
        message = messages[0]
        from_number = message.get("from", "")
        
        # Meta can send different types, handle text for now
        msg_type = message.get("type", "")
        if msg_type == "text":
            body = message.get("text", {}).get("body", "").strip()
        else:
            body = f"(User sent a {msg_type} message which is not supported yet)"
            
        print(f"💬 WhatsApp from {from_number}: {body[:80]}")
        
        if body:
            # Kick off background processing and return 200 immediately
            threading.Thread(
                target=_process_whatsapp_async,
                args=(from_number, body),
                daemon=True,
            ).start()
            
    except Exception as e:
        print(f"Error parsing Meta WhatsApp webhook: {e}")
        
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("AGENT_API_PORT") or "5000")
    app.run(host="0.0.0.0", port=port, debug=False)
