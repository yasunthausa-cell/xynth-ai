"""HTTP API for Xynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded). Replies asynchronously.
- GET  /health
"""
import os
import datetime
import threading
import traceback
import concurrent.futures
from flask import Flask, request, jsonify, Response, send_from_directory, render_template
from langchain_core.messages import HumanMessage
from twilio.twiml.messaging_response import MessagingResponse

from agent import XynthRunner
import scheduler as _sched
import messaging as _msg

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)
print("🚀 Initialising Xynth model chain…")
runner = XynthRunner()
print(f"Twilio configured: {_msg.configured()}, from={_msg.TWILIO_FROM!r}")


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
    return jsonify({"status": "ok", "twilio_configured": _msg.configured()})


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


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Twilio WhatsApp webhook. Twilio sends form-encoded data.

    To avoid Twilio's 15s webhook timeout, we acknowledge immediately and
    process the agent in a background thread. The real reply is sent via
    Twilio's REST API when the agent finishes.
    """
    from_number = request.form.get("From", "")          # e.g. "whatsapp:+1234567890"
    body = (request.form.get("Body") or "").strip()
    print(f"💬 WhatsApp from {from_number}: {body[:80]}")

    twiml = MessagingResponse()

    if not body:
        twiml.message("(empty message)")
        return Response(str(twiml), mimetype="application/xml")

    if not _msg.configured():
        # Fallback: synchronous reply if Twilio REST isn't configured.
        reply = _run_agent(f"wa-{from_number}", body)
        twiml.message(_chunk_for_whatsapp(reply)[0] if reply else "(no response)")
        return Response(str(twiml), mimetype="application/xml")

    # Kick off background processing and immediately return a typing indicator.
    threading.Thread(
        target=_process_whatsapp_async,
        args=(from_number, body),
        daemon=True,
    ).start()

    twiml.message("🤖 Thinking…")
    return Response(str(twiml), mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
