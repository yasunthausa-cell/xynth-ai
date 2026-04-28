"""HTTP API for Xynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded). Replies asynchronously.
- GET  /health
"""
import os
import datetime
import threading
from flask import Flask, request, jsonify, Response, send_from_directory
from langchain_core.messages import HumanMessage
from twilio.twiml.messaging_response import MessagingResponse

from agent import build_agent
import scheduler as _sched
import messaging as _msg

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)
agent, system_prompt = build_agent()
_seen_sessions = set()

print(f"Twilio configured: {_msg.configured()}, from={_msg.TWILIO_FROM!r}")


def _run_agent(session_id: str, message: str) -> str:
    is_first = session_id not in _seen_sessions
    _seen_sessions.add(session_id)
    messages = [system_prompt, HumanMessage(content=message)] if is_first else [HumanMessage(content=message)]
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 15}

    try:
        final = None
        for chunk in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            final = chunk
        return final["messages"][-1].content if final else "(no response)"
    except Exception as e:
        err = str(e)
        if "tool_use_failed" in err or "GraphRecursionError" in err:
            try:
                retry = [HumanMessage(content=message + "\n\n(Use the minimum number of tool calls. Pick ONE tool per need. Stop and answer once you have enough info.)")]
                final = None
                for chunk in agent.stream({"messages": retry}, config=config, stream_mode="values"):
                    final = chunk
                return final["messages"][-1].content if final else f"Error: {err}"
            except Exception as e2:
                return f"❌ Error: {str(e2)}"
        return f"❌ Error: {err}"


def _chunk_for_whatsapp(text: str, limit: int = 1500):
    return _msg.chunk_text(text, limit)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "twilio_configured": _msg.configured()})


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
    return jsonify({"response": _run_agent(session_id, message)})


def _send_whatsapp(to_number: str, text: str):
    _msg.send_text(to_number, text)


def _augment_with_context(from_number: str, body: str) -> str:
    """Prepend lightweight context so the agent knows who/when it's talking to.
    Useful for scheduling tasks where the recipient defaults to the current user.
    """
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (f"[Context — current user WhatsApp: {from_number} | server time: {now_utc}. "
            f"When scheduling tasks for the user, use this number as the recipient unless they say otherwise.]\n\n"
            f"User: {body}")


def _process_whatsapp_async(from_number: str, body: str):
    """Run the agent in a background thread and push the reply via Twilio REST."""
    session_id = f"wa-{from_number}"
    augmented = _augment_with_context(from_number, body)
    try:
        reply = _run_agent(session_id, augmented)
    except Exception as e:
        reply = f"❌ Error: {e}"
    _send_whatsapp(from_number, reply)
    print(f"✅ Async reply sent to {from_number}")


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
