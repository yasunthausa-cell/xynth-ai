"""HTTP API for Xynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded)
- GET  /health
"""
import os
from flask import Flask, request, jsonify, Response
from langchain_core.messages import HumanMessage
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from agent import build_agent

app = Flask(__name__)
agent, system_prompt = build_agent()
_seen_sessions = set()

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_WHATSAPP_NUMBER")  # e.g. "whatsapp:+14155238886"
_twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None


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
    """WhatsApp messages have a hard limit around 1600 chars. Split safely."""
    text = text or ""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < 200:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "twilio_configured": bool(_twilio and TWILIO_FROM)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "(empty message)"}), 400
    return jsonify({"response": _run_agent(session_id, message)})


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Twilio WhatsApp webhook. Twilio sends form-encoded data."""
    from_number = request.form.get("From", "")          # e.g. "whatsapp:+1234567890"
    body = (request.form.get("Body") or "").strip()
    print(f"💬 WhatsApp from {from_number}: {body[:80]}")

    if not body:
        resp = MessagingResponse()
        resp.message("(empty message)")
        return Response(str(resp), mimetype="application/xml")

    session_id = f"wa-{from_number}"
    reply = _run_agent(session_id, body)
    chunks = _chunk_for_whatsapp(reply)

    # If we have Twilio creds, send extra chunks via REST (TwiML can carry only so much).
    # The first chunk goes back as the immediate webhook reply.
    twiml = MessagingResponse()
    if chunks:
        twiml.message(chunks[0])
    if len(chunks) > 1 and _twilio and TWILIO_FROM:
        for extra in chunks[1:]:
            try:
                _twilio.messages.create(from_=TWILIO_FROM, to=from_number, body=extra)
            except Exception as e:
                print(f"Failed to send follow-up chunk: {e}")

    return Response(str(twiml), mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_API_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
