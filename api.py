"""HTTP API for Xynth AI. Used by the WhatsApp bot (and any other client)."""
import os
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage
from agent import build_agent

app = Flask(__name__)
agent, system_prompt = build_agent()
_seen_sessions = set()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "(empty message)"}), 400

    is_first = session_id not in _seen_sessions
    _seen_sessions.add(session_id)

    messages = [system_prompt, HumanMessage(content=message)] if is_first else [HumanMessage(content=message)]
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 15,
    }

    try:
        final = None
        for chunk in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            final = chunk
        reply = final["messages"][-1].content if final else "(no response)"
        return jsonify({"response": reply})
    except Exception as e:
        err = str(e)
        if "tool_use_failed" in err or "GraphRecursionError" in err:
            try:
                retry = [HumanMessage(content=message + "\n\n(Use the minimum number of tool calls. Pick ONE tool per need. Stop and answer once you have enough info.)")]
                final = None
                for chunk in agent.stream({"messages": retry}, config=config, stream_mode="values"):
                    final = chunk
                reply = final["messages"][-1].content if final else f"Error: {err}"
                return jsonify({"response": reply})
            except Exception as e2:
                return jsonify({"response": f"❌ Error: {str(e2)}"}), 500
        return jsonify({"response": f"❌ Error: {err}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_API_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
