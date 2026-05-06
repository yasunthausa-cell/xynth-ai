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
import websocket
import ssl
from flask import Flask, request, jsonify, Response, send_from_directory, render_template, stream_with_context
from flask_sock import Sock
from langchain_core.messages import HumanMessage

from agent import XynthRunner
import scheduler as _sched
import messaging as _msg
import groq_runner as _cf
import builder_runner as _builder
import research_runner as _research
import rag_pipeline as _rag
from limits import check_and_increment, get_user_plan, get_usage_today

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
sock = Sock(app)
print("🚀 Initialising Xynth model chain…")
runner = XynthRunner()
print(f"Meta WA configured: {_msg.configured()}")

@sock.route('/ws/voice')
def voice_realtime(ws):
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        ws.close()
        return

    # Use Alibaba's OpenAI-compatible Realtime endpoint
    target_url = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-omni-plus-realtime"
    
    dash_ws = websocket.WebSocketApp(
        target_url,
        header=[f"Authorization: Bearer {api_key}"]
    )
    
    # Relay from DashScope to Frontend
    def on_message(ws_app, message):
        try:
            ws.send(message)
        except Exception:
            pass
            
    dash_ws.on_message = on_message
    
    def run_dash_ws():
        dash_ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
    t = threading.Thread(target=run_dash_ws)
    t.daemon = True
    t.start()
    
    # Relay from Frontend to DashScope
    try:
        while True:
            data = ws.receive()
            if dash_ws.sock and dash_ws.sock.connected:
                dash_ws.send(data)
    except Exception as e:
        print(f"Voice WS disconnected: {e}")
    finally:
        if dash_ws.sock and dash_ws.sock.connected:
            dash_ws.close()


def _run_agent(session_id: str, message: str) -> str:
    return runner.run(session_id, message)


def _chunk_for_whatsapp(text: str, limit: int = 1500):
    return _msg.chunk_text(text, limit)


@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


@app.route("/home", methods=["GET"])
def home():
    return render_template("chat.html")


@app.route("/build", methods=["GET"])
def builder_page():
    """Serve the app builder UI."""
    return render_template("builder.html")


@app.route("/build/stream", methods=["POST"])
def builder_stream():
    """Stream AI-generated code for the builder."""
    body        = request.get_json(force=True) or {}
    message     = body.get("message", "").strip()
    session_id  = body.get("session_id", "anon")
    project_id  = body.get("project_id")

    # Auth (optional)
    token   = request.headers.get("Authorization", "").replace("Bearer ", "")
    user_id = None
    sb      = None
    if token and _sb:
        try:
            user_id = _sb.auth.get_user(token).user.id
            sb = _sb
        except Exception:
            pass

    if not message:
        return jsonify({"error": "message required"}), 400

    gen = _builder.stream_build(session_id, message, sb=sb, user_id=user_id, project_id=project_id)
    return Response(stream_with_context(gen), content_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ── Research Routes ───────────────────────────────────────────────────────────
@app.route("/research", methods=["GET"])
def research_page():
    return render_template("research.html")


@app.route("/research/stream", methods=["POST"])
def research_stream():
    body       = request.get_json(force=True) or {}
    query      = body.get("query", "").strip()
    session_id = body.get("session_id", "anon")
    chat_id    = body.get("chat_id")
    if not query:
        return jsonify({"error": "query required"}), 400

    token   = request.headers.get("Authorization", "").replace("Bearer ", "")
    user_id, sb = None, None
    if token and _sb:
        try:
            user_id = _sb.auth.get_user(token).user.id
            sb = _sb
        except Exception:
            pass

    gen = _research.stream_research(session_id, query, sb=sb, user_id=user_id, chat_id=chat_id)
    return Response(stream_with_context(gen), content_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ── Document Upload (RAG) ─────────────────────────────────────────────────────
@app.route("/documents/upload", methods=["POST"])
def upload_document():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token or not _sb:
        return jsonify({"error": "Authentication required"}), 401
    try:
        user_id = _sb.auth.get_user(token).user.id
    except Exception:
        return jsonify({"error": "Invalid token"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f         = request.files["file"]
    file_bytes = f.read()
    file_type  = f.content_type or "text/plain"
    doc_title  = f.filename or "Untitled"

    result = _rag.store_document(user_id, doc_title, file_bytes, file_type, _sb)
    return jsonify(result)


@app.route("/documents", methods=["GET"])
def list_documents():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token or not _sb:
        return jsonify([]), 200
    try:
        user_id = _sb.auth.get_user(token).user.id
        docs = _rag.list_user_documents(user_id, _sb)
        return jsonify(docs)
    except Exception:
        return jsonify([]), 200


@app.route("/documents/<doc_title>", methods=["DELETE"])
def delete_document(doc_title):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token or not _sb:
        return jsonify({"error": "Auth required"}), 401
    try:
        user_id = _sb.auth.get_user(token).user.id
        ok = _rag.delete_user_document(user_id, doc_title, _sb)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/static_media/<path:filename>", methods=["GET"])
def serve_generated_media(filename):
    """Serve AI-generated images from the static_media directory."""
    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_media")
    return send_from_directory(media_dir, filename)


@app.route("/manifest.json", methods=["GET"])
def serve_manifest():
    """Serve the PWA manifest."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return send_from_directory(static_dir, "manifest.json")

@app.route("/sw.js", methods=["GET"])
def serve_sw():
    """Serve the Service Worker."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return send_from_directory(static_dir, "sw.js")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "(empty message)"}), 400
    augmented = _augment_with_context(None, message)
    return jsonify({"response": _run_agent(session_id, augmented)})


@app.route("/chat/stream", methods=["GET", "POST"])
def chat_stream():
    """Server-Sent Events stream. Supports both GET (legacy) and POST (new CF runner)."""
    # ── New POST path: Cloudflare Workers AI ─────────────────────────────────
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id", "default"))
        message    = (data.get("message") or "").strip()
        model      = (data.get("model")   or "Xynth 1.5").strip()
        chat_id    = str(data.get("chat_id", ""))
        image_data = data.get("image_data", None)

        deep_dive  = bool(data.get("deep_dive", False))

        user_id = None
        if _sb:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                try:
                    user_res = _sb.auth.get_user(token)
                    if user_res and user_res.user:
                        user_id = user_res.user.id
                except Exception as e:
                    print("Auth error:", e)

        if not message:
            return Response(
                'data: {"type":"error","text":"empty message"}\n\n',
                mimetype="text/event-stream"
            )

        # ── Limit check (DB-backed) ───────────────────────────────────────
        plan = get_user_plan(user_id, _sb) if user_id else "guest"
        allowed, used, limit = check_and_increment(session_id, user_id, _sb, plan)
        if not allowed:
            import datetime
            unlock_utc = (datetime.datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + datetime.timedelta(days=1)).isoformat() + "Z"

            def _over_limit():
                yield f'data: {json.dumps({"type":"limit","plan":plan,"limit":limit,"unlock_utc":unlock_utc,"upgrade_url":"/pricing"})}\n\n'
                yield f'data: {json.dumps({"type":"done"})}\n\n'

            return Response(stream_with_context(_over_limit()), headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })

        def generate_cf():
            # Vision queries still go through groq_runner
            if image_data:
                yield from _cf.stream_chat(
                    session_id=session_id, message=message,
                    model_name=model, sb=_sb, user_id=user_id,
                    chat_id=chat_id, image_data=image_data
                )
            else:
                # All text queries → research engine
                yield from _research.stream_research(
                    session_id=session_id, query=message,
                    sb=_sb, user_id=user_id, chat_id=chat_id,
                    deep_dive=deep_dive
                )

        headers = {
            "Content-Type":    "text/event-stream",
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(stream_with_context(generate_cf()), headers=headers)

    # ── Legacy GET path: original LangGraph runner ────────────────────────────
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
    """Return today's per-model token usage and remaining budget (legacy LangGraph runner)."""
    return jsonify(runner.usage_summary())

@app.route("/usage/session")
def usage_session():
    """Return usage info for current session."""
    session_id = request.args.get("session_id", "default")
    user_id = None
    if _sb:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                user_res = _sb.auth.get_user(token)
                if user_res and user_res.user:
                    user_id = user_res.user.id
            except Exception:
                pass
    info = get_usage_today(session_id, user_id, _sb)
    return jsonify(info)


@app.route("/pricing")
def pricing_page():
    """Pricing page."""
    return render_template("pricing.html",
        paddle_client_token=os.environ.get("PADDLE_CLIENT_TOKEN", ""),
        paddle_pro_price_id=os.environ.get("PADDLE_PRO_PRICE_ID", ""),
    )


@app.route("/paddle/webhook", methods=["POST"])
def paddle_webhook():
    """Paddle Billing webhook handler.
    Set this URL in Paddle dashboard under Notifications.
    Verifies signature using PADDLE_WEBHOOK_SECRET env var.
    """
    import hmac, hashlib
    secret = os.environ.get("PADDLE_WEBHOOK_SECRET", "")
    raw_body = request.get_data()
    sig_header = request.headers.get("Paddle-Signature", "")

    # Verify signature
    if secret and sig_header:
        try:
            parts = dict(p.split("=", 1) for p in sig_header.split(";"))
            ts = parts.get("ts", "")
            h1 = parts.get("h1", "")
            signed = f"{ts}:{raw_body.decode()}"
            expected = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, h1):
                return jsonify({"ok": False, "error": "Invalid signature"}), 401
        except Exception as e:
            print("Paddle sig error:", e)
            return jsonify({"ok": False}), 400

    event = request.get_json(silent=True) or {}
    event_type = event.get("event_type", "")
    data = event.get("data", {})
    custom_data = data.get("custom_data") or {}
    user_id = custom_data.get("user_id", "")
    sub_id = data.get("id", "")

    print(f"[Paddle] {event_type} | user={user_id} | sub={sub_id}")

    if not _sb or not user_id:
        return jsonify({"ok": True})

    try:
        if event_type in ("subscription.activated", "subscription.updated"):
            status = data.get("status", "")
            plan = "pro" if status == "active" else "free"
            _sb.table("profiles").upsert({"id": user_id, "plan": plan, "paddle_subscription_id": sub_id}).execute()
            print(f"[Paddle] Set {user_id} → {plan}")

        elif event_type in ("subscription.canceled", "subscription.paused"):
            _sb.table("profiles").upsert({"id": user_id, "plan": "free", "paddle_subscription_id": sub_id}).execute()
            print(f"[Paddle] Downgraded {user_id} → free")

    except Exception as e:
        print("Paddle DB error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True})

@app.route("/chats", methods=["GET"])
def get_chats():
    if not _sb: return jsonify([])
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "): return jsonify([])
    token = auth_header.split(" ")[1]
    try:
        user_res = _sb.auth.get_user(token)
        if not user_res or not user_res.user: return jsonify([])
        res = _sb.table("chats").select("id, title, created_at").eq("user_id", user_res.user.id).order("created_at", desc=True).execute()
        return jsonify(res.data)
    except Exception as e:
        print("GET /chats error:", e)
        return jsonify([])

@app.route("/chats/<chat_id>", methods=["GET"])
def get_chat_messages(chat_id):
    if not _sb: return jsonify([])
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "): return jsonify([])
    token = auth_header.split(" ")[1]
    try:
        user_res = _sb.auth.get_user(token)
        if not user_res or not user_res.user: return jsonify([])
        # Verify ownership
        chat_res = _sb.table("chats").select("id").eq("id", chat_id).eq("user_id", user_res.user.id).execute()
        if not chat_res.data: return jsonify([])
        
        res = _sb.table("messages").select("role, content, created_at").eq("chat_id", chat_id).order("created_at").execute()
        return jsonify(res.data)
    except Exception as e:
        print("GET /chats/<id> error:", e)
        return jsonify([])

@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    """Delete a chat and all its messages (ownership verified)."""
    if not _sb: return jsonify({"ok": False}), 503
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    token = auth_header.split(" ")[1]
    try:
        user_res = _sb.auth.get_user(token)
        if not user_res or not user_res.user:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        # Verify ownership before deleting
        chat_res = _sb.table("chats").select("id").eq("id", chat_id).eq("user_id", user_res.user.id).execute()
        if not chat_res.data:
            return jsonify({"ok": False, "error": "Not found"}), 404
        _sb.table("messages").delete().eq("chat_id", chat_id).execute()
        _sb.table("chats").delete().eq("id", chat_id).execute()
        return jsonify({"ok": True})
    except Exception as e:
        print("DELETE /chats/<id> error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/announcement", methods=["GET"])
def get_announcement():
    """Fetch the latest active announcement/popup for the web app."""
    if _sb:
        try:
            res = _sb.table("announcements").select("*").eq("active", True).order("created_at", desc=True).limit(1).execute()
            if res.data:
                return jsonify(res.data[0])
        except Exception as e:
            print("Announcement fetch error:", e)
    return jsonify({"active": False})


@app.route("/admin/announcement", methods=["POST"])
def push_announcement():
    """Push a global popup announcement to all users.
    Protected by ADMIN_SECRET env var.
    
    Body (JSON):
        secret   : str  — must match ADMIN_SECRET env var
        title    : str  — popup headline (required)
        message  : str  — body text (required)
        tag      : str  — badge label, e.g. "New Feature" (optional, default "Update")
        image_url: str  — optional banner image URL
        active   : bool — set False to deactivate a previous announcement (default True)
    """
    if not _sb:
        return jsonify({"ok": False, "error": "Supabase not configured"}), 503

    data = request.get_json(silent=True) or {}
    secret = (data.get("secret") or "").strip()
    admin_secret = os.environ.get("ADMIN_SECRET", "")

    if not admin_secret or secret != admin_secret:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    title    = (data.get("title") or "").strip()
    message  = (data.get("message") or "").strip()
    tag      = (data.get("tag") or "Update").strip()
    image_url = (data.get("image_url") or "").strip() or None
    active   = bool(data.get("active", True))

    if not title or not message:
        return jsonify({"ok": False, "error": "title and message are required"}), 400

    try:
        # Deactivate all previous announcements first
        _sb.table("announcements").update({"active": False}).eq("active", True).execute()
        # Insert the new one
        row = {
            "title":     title,
            "message":   message,
            "tag":       tag,
            "image_url": image_url,
            "active":    active,
        }
        res = _sb.table("announcements").insert(row).execute()
        return jsonify({"ok": True, "announcement": res.data[0] if res.data else row})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/admin/announcement/clear", methods=["POST"])
def clear_announcements():
    """Deactivate all active announcements (removes the popup for all users)."""
    if not _sb:
        return jsonify({"ok": False, "error": "Supabase not configured"}), 503

    data = request.get_json(silent=True) or {}
    secret = (data.get("secret") or "").strip()
    admin_secret = os.environ.get("ADMIN_SECRET", "")

    if not admin_secret or secret != admin_secret:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    try:
        _sb.table("announcements").update({"active": False}).eq("active", True).execute()
        return jsonify({"ok": True, "message": "All announcements deactivated."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_cf_session():
    """Reset conversation history for a session in the CF runner."""
    data = request.get_json(silent=True) or {}
    sid = (data.get("session_id") or "").strip()
    if sid:
        _cf.reset_session(sid)
    return jsonify({"ok": True})


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


def _send_whatsapp(to_number: str, text: str, phone_number_id: str = ""):
    _msg.send_text(to_number, text, phone_number_id=phone_number_id)


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
    token = os.environ.get("META_WA_ACCESS_TOKEN", "")
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


def _process_whatsapp_async(from_number: str, body: str, phone_number_id: str = ""):
    """Run the agent in a background thread and push the reply when done."""
    session_id = f"wa-{from_number}"
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
        _send_whatsapp(from_number, reply, phone_number_id=phone_number_id)
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

        # Extract which phone number ID received this message (the bot's number)
        # This ensures we reply FROM the same number the user messaged, not the test number
        phone_number_id = value.get("metadata", {}).get("phone_number_id", "")

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

        print(f"💬 WA from {from_number} [{msg_type}] via phone_id={phone_number_id}: {body[:80]}")
        if body:
            threading.Thread(
                target=_process_whatsapp_async,
                args=(from_number, body, phone_number_id),
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
