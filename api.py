"""HTTP API for Resynth AI.
- POST /chat       → JSON {session_id, message} → {response}
- POST /whatsapp   → Twilio WhatsApp webhook (form-encoded). Replies asynchronously.
- GET  /health
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
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
_SB_URL = os.environ.get("SUPABASE_URL", "")
_SB_KEY = os.environ.get("SUPABASE_KEY", "")

try:
    from supabase import create_client as _sb_create
    _sb = _sb_create(_SB_URL, _SB_KEY) if _SB_URL and _SB_KEY else None
except Exception as e:
    print("Supabase init error:", e)
    _sb = None



def _get_user_id(token: str):
    """Bypass supabase SDK to get user ID from token."""
    if not _SB_URL or not _SB_KEY: return None
    import requests
    headers = {"apikey": _SB_KEY, "Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{_SB_URL}/auth/v1/user", headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json().get("id")
    except Exception:
        pass
    return None

app = Flask(__name__)
sock = Sock(app)
print("🚀 Initialising Resynth model chain…")
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

@app.route('/api/visualize', methods=['POST'])
def visualize_research():
    """Generate a Mermaid mindmap of the research content."""
    data = request.json or {}
    context = data.get('context', '')
    query = data.get('query', '')
    
    prompt = f"""Based on this research about '{query}', create a Mermaid mindmap that visualizes the core concepts and their connections.
    Output ONLY the Mermaid code starting with 'mindmap'.
    Keep it concise with 3-4 main branches.
    
    Context: {context[:3000]}
    """
    try:
        from research_runner import _get_client, FAST_MODEL
        client, _, _ = _get_client()
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        content = resp.choices[0].message.content.strip()
        # Clean up any markdown blocks
        if "```mermaid" in content: content = content.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        return jsonify({"mermaid": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/bibtex', methods=['POST'])
def project_bibtex():
    """Export all sources from a project as a combined BibTeX."""
    data = request.json or {}
    project_id = data.get('project_id')
    if not project_id or not _SB_URL or not _SB_KEY: return jsonify({"error": "missing"}), 400
    
    try:
        # 1. Get all chat IDs in project
        r1 = requests.get(f"{_SB_URL}/rest/v1/chats?project_id=eq.{project_id}", headers={"apikey":_SB_KEY,"Authorization":f"Bearer {_SB_KEY}"})
        chat_ids = [c['id'] for c in (r1.json() or [])]
        if not chat_ids: return jsonify({"bibtex": ""})
        
        # 2. Get all assistant messages with sources
        all_bib = []
        for cid in chat_ids:
            r2 = requests.get(f"{_SB_URL}/rest/v1/messages?chat_id=eq.{cid}&role=eq.assistant", headers={"apikey":_SB_KEY,"Authorization":f"Bearer {_SB_KEY}"})
            for msg in (r2.json() or []):
                content = msg.get('content', '')
                pass
        
        return jsonify({"bibtex": "% Project Bibliography\n@article{...}"})
    except: return jsonify({"error": "failed"}), 500

@app.route('/api/share', methods=['POST'])
def share_report():
    """Create a public share link for a research report."""
    data = request.json or {}
    content = data.get('content')
    sources = data.get('sources')
    if not _SB_URL or not _SB_KEY: return jsonify({"error": "No SB"}), 500
    try:
        r = requests.post(f"{_SB_URL}/rest/v1/shared_reports", headers={"apikey":_SB_KEY,"Authorization":f"Bearer {_SB_KEY}","Prefer":"return=representation"}, json={"content":content,"sources":sources})
        if r.status_code in (200,201) and r.json():
            return jsonify({"url": f"{request.host_url}share/{r.json()[0]['id']}"})
    except: pass
    return jsonify({"error":"failed"}), 500

@app.route('/share/<report_id>')
def view_shared(report_id):
    """Public view for shared research."""
    r = requests.get(f"{_SB_URL}/rest/v1/shared_reports?id=eq.{report_id}", headers={"apikey":_SB_KEY,"Authorization":f"Bearer {_SB_KEY}"})
    if r.status_code == 200 and r.json():
        return render_template('shared.html', report=r.json()[0])
    return "Not found", 404


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


# ── In-memory session document store ─────────────────────────────────────────
# Keeps the last uploaded document text per session so follow-up
# questions work without re-uploading the file.
_session_docs: dict = {}


@app.route("/session/clear-doc", methods=["POST"])
def clear_session_doc():
    """Clear the stored document for a session."""
    data = request.get_json(silent=True) or {}
    sid = str(data.get("session_id", ""))
    _session_docs.pop(sid, None)
    return jsonify({"ok": True})


@app.route("/chat/stream", methods=["GET", "POST"])
def chat_stream():
    """Server-Sent Events stream. Supports both GET (legacy) and POST (new CF runner)."""
    # ── New POST path: Cloudflare Workers AI ─────────────────────────────────
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id", "default"))
        message    = (data.get("message") or "").strip()
        model      = (data.get("model")   or "Resynth 1.5").strip()
        chat_id    = str(data.get("chat_id", ""))
        image_data = data.get("image_data", None)

        deep_dive  = bool(data.get("deep_dive", False))
        lit_review = bool(data.get("lit_review", False))

        # Handle document attachments (PDF/TXT/DOCX) — extract text + store in session
        if image_data and image_data.startswith("data:"):
            mime_type = image_data.split(";")[0].split(":")[1]
            DOC_MIMES = [
                "application/pdf",
                "text/plain",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]
            if mime_type in DOC_MIMES:
                import base64
                from rag_pipeline import extract_text_from_pdf, extract_text_from_txt
                try:
                    file_bytes = base64.b64decode(image_data.split(",")[1])
                    doc_text = extract_text_from_pdf(file_bytes) if mime_type == "application/pdf" else extract_text_from_txt(file_bytes)
                    if doc_text and doc_text.strip():
                        _session_docs[session_id] = doc_text.strip()
                except Exception as e:
                    print("Doc parse error:", e)
                image_data = None  # Route to text engine

        auth_header = request.headers.get("Authorization", "")
        user_id = None
        current_token = None
        if _SB_URL:
            if auth_header.startswith("Bearer "):
                current_token = auth_header.split(" ")[1]
                user_id = _get_user_id(current_token)

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
            # Retrieve any stored session document to include as context
            session_doc = _session_docs.get(session_id)
            # Route ALL queries (text or image) to research engine for visual search
            yield from _research.stream_research(
                session_id=session_id, query=message,
                jwt_token=current_token, user_id=user_id, chat_id=chat_id,
                deep_dive=deep_dive, sb=_sb, session_doc=session_doc,
                lit_review=lit_review, image_data=image_data,
                citation_style=data.get('citation_style', 'inline'),
                strategy=data.get('strategy', 'balanced'),
                debate=data.get('debate', False)
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
    return render_template("pricing.html")


POLAR_ACCESS_TOKEN = os.environ.get("POLAR_ACCESS_TOKEN", "")
POLAR_PRO_PRODUCT_ID = os.environ.get("POLAR_PRO_PRODUCT_ID", "")
POLAR_WEBHOOK_SECRET = os.environ.get("POLAR_WEBHOOK_SECRET", "")
POLAR_SERVER = os.environ.get("POLAR_SERVER", "sandbox")
POLAR_BASE = "https://sandbox-api.polar.sh" if POLAR_SERVER == "sandbox" else "https://api.polar.sh"

@app.route("/polar/checkout", methods=["POST"])
def polar_create_checkout():
    if not POLAR_ACCESS_TOKEN or not POLAR_PRO_PRODUCT_ID:
        return jsonify({"error": "Polar is not configured"}), 503

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    payload = {
        "products": [POLAR_PRO_PRODUCT_ID],
        "metadata": {"user_id": user_id, "app": "resynth-web"},
    }
    headers = {
        "Authorization": f"Bearer {POLAR_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    import requests
    try:
        r = requests.post(f"{POLAR_BASE}/v1/checkouts/", json=payload, headers=headers, timeout=15)
        if r.status_code >= 400:
            print(f"Polar checkout error {r.status_code}: {r.text}")
            return jsonify({"error": f"Polar error: {r.text}"}), 502
        
        data = r.json()
        checkout_id = data.get("id")
        url = data.get("url")
        if not url or not checkout_id:
            return jsonify({"error": "Invalid Polar response"}), 502
            
        return jsonify({"url": url, "checkout_id": checkout_id})
    except Exception as e:
        print("polar_create_checkout failed:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/polar/webhook", methods=["POST"])
def polar_webhook():
    event = request.get_json(silent=True) or {}
    event_type = event.get("type")
    data = event.get("data", {}) or {}
    
    metadata = data.get("metadata") or {}
    user_id = metadata.get("user_id")
    checkout_id = data.get("id") or data.get("checkout_id")
    
    print(f"📥 Polar webhook received: {event_type} user={user_id} ckid={checkout_id} status={data.get('status')}")

    PAID_EVENTS = {
        "checkout.updated",
        "checkout.confirmed", 
        "order.created",
        "order.paid",
        "subscription.created",
        "subscription.active",
    }

    paid = False
    if event_type in PAID_EVENTS:
        status = (data.get("status") or "").lower()
        if event_type.startswith("order.") or event_type.startswith("subscription."):
            paid = True
        elif status in {"succeeded", "completed", "confirmed", "success"}:
            paid = True

    if paid and user_id and _sb:
        try:
            _sb.table("profiles").upsert({"id": user_id, "plan": "pro", "paddle_subscription_id": checkout_id}).execute()
            print(f"✅ Pro granted to user {user_id}")
        except Exception as e:
            print("DB error on webhook:", e)

    return jsonify({"ok": True})

@app.route("/polar/customer-portal", methods=["POST"])
def polar_customer_portal():
    if not POLAR_ACCESS_TOKEN:
        return jsonify({"error": "Polar not configured"}), 503
    portal_url = f"https://{'sandbox.' if POLAR_SERVER == 'sandbox' else ''}polar.sh/purchases/subscriptions"
    return jsonify({"url": portal_url})

@app.route("/chats", methods=["GET"])
def get_chats():
    try:
        try:
            url = _SB_URL
            key = _SB_KEY
        except NameError:
            url = os.environ.get("SUPABASE_URL", "")
            key = os.environ.get("SUPABASE_KEY", "")
        
        if not url or not key: return jsonify([])
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "): return jsonify([])
        token = auth_header.split(" ")[1]
        
        user_id = _get_user_id(token)
        if not user_id: return jsonify([])
        
        import requests
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        url = f"{url}/rest/v1/chats?user_id=eq.{user_id}&order=created_at.desc"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return jsonify(r.json())
        else:
            return jsonify({"error": "REST error", "details": r.text}), 500
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/chats/<chat_id>", methods=["GET"])
def get_chat_messages(chat_id):
    if not _SB_URL or not _SB_KEY: return jsonify([])
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "): return jsonify([])
    token = auth_header.split(" ")[1]
    try:
        user_id = _get_user_id(token)
        if not user_id: return jsonify([])
        
        # Direct REST to bypass RLS SDK issues
        import requests
        headers = {
            "apikey": _SB_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        # First verify ownership
        check_url = f"{_SB_URL}/rest/v1/chats?id=eq.{chat_id}&user_id=eq.{user_id}&select=id"
        r_check = requests.get(check_url, headers=headers)
        if r_check.status_code != 200 or not r_check.json():
            return jsonify([])
            
        msg_url = f"{_SB_URL}/rest/v1/messages?chat_id=eq.{chat_id}&order=created_at.asc&select=role,content,created_at"
        r_msg = requests.get(msg_url, headers=headers)
        if r_msg.status_code == 200:
            return jsonify(r_msg.json())
        return jsonify([])
    except Exception as e:
        print("GET /chats/<id> error:", e)
        return jsonify([])

@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if not _SB_URL or not _SB_KEY: return jsonify({"ok": False}), 503
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    token = auth_header.split(" ")[1]
    try:
        user_id = _get_user_id(token)
        if not user_id:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
            
        import requests
        headers = {
            "apikey": _SB_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        # Verify ownership
        check_url = f"{_SB_URL}/rest/v1/chats?id=eq.{chat_id}&user_id=eq.{user_id}&select=id"
        r_check = requests.get(check_url, headers=headers)
        if r_check.status_code != 200 or not r_check.json():
            return jsonify({"ok": False, "error": "Not found"}), 404
            
        # Delete messages then chat
        requests.delete(f"{_SB_URL}/rest/v1/messages?chat_id=eq.{chat_id}", headers=headers)
        requests.delete(f"{_SB_URL}/rest/v1/chats?id=eq.{chat_id}", headers=headers)
        return jsonify({"ok": True})
    except Exception as e:
        print("DELETE /chats/<id> error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/chats/<chat_id>", methods=["PATCH"])
@app.route("/chats/<chat_id>/title", methods=["PATCH"])
def rename_chat(chat_id):
    if not _SB_URL or not _SB_KEY: return jsonify({"ok": False}), 503
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    token = auth_header.split(" ")[1]
    data = request.get_json(silent=True) or {}
    new_title = data.get("title")
    if not new_title:
        return jsonify({"ok": False, "error": "Missing title"}), 400
    try:
        user_id = _get_user_id(token)
        if not user_id:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
            
        import requests
        headers = {
            "apikey": _SB_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        check_url = f"{_SB_URL}/rest/v1/chats?id=eq.{chat_id}&user_id=eq.{user_id}&select=id"
        r_check = requests.get(check_url, headers=headers)
        if r_check.status_code != 200 or not r_check.json():
            return jsonify({"ok": False, "error": "Not found"}), 404
            
        patch_url = f"{_SB_URL}/rest/v1/chats?id=eq.{chat_id}"
        requests.patch(patch_url, headers=headers, json={"title": new_title})
        return jsonify({"ok": True})
    except Exception as e:
        print("PATCH /chats/<id> error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/test_auth", methods=["GET"])
def test_auth():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "No Bearer token"})
    token = auth_header.split(" ")[1]
    
    diag = {"token_prefix": token[:10], "sb_exists": bool(_sb)}
    if not _sb: return jsonify(diag)
    
    try:
        user_res = _sb.auth.get_user(token)
        diag["user_id"] = user_res.user.id if user_res and user_res.user else None
    except Exception as e:
        diag["get_user_error"] = str(e)
        return jsonify(diag)
        
    try:
        auth_client = _get_auth_client(token)
        diag["auth_client_created"] = True
        
        # Test 1: Fetch chats
        res = auth_client.table("chats").select("id, title").limit(5).execute()
        diag["chats_fetched"] = len(res.data)
        diag["chats_sample"] = res.data
        
        # Test 2: Try to insert a dummy chat
        ins = auth_client.table("chats").insert({"user_id": diag["user_id"], "title": "Diag Test Chat"}).execute()
        diag["chat_inserted"] = bool(ins.data)
        if ins.data:
            auth_client.table("chats").delete().eq("id", ins.data[0]["id"]).execute()
            
    except Exception as e:
        diag["query_error"] = str(e)
        
    return jsonify(diag)

@app.route("/api/generate-title", methods=["POST"])
def generate_title():
    data = request.get_json(silent=True) or {}
    raw = (data.get("message") or "").strip()
    fallback = data.get("fallback") or ""
    def _fallback_title(text):
        cleaned = " ".join((text or "").split())
        return cleaned[:48].rstrip(" ,.;:-—") or "New chat"
        
    if not raw:
        return jsonify({"title": _fallback_title(fallback)})
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"title": _fallback_title(raw)})
        
    try:
        import requests
        system = (
            "You generate short, descriptive titles for chat conversations. "
            "Given the user's first message, output a 2 to 4 word Title Case label "
            "summarizing the topic or intent. Do not use quotes, periods, or emojis. "
            "Examples:\n"
            "- 'Hi' -> Greetings\n"
            "- 'how does CRISPR work?' -> CRISPR Gene Editing\n"
            "- 'help me write a cover letter for a software job' -> Software Cover Letter\n"
            "- 'What is the capital of France?' -> Capital of France\n"
            "Output only the title text, nothing else."
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": raw[:400].replace("\n", " ")}
            ],
            "temperature": 0.5,
            "max_tokens": 15
        }
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=5
        )
        res.raise_for_status()
        title = res.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'").rstrip(".")
        if not title or len(title) > 60:
            title = _fallback_title(raw)
        return jsonify({"title": title})
    except Exception as e:
        print("generate_title error:", e)
        return jsonify({"title": _fallback_title(raw)})


@app.route("/chats/<chat_id>/title", methods=["PATCH"])
def update_chat_title(chat_id):
    """Update a chat's auto-generated title."""
    if not _sb: return jsonify({"ok": False}), 503
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False}), 401
    token = auth_header.split(" ")[1]
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()[:80]
    if not title:
        return jsonify({"ok": False, "error": "title required"}), 400
    try:
        user_res = _sb.auth.get_user(token)
        if not user_res or not user_res.user:
            return jsonify({"ok": False}), 401
        _sb.table("chats").update({"title": title}).eq("id", chat_id).eq("user_id", user_res.user.id).execute()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/account", methods=["GET"])
def account_info():
    """Return profile + today's usage for the authenticated user."""
    if not _sb: return jsonify({"error": "no db"}), 503
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "unauthorized"}), 401
    token = auth_header.split(" ")[1]
    try:
        from limits import get_usage_today, get_user_plan
        user_res = _sb.auth.get_user(token)
        if not user_res or not user_res.user:
            return jsonify({"error": "unauthorized"}), 401
        user = user_res.user
        plan = get_user_plan(user.id, _sb)
        usage = get_usage_today(user.id, user.id, _sb)
        return jsonify({
            "email": user.email,
            "plan": plan,
            "usage": usage,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/auth/welcome", methods=["POST"])
def send_welcome_email():
    """Send a Mailjet welcome email after signup. Called from frontend."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    name  = (data.get("name") or "there").strip()
    if not email:
        return jsonify({"ok": False}), 400

    mj_key    = os.environ.get("MAILJET_API_KEY", "")
    mj_secret = os.environ.get("MAILJET_API_SECRET", "")
    if not mj_key or not mj_secret:
        return jsonify({"ok": False, "error": "Mailjet not configured"}), 503

    try:
        import requests as _req
        payload = {
            "Messages": [{
                "From": {"Email": "hello@resynth.com", "Name": "Resynth AI"},
                "To":   [{"Email": email, "Name": name}],
                "Subject": "Welcome to Resynth AI 🔮",
                "HTMLPart": f"""
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:auto;background:#0a0a0f;color:#fff;border-radius:16px;padding:40px;">
                  <div style="font-size:28px;font-weight:800;margin-bottom:8px;">Welcome to Resynth, {name}. 🔮</div>
                  <div style="color:#8888aa;font-size:15px;line-height:1.7;margin-bottom:28px;">
                    You now have access to a research AI that searches the live web, cites every source, and thinks deeply on any topic.<br><br>
                    You start with <strong style="color:#fff">20 free messages per day</strong>. Upgrade to Pro for unlimited access.
                  </div>
                  <a href="https://resynth-ai-production.up.railway.app" style="display:inline-block;background:#fff;color:#0a0a0f;font-weight:700;padding:14px 28px;border-radius:10px;text-decoration:none;font-size:15px;">Open Resynth →</a>
                  <div style="margin-top:40px;padding-top:20px;border-top:1px solid #1f1f2e;color:#555566;font-size:12px;">
                    © 2026 Resynth Inc. · <a href="https://resynth-ai-production.up.railway.app/pricing" style="color:#555566;">Upgrade to Pro</a>
                  </div>
                </div>
                """
            }]
        }
        resp = _req.post(
            "https://api.mailjet.com/v3.1/send",
            auth=(mj_key, mj_secret),
            json=payload,
            timeout=8
        )
        return jsonify({"ok": resp.status_code == 200})
    except Exception as e:
        print("Mailjet error:", e)
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
