import atexit
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from functools import wraps

import psycopg2
import psycopg2.extras
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from lightrag_model import LightRAGModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google OAuth config
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]

def verify_token(token: str) -> dict:
    """Validate Google ID token and return claims."""
    idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
    return idinfo

# ---------------------------------------------------------------------------
# Postgres session store (same DB, separate table)
# ---------------------------------------------------------------------------
DB_DSN = (
    f"host={os.environ['POSTGRES_HOST']} "
    f"port={os.environ['POSTGRES_PORT']} "
    f"dbname={os.environ['POSTGRES_DATABASE']} "
    f"user={os.environ['POSTGRES_USER']} "
    f"password={os.environ['POSTGRES_PASSWORD']}"
)

def _get_conn():
    return psycopg2.connect(DB_DSN, cursor_factory=psycopg2.extras.RealDictCursor)

def init_db():
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          UUID PRIMARY KEY,
                user_email  TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'Нова розмова',
                mode        TEXT NOT NULL DEFAULT 'hybrid',
                messages    JSONB NOT NULL DEFAULT '[]',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    logger.info("chat_sessions table ready")

# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            g.claims = verify_token(auth.split(" ", 1)[1])
            g.user_email = g.claims.get("email", "").lower()
        except (ValueError, Exception) as e:
            logger.warning(f"Auth failed: {e}")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logger.info("Initializing LightRAG model...")
rag = LightRAGModel(
    embedding_model="bge-m3",
    llm_model_name="mamaylum-4b",
    chroma_directory="../chroma_db",
    chroma_collection="naukma_documents_no_chunks",
)
logger.info("LightRAG model ready.")
init_db()

# ---------------------------------------------------------------------------
# Sessions endpoints
# ---------------------------------------------------------------------------
@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE user_email = %s ORDER BY updated_at DESC",
            (g.user_email,),
        )
        rows = cur.fetchall()
    return jsonify([_row_to_session(r) for r in rows])


@app.route("/api/sessions", methods=["POST"])
@require_auth
def create_session():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "hybrid")
    if mode not in ("naive", "local", "global", "hybrid"):
        return jsonify({"error": "Invalid mode"}), 400
    session_id = str(uuid.uuid4())
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (id, user_email, mode) VALUES (%s, %s, %s) RETURNING *",
            (session_id, g.user_email, mode),
        )
        row = cur.fetchone()
    return jsonify(_row_to_session(row)), 201


@app.route("/api/sessions/<session_id>", methods=["GET"])
@require_auth
def get_session(session_id):
    row = _fetch_session(session_id)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(_row_to_session(row))


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
@require_auth
def patch_session(session_id):
    row = _fetch_session(session_id)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    allowed = {k: v for k, v in data.items() if k in ("title", "mode")}
    if not allowed:
        return jsonify({"error": "Nothing to update"}), 400
    sets = ", ".join(f"{k} = %s" for k in allowed)
    vals = list(allowed.values()) + [datetime.now(timezone.utc), session_id, g.user_email]
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE chat_sessions SET {sets}, updated_at = %s WHERE id = %s AND user_email = %s", vals)
    return jsonify({"status": "ok"})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id):
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chat_sessions WHERE id = %s AND user_email = %s",
            (session_id, g.user_email),
        )
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    mode = data.get("mode", "hybrid")
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Порожнє питання"}), 400
    if mode not in ("naive", "local", "global", "hybrid"):
        return jsonify({"error": "Невідомий режим"}), 400

    row = _fetch_session(session_id) if session_id else None
    history = list(row["messages"]) if row else []

    result = rag.query(question, mode=mode, history=history)

    # Persist updated messages + auto-title
    if row is not None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["response"]})
        if len(history) > 20:
            history = history[-20:]
        title = row["title"]
        if title == "Нова розмова":
            title = question[:50] + ("…" if len(question) > 50 else "")
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET messages = %s, title = %s, updated_at = %s WHERE id = %s AND user_email = %s",
                (json.dumps(history, ensure_ascii=False), title, datetime.now(timezone.utc), session_id, g.user_email),
            )

    return jsonify({"answer": result["response"], "mode": result["mode"], "session_id": session_id})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_session(session_id: str):
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE id = %s AND user_email = %s",
            (session_id, g.user_email),
        )
        return cur.fetchone()

def _row_to_session(row) -> dict:
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "mode": row["mode"],
        "messages": row["messages"],
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }

atexit.register(rag.finalize)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

