import atexit
import json
import logging
import uuid
from datetime import datetime, timezone

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from auth import require_auth
from db import fetch_session, get_conn, init_db, row_to_session
from lightrag_model import LightRAGModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

logger.info("Initializing LightRAG model...")
rag = LightRAGModel(
    embedding_model="bge-m3",
    llm_model_name="mamaylum-12b",
    chroma_directory="../chroma_db",
    chroma_collection="naukma_documents_no_chunks",
)
logger.info("LightRAG model ready.")
init_db()


@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE user_email = %s ORDER BY updated_at DESC",
            (g.user_email,),
        )
        rows = cur.fetchall()
    return jsonify([row_to_session(r) for r in rows])


@app.route("/api/sessions", methods=["POST"])
@require_auth
def create_session():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "hybrid")
    if mode not in ("naive", "local", "global", "hybrid"):
        return jsonify({"error": "Invalid mode"}), 400
    session_id = str(uuid.uuid4())
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (id, user_email, mode) VALUES (%s, %s, %s) RETURNING *",
            (session_id, g.user_email, mode),
        )
        row = cur.fetchone()
    return jsonify(row_to_session(row)), 201


@app.route("/api/sessions/<session_id>", methods=["GET"])
@require_auth
def get_session(session_id):
    row = fetch_session(session_id)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row_to_session(row))


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
@require_auth
def patch_session(session_id):
    row = fetch_session(session_id)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    allowed = {k: v for k, v in data.items() if k in ("title", "mode")}
    if not allowed:
        return jsonify({"error": "Nothing to update"}), 400
    sets = ", ".join(f"{k} = %s" for k in allowed)
    vals = list(allowed.values()) + [datetime.now(timezone.utc), session_id, g.user_email]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE chat_sessions SET {sets}, updated_at = %s WHERE id = %s AND user_email = %s", vals)
    return jsonify({"status": "ok"})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chat_sessions WHERE id = %s AND user_email = %s",
            (session_id, g.user_email),
        )
    return jsonify({"status": "ok"})


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

    row = fetch_session(session_id) if session_id else None
    history = list(row["messages"]) if row else []

    result = rag.query(question, mode=mode, history=history)

    # Persist updated messages
    if row is not None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["response"], "references": result.get("references", [])})
        if len(history) > 20:
            history = history[-20:]
        title = row["title"]
        if title == "Нова розмова":
            title = question[:50] + ("…" if len(question) > 50 else "")
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET messages = %s, title = %s, updated_at = %s WHERE id = %s AND user_email = %s",
                (json.dumps(history, ensure_ascii=False), title, datetime.now(timezone.utc), session_id, g.user_email),
            )

    return jsonify({"answer": result["response"], "references": result.get("references", []), "mode": result["mode"], "session_id": session_id})


atexit.register(rag.finalize)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

