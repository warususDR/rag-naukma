"""
NaUKMA RAG Web Interface
Flask-based chat API for LightRAG pipeline.
"""

import atexit
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
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
    llm_model_name="mamaylum-4b",
    chroma_directory="../chroma_db",
    chroma_collection="naukma_documents_no_chunks",
)
logger.info("LightRAG model ready.")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    mode = data.get("mode", "hybrid")

    if not question:
        return jsonify({"error": "Порожнє питання"}), 400

    if mode not in ("naive", "local", "global", "hybrid"):
        return jsonify({"error": "Невідомий режим"}), 400

    result = rag.query(question, mode=mode)

    return jsonify({
        "answer": result["response"],
        "mode": result["mode"],
    })


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify({"history": rag.get_history()})


@app.route("/api/clear", methods=["POST"])
def clear():
    rag.clear_history()
    return jsonify({"status": "ok"})


atexit.register(rag.finalize)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
