from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, request

from app.config import UPLOAD_DIR, ensure_storage
from app.rag import RAGService


def create_app() -> Flask:
    ensure_storage()
    app = Flask(__name__)
    service = RAGService()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/ingest")
    def ingest():
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Send a PDF as form field 'file'."}), 400
        if not uploaded.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF uploads are supported."}), 400
        safe_name = Path(uploaded.filename).name
        destination = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}-{safe_name}"
        uploaded.save(destination)
        try:
            return jsonify(service.ingest(destination)), 201
        except Exception as exc:
            destination.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 422

    @app.post("/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(service.ask(
                question=str(payload.get("question", "")),
                document_id=payload.get("document_id"),
                top_k=int(payload.get("top_k", 5)),
            ))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
