from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PDF_CHAT_DATA_DIR", PROJECT_ROOT / "data"))
UPLOAD_DIR = Path(os.getenv("PDF_CHAT_UPLOAD_DIR", PROJECT_ROOT / "uploads"))
CHROMA_DIR = Path(os.getenv("PDF_CHAT_CHROMA_DIR", DATA_DIR / "chroma"))
EMBEDDING_MODEL = os.getenv(
    "PDF_CHAT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GENERATION_MODEL = os.getenv("PDF_CHAT_GENERATION_MODEL", "google/flan-t5-small")
MAX_CONTEXT_CHARS = int(os.getenv("PDF_CHAT_MAX_CONTEXT_CHARS", "3600"))
MAX_RETRIEVAL_DISTANCE = float(os.getenv("PDF_CHAT_MAX_RETRIEVAL_DISTANCE", "0.62"))
RERANKER_MODEL = os.getenv(
    "PDF_CHAT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def ensure_storage() -> None:
    for directory in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
