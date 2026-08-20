from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_RETRIEVAL_DISTANCE,
    RERANKER_MODEL,
    ensure_storage,
)
from app.ingestion import document_id_for, extract_chunks


def preferred_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device=preferred_device())


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    """A second local model that is more precise than embedding similarity alone."""
    return CrossEncoder(RERANKER_MODEL, device=preferred_device())


@lru_cache(maxsize=1)
def generator() -> tuple[Any, Any, str]:
    device = preferred_device()
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model, device


class RAGService:
    def __init__(self) -> None:
        ensure_storage()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks", metadata={"hnsw:space": "cosine"}
        )

    def ingest(self, pdf_path: Path) -> dict[str, Any]:
        chunks = extract_chunks(pdf_path)
        if not chunks:
            raise ValueError("No readable text or tables were found in this PDF.")
        doc_id = document_id_for(pdf_path)
        self.collection.delete(where={"document_id": doc_id})
        texts = [chunk.text for chunk in chunks]
        vectors = embedder().encode(texts, normalize_embeddings=True).tolist()
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadata = [
            {"document_id": doc_id, "filename": pdf_path.name, "page": chunk.page, "kind": chunk.kind}
            for chunk in chunks
        ]
        self.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadata)
        return {"document_id": doc_id, "filename": pdf_path.name, "chunks": len(chunks)}

    def ask(self, question: str, document_id: str | None = None, top_k: int = 5) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("A question is required.")
        query_vector = embedder().encode([question], normalize_embeddings=True).tolist()
        # Retrieve broadly, then let the cross-encoder rank question/passage pairs.
        requested = min(max(top_k, 1), 8)
        kwargs: dict[str, Any] = {
            "query_embeddings": query_vector,
            "n_results": min(max(requested * 3, 12), 30),
        }
        if document_id:
            kwargs["where"] = {"document_id": document_id}
        result = self.collection.query(**kwargs)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        candidates = [
            (text, meta, float(distance))
            for text, meta, distance in zip(documents, metadatas, distances)
            if float(distance) <= MAX_RETRIEVAL_DISTANCE
        ]
        if not candidates:
            return {"answer": "I could not find relevant content in the indexed PDFs.", "sources": []}
        scores = reranker().predict([(question, text) for text, _, _ in candidates])
        ranked = sorted(
            zip(candidates, scores), key=lambda item: float(item[1]), reverse=True
        )[:requested]
        context_parts: list[str] = []
        evidence: list[dict[str, Any]] = []
        used = 0
        for ((text, meta, distance), score) in ranked:
            item = f"[Excerpt {len(context_parts) + 1}, page {meta['page']}] {text}"
            if used + len(item) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(item)
            used += len(item)
            evidence.append({
                "filename": meta["filename"],
                "page": int(meta["page"]),
                "kind": meta["kind"],
                "text": text,
                "similarity": round(1 - distance, 3),
                "rerank_score": round(float(score), 3),
            })
        if not context_parts:
            return {"answer": "I could not fit relevant evidence into the answer context.", "sources": [], "evidence": []}
        context = "\n\n".join(context_parts)
        prompt = (
            "Use ONLY the excerpts below. Do not use general knowledge, guess, infer missing "
            "rules, or invent details. If the excerpts do not explicitly answer the question, "
            "output exactly: NOT_ENOUGH_EVIDENCE. Otherwise give a concise answer and add the "
            "supporting page in brackets, for example [p. 12].\n\n"
            f"EXCERPTS:\n{context}\n\nQUESTION: {question}\nANSWER:"
        )
        answer = self._generate(prompt)
        if not answer or "NOT_ENOUGH_EVIDENCE" in answer.upper():
            answer = "I don’t have enough evidence in the indexed PDF to answer that reliably."
        seen: set[tuple[str, int]] = set()
        sources = []
        for item in evidence:
            meta = item
            key = (meta["filename"], int(meta["page"]))
            if key not in seen:
                seen.add(key)
                sources.append({"filename": key[0], "page": key[1], "kind": meta["kind"]})
        return {"answer": answer, "sources": sources, "evidence": evidence}

    @staticmethod
    def _generate(prompt: str) -> str:
        tokenizer, model, device = generator()
        batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=180,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()
