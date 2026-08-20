from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber
from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    kind: str


def document_id_for(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return f"{slug[:40]}-{digest}"


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def table_to_summary(table: list[list[object | None]], table_number: int) -> str:
    """Make a table searchable without requiring a second generative model."""
    cleaned = [[normalise(cell) for cell in row] for row in table if any(row)]
    if not cleaned:
        return ""
    header = cleaned[0]
    rows = cleaned[1:]
    labels = [cell or f"Column {i + 1}" for i, cell in enumerate(header)]
    rendered_rows: list[str] = []
    for row in rows[:40]:
        pairs = [f"{labels[i]}: {value}" for i, value in enumerate(row) if value]
        if pairs:
            rendered_rows.append("; ".join(pairs))
    body = " | ".join(rendered_rows)
    return f"Table {table_number}. Columns: {', '.join(labels)}. Entries: {body}"


def split_text(text: str, max_chars: int = 1000, overlap: int = 160) -> Iterable[str]:
    text = normalise(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:] if current else ""
        current = f"{tail} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def extract_chunks(pdf_path: Path) -> list[Chunk]:
    """Extract page text and table summaries, retaining original page numbers."""
    chunks: list[Chunk] = []
    reader = PdfReader(str(pdf_path))
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(reader.pages):
            page_number = index + 1
            text = page.extract_text() or ""
            for part in split_text(text):
                chunks.append(Chunk(part, page_number, "text"))
            # pdfplumber may be shorter than pypdf for malformed PDFs, so guard it.
            if index < len(pdf.pages):
                for table_number, table in enumerate(pdf.pages[index].extract_tables(), start=1):
                    summary = table_to_summary(table, table_number)
                    if summary:
                        chunks.append(Chunk(summary, page_number, "table"))
    return chunks
