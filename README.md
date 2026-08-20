# Local PDF Rulebook Chatbot

A local ChatGPT-style assistant for asking questions about rulebooks and other PDFs. It extracts prose and tables, turns tables into searchable natural-language summaries, and answers with retrieval-augmented generation (RAG).

## What is included

- PDF ingestion with page-level source citations
- Table extraction and row-aware table summaries via `pdfplumber`
- ChromaDB vector storage and semantic retrieval
- Hugging Face embeddings, cross-encoder reranking, and FLAN-T5 answer generation
- Flask JSON API and a Streamlit chat UI
- Apple Silicon support: MPS is used when PyTorch exposes it, otherwise CPU

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: API
python -m app.api

# Terminal 2: UI
streamlit run streamlit_app.py
```

The first question may download the selected Hugging Face models. Model files are cached locally by the Hugging Face libraries; after that, normal use stays local. To use already-downloaded models only, run with `HF_HUB_OFFLINE=1`.

## Answer grounding

The chatbot deliberately refuses questions when search evidence is weak. It retrieves a broader set of passages, reranks them with a local cross-encoder, discards low-similarity matches, and instructs the answer model to return `NOT_ENOUGH_EVIDENCE` rather than fill gaps from general knowledge. Expand **Retrieved evidence** under an answer to inspect the exact PDF excerpts used.

If the PDF is a scan made of page images rather than selectable text, run OCR before uploading it; normal PDF extraction cannot reliably retrieve text that is only present in images.

Open the Streamlit address shown in the terminal, upload a PDF, wait for indexing, then start chatting.

## API

`POST /ingest` accepts form field `file` (a PDF) and returns the document ID and chunk count.

`POST /chat`

```json
{"question": "What happens after a penalty?", "document_id": "optional-id", "top_k": 5}
```

The response contains `answer` and an array of page citations in `sources`.

## Configuration

Copy `.env.example` to `.env` to override model IDs or storage paths. `google/flan-t5-small` is the memory-friendly default. On a machine with more memory, `google/flan-t5-base` generally gives better answers.

## Project layout

```text
app/
  api.py          Flask endpoints
  config.py       local paths and runtime options
  ingestion.py    PDF/table extraction and chunking
  rag.py          embeddings, Chroma retrieval, FLAN-T5 answers
streamlit_app.py  browser chat UI
tests/            lightweight ingestion tests
```
