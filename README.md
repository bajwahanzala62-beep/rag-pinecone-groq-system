# Intermediate RAG System — Pinecone + Groq + Streamlit

A Retrieval-Augmented Generation system that answers questions **strictly**
from uploaded PDF content, with page-level source attribution and
similarity scores, designed to prevent hallucination.

## Architecture

```
PDF Upload (Streamlit)
   -> pdf_processor.py   : extract + clean text per page
   -> chunker.py         : sentence-aware overlapping chunks + metadata
   -> embeddings.py      : all-MiniLM-L6-v2 (384-dim) sentence-transformer
   -> vector_store.py    : Pinecone upsert (1 namespace per document)
   -> retriever.py       : embed query -> Pinecone query -> similarity gate
   -> generator.py       : Groq LLM, context-only system prompt, fallback message
   -> app.py             : Streamlit UI wiring + source attribution display
```

## Why these design choices

| Decision | Reason |
|---|---|
| **Namespace per document** | Free multi-document support without separate indexes; queries can be scoped to one, several, or all documents. |
| **all-MiniLM-L6-v2 embeddings** | Runs on CPU, no API cost, 384 dims keeps Pinecone storage/query cheap, strong quality-to-size ratio. |
| **Sentence-aware chunking** | Avoids cutting sentences mid-way (pure character chunking) while keeping chunk sizes controllable (pure sentence chunking can be uneven). Overlap preserves context across chunk boundaries. |
| **Groq (Llama 3.1 8B Instant)** | Fast, low-cost/free-tier LLM API; temperature=0 for deterministic, less "creative" answers. |
| **Three-layer hallucination prevention** | (1) similarity threshold drops weak matches before the LLM ever sees them, (2) if zero chunks survive, the LLM is never called — hardcoded fallback returned instead, (3) system prompt hard-constrains the model to context-only answers. |

## Setup

1. **Clone / open the project**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get your API keys:**
   - Pinecone: https://app.pinecone.io → API Keys (free tier is enough)
   - Groq: https://console.groq.com → API Keys (free tier available)

3. **Configure environment variables.** Copy `.env.example` to `.env` and fill in:
   ```
   PINECONE_API_KEY=...
   GROQ_API_KEY=...
   ```
   (Index/region/model defaults already work out of the box.)

4. **Run locally:**
   ```bash
   streamlit run app.py
   ```

5. Upload a PDF, wait for "Indexed", then ask a question.

## Deploying on Streamlit Community Cloud

1. Push this project to a **public (or private-with-access) GitHub repo**.
2. Go to https://share.streamlit.io → "New app" → point it at `app.py`.
3. In **App settings → Secrets**, paste (TOML format):
   ```toml
   PINECONE_API_KEY = "your_key"
   PINECONE_INDEX_NAME = "rag-intermediate-index"
   PINECONE_CLOUD = "aws"
   PINECONE_REGION = "us-east-1"
   GROQ_API_KEY = "your_key"
   GROQ_MODEL = "llama-3.1-8b-instant"
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   ```
   `config.py` reads from `os.getenv`, and Streamlit Cloud automatically
   injects `st.secrets` entries as environment variables, so no code
   changes are needed between local `.env` and cloud `secrets.toml`.
4. Deploy. First load will be slower (downloading the embedding model
   ~80MB) — this is normal and only happens once per container.

## Implemented "Intermediate Enhancements" (5 of 7 required minimum 3)

- ✅ Multi-document support (namespace-per-doc + multiselect in sidebar)
- ✅ Query history (session memory, shown in sidebar)
- ✅ Adjustable chunk size / overlap from the UI
- ✅ Adjustable top-K retrieval from the UI
- ✅ Metadata filtering by page number
- ✅ Confidence scoring display (similarity score per source excerpt)
- ✅ Logging user queries (`query_log.csv`, written by `src/logger.py`)

## Error handling covered

- Invalid/corrupt/encrypted/scanned PDFs → `InvalidPDFError`, shown as a UI error
- File size > 20MB → rejected before parsing
- Empty query string → blocked before hitting Pinecone/LLM
- Pinecone connection/auth/query failures → `PineconeConnectionError`, shown as a UI error
- LLM call failures → caught, surfaced as a UI error (retrieved sources are not lost)

## Report checklist (for your 3–5 page technical report)

1. **Introduction & objective** — restate the assignment goal in your own words.
2. **Architecture diagram** — use the pipeline diagram above as a starting point;
   redraw it in draw.io / Excalidraw / PowerPoint for the deliverable.
3. **Design decisions** — use the table above as your source material, elaborate on trade-offs.
4. **Embedding model** — all-MiniLM-L6-v2: dimension, why local vs. API-based, cost implications.
5. **Pinecone configuration** — serverless spec, cosine metric, namespace-per-document strategy, metadata schema (`text`, `page_number`, `doc_name`, `chunk_index`).
6. **Hallucination prevention** — describe the 3-layer gate (this is 15% of your grade — be explicit).
7. **Challenges faced** — e.g. scanned PDFs with no extractable text, chunk boundary context loss, tuning similarity threshold.
8. **Performance analysis** — test with 2-3 real PDFs; report indexing time, query latency, and a few example Q&A pairs including a deliberate "not in document" test case.
9. **Conclusion & future work** — e.g. OCR support, reranking, hybrid search (BM25 + vector).

## Suggested test cases for your demo video

1. A question clearly answered on a specific page → check the page number and excerpt match.
2. A question about something **not** in the PDF → verify you get the fallback message.
3. Upload two PDFs, restrict the multiselect to one → verify no cross-contamination.
4. Lower the similarity threshold to 0 → show weaker matches getting through (demonstrates the gate working).
5. Adjust chunk size live and re-index → show how retrieval quality/granularity changes.
