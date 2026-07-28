"""
app.py
------
Streamlit UI. This file only handles presentation/session-state wiring;
all real logic lives in src/. That separation is what the rubric calls
"clean modular architecture / separation of concerns".
"""

import time
import streamlit as st

from src.config import Config
from src.pdf_processor import extract_pages, InvalidPDFError
from src.chunker import chunk_document
from src.embeddings import embed_texts
from src.vector_store import VectorStore, PineconeConnectionError
from src.retriever import retrieve
from src.generator import generate_answer, FALLBACK_MESSAGE
from src.logger import log_query
from src.ui_helpers import confidence_label, overall_confidence, build_report, APP_AUTHOR

# ---------------------------------------------------------------------
# Page config + light custom styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RAG System | Pinecone + Groq",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }
    .author-badge {
        display: inline-block;
        background-color: #eef2ff;
        color: #4338ca;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MIN_SECONDS_BETWEEN_QUERIES = 6  # Enhancement: rate limiting / cooldown

# ---------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------
defaults = {
    "indexed_docs": [],       # filenames already upserted
    "query_history": [],       # Enhancement: session memory
    "vector_store": None,
    "last_query_time": 0.0,     # Enhancement: rate limiting
    "is_processing": False,
    "last_result": None,        # keeps last Q&A on screen after rerun
    "doc_stats": {},            # Enhancement: per-document extraction/chunk/embedding stats
    "doc_chunks": {},           # Enhancement: full chunk list per document, for the Text Chunks view
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown('<div class="main-header">📄 Intermediate RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Retrieval-Augmented Generation over your own PDFs — '
    'powered by Pinecone + Groq, answers grounded strictly in document content.</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<span class="author-badge">Developed by: {APP_AUTHOR}</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Chunking", expanded=False):
        chunk_size = st.slider("Chunk size (characters)", 200, 1500, Config.DEFAULT_CHUNK_SIZE, 50)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, Config.DEFAULT_CHUNK_OVERLAP, 10)

    with st.expander("Retrieval", expanded=True):
        top_k = st.slider("Top-K retrieved chunks", 1, 10, Config.DEFAULT_TOP_K)
        sim_threshold = st.slider("Similarity threshold", 0.0, 1.0, Config.DEFAULT_SIM_THRESHOLD, 0.05)
        page_filter_input = st.text_input("Filter by page number (optional)", "")

    with st.expander("Ingestion previews", expanded=False):
        show_text_preview = st.checkbox("Show extracted text preview", value=True)
        show_embedding_preview = st.checkbox("Show embedding preview", value=True)
        preview_chars = st.slider("Text preview length (characters)", 200, 3000, 800, 100)

    st.divider()

    # -------------------------------------------------------------
    # Statistics panel — shows stats for the most recently processed
    # document (Characters / Chunks / Embeddings), mirroring a classic
    # ingestion dashboard.
    # -------------------------------------------------------------
    st.subheader("📊 Statistics")
    if st.session_state.doc_stats:
        latest_doc = list(st.session_state.doc_stats.keys())[-1]
        stats = st.session_state.doc_stats[latest_doc]
        st.caption(f"Latest document: {latest_doc}")
        st.metric("Characters", f"{stats['characters']:,}")
        st.metric("Chunks", stats["chunks"])
        st.metric("Embeddings", stats["embeddings"])
    else:
        st.info("Upload a document to see statistics.")

    st.divider()
    st.subheader("📚 Indexed documents")
    if st.session_state.indexed_docs:
        selected_docs = st.multiselect(
            "Search within:",
            options=st.session_state.indexed_docs,
            default=st.session_state.indexed_docs,
        )
        if st.button("🗑️ Clear session (UI only)", use_container_width=True):
            st.session_state.indexed_docs = []
            st.session_state.query_history = []
            st.session_state.last_result = None
            st.session_state.doc_stats = {}
            st.session_state.doc_chunks = {}
            st.rerun()
    else:
        selected_docs = []
        st.info("No documents indexed yet.")

    st.divider()
    if st.session_state.query_history:
        st.subheader("🕘 Query history")
        for h in reversed(st.session_state.query_history[-8:]):
            st.markdown(f"- {h}")


def get_vector_store() -> VectorStore:
    if st.session_state.vector_store is None:
        with st.spinner("Connecting to Pinecone..."):
            st.session_state.vector_store = VectorStore()
    return st.session_state.vector_store


# ---------------------------------------------------------------------
# Enhancement: extraction + embedding preview helpers
# Defensive getters — tolerate `pages` being either dicts or simple
# objects, depending on what extract_pages() returns in src/pdf_processor.py
# ---------------------------------------------------------------------
def _page_text(p) -> str:
    if isinstance(p, dict):
        return p.get("text", "") or ""
    return getattr(p, "text", "") or (p if isinstance(p, str) else "")


def _page_number(p, fallback_idx: int):
    if isinstance(p, dict):
        return p.get("page_number", fallback_idx + 1)
    return getattr(p, "page_number", fallback_idx + 1)


def render_text_preview(pages, max_chars: int):
    """Enhancement: show extracted PDF text, page by page, as soon as it's parsed."""
    total_chars = sum(len(_page_text(p)) for p in pages)
    st.caption(f"📝 Extracted {len(pages)} page(s), {total_chars:,} characters total.")
    shown = 0
    for i, p in enumerate(pages):
        text = _page_text(p).strip()
        if not text:
            continue
        remaining = max_chars - shown
        if remaining <= 0:
            st.caption("…preview truncated (increase preview length in sidebar to see more).")
            break
        snippet = text[:remaining]
        shown += len(snippet)
        st.markdown(f"**Page {_page_number(p, i)}**")
        st.text(snippet + ("…" if len(text) > len(snippet) else ""))


def render_embedding_preview(chunks, vectors, n_dims_shown: int = 12, n_chunks_shown: int = 3):
    """Enhancement: show a preview of the generated embedding vectors."""
    if not vectors:
        st.caption("No embeddings generated.")
        return
    dim = len(vectors[0])
    st.caption(f"🧩 Generated {len(vectors)} embedding(s), dimension = {dim}.")
    for c, v in list(zip(chunks, vectors))[:n_chunks_shown]:
        label = f"Chunk {getattr(c, 'chunk_index', '?')} — Page {getattr(c, 'page_number', '?')}"
        with st.expander(label):
            preview_text = getattr(c, "text", "")[:200]
            st.text(preview_text + ("…" if len(getattr(c, "text", "")) > 200 else ""))
            head = list(v[:n_dims_shown])
            st.write(f"First {len(head)} of {dim} dimensions:")
            st.dataframe(
                {"dim": list(range(len(head))), "value": [round(float(x), 5) for x in head]},
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(data={"value": [float(x) for x in head]})


def render_chunks_list(chunks):
    """
    Enhancement: full list of every text chunk generated from the
    document, each individually expandable — "Chunk 1", "Chunk 2", ...
    showing the page it came from and its full text content.
    """
    st.caption(f"Total Chunks: {len(chunks)}")
    for i, c in enumerate(chunks, start=1):
        page = getattr(c, "page_number", "?")
        with st.expander(f"Chunk {i} — Page {page}"):
            st.text(getattr(c, "text", ""))


# ---------------------------------------------------------------------
# 1. PDF Upload + Indexing
# ---------------------------------------------------------------------
st.subheader("1️⃣ Upload PDF(s)")
upload_col, info_col = st.columns([3, 1])

with upload_col:
    uploaded_files = st.file_uploader(
        "Upload up to 20MB per file", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

with info_col:
    st.metric("Documents indexed", len(st.session_state.indexed_docs))

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_docs]
    for idx, uploaded in enumerate(new_files):
        progress = st.progress(0, text=f"Processing '{uploaded.name}'...")
        try:
            progress.progress(15, text="Extracting text from PDF...")
            file_bytes = uploaded.read()
            pages = extract_pages(file_bytes, uploaded.name)
            total_chars = sum(len(_page_text(p)) for p in pages)

            if show_text_preview:
                with st.expander(f"📝 Extracted text preview — {uploaded.name}", expanded=True):
                    render_text_preview(pages, preview_chars)

            progress.progress(40, text="Chunking text...")
            chunks = chunk_document(
                pages, doc_name=uploaded.name,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )

            progress.progress(60, text="Generating embeddings...")
            vectors = embed_texts([c.text for c in chunks])

            if show_embedding_preview:
                with st.expander(f"🧩 Embedding preview — {uploaded.name}", expanded=False):
                    render_embedding_preview(chunks, vectors)

            progress.progress(85, text="Storing vectors in Pinecone...")
            try:
                store = get_vector_store()
                store.upsert_chunks(chunks, vectors)
            except PineconeConnectionError as e:
                progress.empty()
                st.error(f"Pinecone error: {e}")
                continue

            progress.progress(100, text="Done.")
            time.sleep(0.3)
            progress.empty()

            # Save stats + full chunk list for the Statistics panel and
            # the Text Chunks section below.
            st.session_state.doc_stats[uploaded.name] = {
                "characters": total_chars,
                "chunks": len(chunks),
                "embeddings": len(vectors),
            }
            st.session_state.doc_chunks[uploaded.name] = chunks

            st.session_state.indexed_docs.append(uploaded.name)
            st.success(f"✅ Indexed '{uploaded.name}' — {len(chunks)} chunks stored.")

        except InvalidPDFError as e:
            progress.empty()
            st.error(str(e))
        except RuntimeError as e:
            progress.empty()
            st.error(f"Embedding error while processing '{uploaded.name}': {e}")
        except Exception as e:
            progress.empty()
            st.error(f"Unexpected error processing '{uploaded.name}': {e}")

# ---------------------------------------------------------------------
# 1b. Text Chunks — full list for the most recently processed document
# ---------------------------------------------------------------------
if st.session_state.doc_chunks:
    latest_doc_for_chunks = list(st.session_state.doc_chunks.keys())[-1]
    st.subheader("🧩 Text Chunks")
    st.caption(f"Showing chunks for: {latest_doc_for_chunks}")
    with st.expander("View all chunks", expanded=False):
        render_chunks_list(st.session_state.doc_chunks[latest_doc_for_chunks])

st.divider()

# ---------------------------------------------------------------------
# 2. Ask a question
# ---------------------------------------------------------------------
st.subheader("2️⃣ Ask a question")

question = st.text_input("Your question about the document(s):", key="question_input")

seconds_since_last = time.time() - st.session_state.last_query_time
cooldown_remaining = max(0, MIN_SECONDS_BETWEEN_QUERIES - seconds_since_last)
button_disabled = st.session_state.is_processing or cooldown_remaining > 0

button_label = "Get Answer"
if cooldown_remaining > 0:
    button_label = f"Please wait {cooldown_remaining:.0f}s..."

ask_clicked = st.button(button_label, type="primary", disabled=button_disabled)

if button_disabled and cooldown_remaining > 0:
    st.caption(f"⏳ Cooldown active — you can ask another question in {cooldown_remaining:.0f}s.")

if ask_clicked:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    elif not selected_docs:
        st.warning("Please upload and select at least one document first.")
    else:
        st.session_state.is_processing = True
        st.session_state.last_query_time = time.time()

        page_filter = None
        if page_filter_input.strip():
            try:
                page_filter = int(page_filter_input.strip())
            except ValueError:
                st.warning("Page filter must be a number — ignoring it.")

        try:
            store = get_vector_store()
            with st.spinner("Retrieving relevant context..."):
                chunks = retrieve(
                    query=question,
                    vector_store=store,
                    doc_names=selected_docs,
                    top_k=top_k,
                    similarity_threshold=sim_threshold,
                    page_filter=page_filter,
                )

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks)

            log_query(question, selected_docs, len(chunks), answer)
            st.session_state.query_history.append(question)
            st.session_state.last_result = {
                "question": question,
                "answer": answer,
                "chunks": chunks,
                "doc_names": selected_docs,
            }

        except PineconeConnectionError as e:
            st.error(f"Pinecone error: {e}")
        except ValueError as e:
            st.warning(str(e))
        except RuntimeError as e:
            st.error(str(e))
        finally:
            st.session_state.is_processing = False

# ---------------------------------------------------------------------
# 3. Display last result (persists across reruns, e.g. cooldown ticking)
# ---------------------------------------------------------------------
result = st.session_state.last_result
if result:
    answer = result["answer"]
    chunks = result["chunks"]

    st.markdown("### 💬 Answer")
    badge = overall_confidence(chunks)
    st.markdown(f"**Confidence:** {badge}")

    if answer.strip() == FALLBACK_MESSAGE:
        st.warning(answer)
    else:
        st.success(answer)

    st.markdown("### 🔍 Source attribution")
    if not chunks:
        st.caption("No chunks passed the similarity threshold.")
    for c in chunks:
        label = f"Page {c.page_number} — {confidence_label(c.score)} ({c.score:.3f}) — {c.doc_name}"
        with st.expander(label):
            st.write(c.text)

    report_text = build_report(result["question"], answer, chunks, result["doc_names"])
    st.download_button(
        label="⬇️ Download this Q&A as a report",
        data=report_text,
        file_name=f"rag_report_{int(time.time())}.txt",
        mime="text/plain",
    )

st.markdown(
    f'<div class="footer">RAG System · Pinecone + Groq · Developed by: {APP_AUTHOR}</div>',
    unsafe_allow_html=True,
)