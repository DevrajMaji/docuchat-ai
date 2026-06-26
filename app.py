import streamlit as st
import os
import time
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Chat bubbles */
    .user-bubble {
        background: #3b82f6;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-bubble {
        background: #1e293b;
        color: #e2e8f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #334155;
        font-size: 0.95rem;
    }
    .source-tag {
        background: #0f172a;
        color: #60a5fa;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 10px;
        border: 1px solid #1d4ed8;
        margin: 2px;
        display: inline-block;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    /* Main header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "rag_engine": None,
        "documents_loaded": False,
        "total_chunks": 0,
        "total_docs": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuChat AI")
    st.markdown("---")

    # API Key — reads from HF Spaces secret or manual input
    _env_key = os.environ.get("GROQ_API_KEY", "")
    if _env_key:
        groq_api_key = _env_key
        st.success("🔑 API key loaded from environment", icon="✅")
    else:
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get free key at console.groq.com"
        )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50,
                           help="Characters per chunk")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10,
                              help="Overlap between chunks")
    top_k = st.slider("Retrieved Chunks (top-k)", 1, 8, 3,
                      help="Chunks sent to LLM")
    model_name = st.selectbox(
        "LLM Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        help="Groq model to use"
    )

    st.markdown("---")
    st.markdown("### 📁 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and groq_api_key:
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                # 1. Extract text
                processor = DocumentProcessor(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
                all_chunks = []
                for f in uploaded_files:
                    chunks = processor.process_pdf(f)
                    all_chunks.extend(chunks)

                # 2. Build vector store
                vs = VectorStore()
                vs.build(all_chunks)
                st.session_state.vector_store = vs

                # 3. Init RAG engine
                st.session_state.rag_engine = RAGEngine(
                    vector_store=vs,
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    top_k=top_k
                )
                st.session_state.documents_loaded = True
                st.session_state.total_chunks = len(all_chunks)
                st.session_state.total_docs = len(uploaded_files)
                st.session_state.chat_history = []

            st.success(f"✅ {len(uploaded_files)} doc(s) ready!")

    elif uploaded_files and not groq_api_key:
        st.warning("⚠️ Enter your Groq API key first.")

    # Stats
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.5rem;font-weight:700;color:#60a5fa">{st.session_state.total_docs}</div>
                <div style="font-size:0.7rem;color:#94a3b8">DOCS</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.5rem;font-weight:700;color:#34d399">{st.session_state.total_chunks}</div>
                <div style="font-size:0.7rem;color:#94a3b8">CHUNKS</div>
            </div>""", unsafe_allow_html=True)

    # Clear
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
    <h1>🧠 DocuChat AI</h1>
    <p style="color:#64748b">End-to-End RAG • FAISS • Groq LLM • Streamlit</p>
</div>""", unsafe_allow_html=True)

if not st.session_state.documents_loaded:
    st.markdown("""
    <div style="text-align:center; padding: 60px 0; color: #475569;">
        <div style="font-size:4rem">📄</div>
        <h3>Upload PDFs in the sidebar to begin</h3>
        <p>The system will chunk, embed, and index your documents automatically.</p>
        <br>
        <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px 24px;min-width:140px;">
                <div style="font-size:1.5rem">📋</div><div style="color:#60a5fa;font-weight:600">Chunk</div>
                <div style="font-size:0.8rem;color:#64748b">Smart text splitting</div>
            </div>
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px 24px;min-width:140px;">
                <div style="font-size:1.5rem">🔢</div><div style="color:#60a5fa;font-weight:600">Embed</div>
                <div style="font-size:0.8rem;color:#64748b">Sentence-BERT</div>
            </div>
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px 24px;min-width:140px;">
                <div style="font-size:1.5rem">🔍</div><div style="color:#60a5fa;font-weight:600">Retrieve</div>
                <div style="font-size:0.8rem;color:#64748b">FAISS vector search</div>
            </div>
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px 24px;min-width:140px;">
                <div style="font-size:1.5rem">💬</div><div style="color:#60a5fa;font-weight:600">Generate</div>
                <div style="font-size:0.8rem;color:#64748b">Groq LLaMA3</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                src_html = ""
                if msg.get("sources"):
                    src_html = "<div style='margin-top:8px'>"
                    for s in msg["sources"]:
                        src_html += f'<span class="source-tag">📄 {s}</span>'
                    src_html += "</div>"
                st.markdown(
                    f'<div class="bot-bubble">🤖 {msg["content"]}{src_html}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="What does this document say about...?",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True, type="primary")

    if (send or user_input) and user_input.strip():
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🔍 Retrieving & generating..."):
            result = st.session_state.rag_engine.answer(question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", [])
        })
        st.rerun()
