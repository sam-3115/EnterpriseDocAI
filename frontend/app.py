import os
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# Page layout & styling
st.set_page_config(
    page_title="EnterpriseDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Custom CSS for rich dashboard experience
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .role-badge-admin {
        background-color: #EF4444;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .role-badge-user {
        background-color: #3B82F6;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .conf-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .conf-med {
        background-color: #FEF08A;
        color: #713F12;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .conf-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .conflict-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .chunk-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# API client methods
def api_login(username, password):
    try:
        res = requests.post(f"{BACKEND_URL}/api/auth/login", json={"username": username, "password": password}, timeout=10)
        return res.status_code, res.json()
    except Exception as e:
        return 500, {"detail": f"Backend connection error: {e}"}

def api_register(username, password, role="user"):
    try:
        res = requests.post(f"{BACKEND_URL}/api/auth/register", json={"username": username, "password": password, "role": role}, timeout=10)
        return res.status_code, res.json()
    except Exception as e:
        return 500, {"detail": f"Backend connection error: {e}"}

def api_get_documents():
    try:
        res = requests.get(f"{BACKEND_URL}/api/documents", timeout=10)
        if res.status_code == 200:
            return res.json().get("documents", [])
        return []
    except Exception:
        return []

def api_upload_document(uploaded_file, uploaded_by, role, api_key):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"uploaded_by": uploaded_by, "role": role, "api_key": api_key}
        res = requests.post(f"{BACKEND_URL}/api/documents/upload", files=files, data=data, timeout=60)
        return res.status_code, res.json()
    except Exception as e:
        return 500, {"detail": f"Upload error: {e}"}

def api_delete_document(doc_id, role):
    try:
        res = requests.delete(f"{BACKEND_URL}/api/documents/{doc_id}?role={role}", timeout=10)
        return res.status_code, res.json()
    except Exception as e:
        return 500, {"detail": f"Delete error: {e}"}

def api_get_chunks(doc_id):
    try:
        res = requests.get(f"{BACKEND_URL}/api/documents/{doc_id}/chunks", timeout=10)
        if res.status_code == 200:
            return res.json().get("chunks", [])
        return []
    except Exception:
        return []

def api_chat(query, conversation_history, top_k, api_key):
    try:
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "top_k": top_k,
            "api_key": api_key
        }
        res = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=40)
        if res.status_code == 200:
            return res.json()
        return {"answer": f"Error from backend: {res.text}", "sources": [], "confidence": "None"}
    except Exception as e:
        return {"answer": f"Failed to contact assistant service: {e}", "sources": [], "confidence": "None"}

def api_evaluate(top_k, api_key):
    try:
        params = {"top_k": top_k}
        if api_key:
            params["api_key"] = api_key
        res = requests.get(f"{BACKEND_URL}/api/evaluate", params=params, timeout=60)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception:
        return None

def api_get_stats():
    try:
        res = requests.get(f"{BACKEND_URL}/api/stats", timeout=5)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception:
        return None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## **EnterpriseDoc AI**")
    st.caption("HCLTech Hackathon · Full Production RAG")

    st.markdown("---")
    st.markdown("### 🔑 Gemini Flash API Key")
    key_input = st.text_input(
        "API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Paste your Gemini API key here. It will be used for live embeddings and Gemini Flash answers."
    )
    if key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = key_input
        st.success("API key updated for active session!")

    st.markdown("---")
    # Authentication & User State
    if not st.session_state.logged_in:
        st.markdown("### 🔐 Authentication")
        tab_login, tab_reg = st.tabs(["Login", "Register"])
        with tab_login:
            st.caption("Default Admin: `admin` / `admin123`")
            u_name = st.text_input("Username", key="s_u_name")
            u_pwd = st.text_input("Password", type="password", key="s_u_pwd")
            if st.button("Log In", type="primary", use_container_width=True):
                if u_name and u_pwd:
                    code, resp = api_login(u_name, u_pwd)
                    if code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_info = resp.get("user")
                        st.rerun()
                    else:
                        st.error(resp.get("detail", "Login failed"))
                else:
                    st.warning("Please enter credentials.")

        with tab_reg:
            r_name = st.text_input("New Username", key="s_r_name")
            r_pwd = st.text_input("New Password", type="password", key="s_r_pwd")
            r_role = st.selectbox("Role", ["user", "admin"], key="s_r_role")
            if st.button("Register Account", use_container_width=True):
                if r_name and r_pwd:
                    code, resp = api_register(r_name, r_pwd, r_role)
                    if code == 201:
                        st.success("Account registered! You may now log in.")
                    else:
                        st.error(resp.get("detail", "Registration failed"))
                else:
                    st.warning("Please fill in all fields.")
    else:
        user = st.session_state.user_info
        role = user.get("role", "user").upper()
        badge_cls = "role-badge-admin" if role == "ADMIN" else "role-badge-user"
        st.markdown(f"**Logged in as:** `{user.get('username')}`")
        st.markdown(f"**Role:** <span class='{badge_cls}'>{role}</span>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    menu = st.radio(
        "Sections",
        ["📊 Dashboard", "📁 Knowledge Base", "🤖 AI Assistant", "📚 Sources Explorer", "🧪 Evaluation & Benchmarks"],
        index=0
    )

# --- TOP STATUS BAR ---
head_col, status_col = st.columns([4, 1])
with head_col:
    st.markdown("<div class='main-header'>EnterpriseDoc AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Intelligent Enterprise Documentation Q&A with Grounded RAG & Citations</div>", unsafe_allow_html=True)

with status_col:
    stats = api_get_stats()
    if stats:
        st.success("🟢 Systems Operational")
    else:
        st.error("🔴 Backend Offline")

# ==========================================
# SECTION 1: DASHBOARD
# ==========================================
if menu == "📊 Dashboard":
    st.markdown("### 📊 System Dashboard & Statistics")
    
    db_stats = stats.get("database", {}) if stats else {}
    chroma_stats = stats.get("chroma", {}) if stats else {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", db_stats.get("total_documents", 0))
    with col2:
        st.metric("Indexed Chunks", db_stats.get("total_chunks", 0))
    with col3:
        st.metric("ChromaDB Vectors", chroma_stats.get("total_embeddings", 0))
    with col4:
        st.metric("Active Users", db_stats.get("total_users", 0))

    st.markdown("---")
    st.markdown("### 🏗️ RAG System Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Ingestion Pipeline")
        st.write("• Multi-format parsing (PDF, HTML, MD, TXT)\n• Text cleaning & normalization\n• Contextual chunking with metadata\n• SHA-256 deduplication")
    with c2:
        st.markdown("#### 2. Hybrid Retrieval")
        st.write("• Dense semantic retrieval (ChromaDB)\n• Sparse keyword matching (BM25)\n• Reciprocal Rank Fusion (RRF)\n• Cross-encoder contextual reranker")
    with c3:
        st.markdown("#### 3. Grounded Generation")
        st.write("• Gemini Flash LLM\n• Strict evidence adherence\n• Inline citation tracking\n• Contradiction & Confidence ratings")

    st.markdown("---")
    st.markdown("### 📑 Enterprise Documents Registry")
    docs = api_get_documents()
    if docs:
        df = pd.DataFrame([{
            "ID": d["id"],
            "Document": d["filename"],
            "Format": d["file_type"].upper(),
            "Size (KB)": round(d["file_size"]/1024, 2),
            "Chunks": d["chunk_count"],
            "Uploaded By": d["uploaded_by"],
            "Created At": str(d["created_at"])[:19]
        } for d in docs])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No enterprise documents loaded yet. Navigate to **📁 Knowledge Base** to upload documentation.")

# ==========================================
# SECTION 2: KNOWLEDGE BASE
# ==========================================
elif menu == "📁 Knowledge Base":
    st.markdown("### 📁 Document Management & Ingestion")

    is_admin = False
    cur_user = "guest"
    if st.session_state.logged_in and st.session_state.user_info:
        cur_user = st.session_state.user_info.get("username", "user")
        is_admin = (st.session_state.user_info.get("role") == "admin")

    st.markdown("#### 📤 Upload Enterprise Document (Admin Only)")
    if is_admin:
        with st.form("doc_upload_form", clear_on_submit=True):
            f_upload = st.file_uploader("Select file", type=["pdf", "html", "htm", "md", "markdown", "txt"])
            submitted = st.form_submit_button("Upload & Index Document", type="primary")

            if submitted:
                if not f_upload:
                    st.warning("Please choose a document to upload.")
                else:
                    with st.spinner(f"Extracting, chunking and embedding '{f_upload.name}'..."):
                        code, resp = api_upload_document(
                            uploaded_file=f_upload,
                            uploaded_by=cur_user,
                            role="admin",
                            api_key=st.session_state.gemini_api_key
                        )
                        if code == 200:
                            st.success(f"✅ {resp.get('message')}")
                            st.rerun()
                        elif code == 409:
                            st.error(f"⚠️ **Duplicate Upload Rejected**: {resp.get('detail')}")
                        elif code == 403:
                            st.error(f"⛔ **Forbidden**: {resp.get('detail')}")
                        else:
                            st.error(f"❌ Upload Failed: {resp.get('detail')}")
    else:
        st.warning("🔒 **Admin Required**: Document uploads and deletions are restricted to administrators. Please log in as `admin`.")

    st.markdown("---")
    st.markdown("#### 📚 Active Repository Documents")
    docs = api_get_documents()
    if not docs:
        st.info("No documents found in knowledge base.")
    else:
        for d in docs:
            with st.expander(f"📄 **{d['filename']}** ({d['file_type'].upper()} · {d['chunk_count']} Chunks · {round(d['file_size']/1024, 1)} KB)"):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"**ID:** `{d['id']}` | **Uploaded by:** `{d['uploaded_by']}` | **Date:** `{str(d['created_at'])[:19]}`")
                    st.caption(f"SHA-256: `{d['file_hash']}`")
                with col_b:
                    if is_admin:
                        if st.button("🗑️ Delete Document", key=f"del_btn_{d['id']}", type="secondary"):
                            sc, dr = api_delete_document(d['id'], role="admin")
                            if sc == 200:
                                st.success("Document deleted.")
                                st.rerun()
                            else:
                                st.error(dr.get("detail", "Delete failed"))

                st.markdown("##### Contextual Chunks & Metadata Explorer")
                chunks = api_get_chunks(d['id'])
                for c in chunks[:6]:
                    st.markdown(f"""
                    <div class='chunk-card'>
                        <strong>Chunk #{c['chunk_index'] + 1}</strong> &nbsp;|&nbsp; 
                        <em>Page:</em> {c['page_number']} &nbsp;|&nbsp; 
                        <em>Section:</em> <strong>{c['section_title']}</strong>
                        <pre style='white-space: pre-wrap; font-size: 0.85rem; margin-top: 5px; background: #F8FAFC; padding: 6px; border-radius: 4px;'>{c['content']}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                if len(chunks) > 6:
                    st.caption(f"... {len(chunks)-6} additional chunks stored.")

# ==========================================
# SECTION 3: AI ASSISTANT (RAG Q&A)
# ==========================================
elif menu == "🤖 AI Assistant":
    st.markdown("### 🤖 Enterprise Documentation Q&A Assistant")
    st.caption("Ask questions about company policies, architecture, APIs, and guidelines. Answers are strictly verified against indexed documents.")

    # Top control bar
    top_c1, top_c2 = st.columns([4, 1])
    with top_c2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 View {len(msg['sources'])} Cited Sources & Metadata"):
                    for s in msg["sources"]:
                        st.markdown(f"**[Source {s['source_id']}]** 📄 `{s['filename']}` | Page {s['page_number']} | Section: *{s['section_title']}* (Score: {s.get('score', 0.0)})")
                        st.caption(s["excerpt"])
            if msg.get("confidence"):
                conf = msg["confidence"]
                conf_cls = "conf-high" if conf == "High" else ("conf-med" if conf == "Medium" else "conf-low")
                st.markdown(f"<span class='{conf_cls}'>Confidence: {conf}</span>", unsafe_allow_html=True)

    # Chat Input
    query_input = st.chat_input("Ask a question about enterprise documentation...")
    if query_input:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving passages and generating grounded answer with Gemini Flash..."):
                rag_resp = api_chat(
                    query=query_input,
                    conversation_history=st.session_state.chat_history[:-1],
                    top_k=4,
                    api_key=st.session_state.gemini_api_key
                )

                answer = rag_resp.get("answer", "")
                sources = rag_resp.get("sources", [])
                conf = rag_resp.get("confidence", "Medium")
                has_conflict = rag_resp.get("has_contradiction", False)

                st.markdown(answer)

                if has_conflict:
                    st.markdown("""
                    <div class='conflict-box'>
                        ⚠️ <strong>Potential Contradiction Detected:</strong> Multiple retrieved document sources contain conflicting policies or differing values. Please review the cited sources below.
                    </div>
                    """, unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📚 View {len(sources)} Cited Sources & Metadata"):
                        for s in sources:
                            st.markdown(f"**[Source {s['source_id']}]** 📄 `{s['filename']}` | Page {s['page_number']} | Section: *{s['section_title']}* (Score: {s.get('score', 0.0)})")
                            st.caption(s["excerpt"])

                conf_cls = "conf-high" if conf == "High" else ("conf-med" if conf == "Medium" else "conf-low")
                st.markdown(f"<span class='{conf_cls}'>Confidence: {conf}</span>", unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": conf
                })

# ==========================================
# SECTION 4: SOURCES EXPLORER
# ==========================================
elif menu == "📚 Sources Explorer":
    st.markdown("### 📚 Enterprise Sources Explorer")
    st.caption("Inspect indexed documents, review page structures, and view raw extracted text.")

    docs = api_get_documents()
    if not docs:
        st.info("No documents indexed yet.")
    else:
        doc_options = {d["filename"]: d["id"] for d in docs}
        selected_name = st.selectbox("Select Document to Inspect", list(doc_options.keys()))
        selected_id = doc_options[selected_name]

        chunks = api_get_chunks(selected_id)
        st.markdown(f"#### Document Chunks for **{selected_name}** ({len(chunks)} total)")

        for c in chunks:
            with st.container():
                st.markdown(f"""
                <div class='chunk-card'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span><strong>Chunk #{c['chunk_index'] + 1}</strong> &nbsp;📄 Page: {c['page_number']}</span>
                        <span style='color: #4B5563;'>Section: <strong>{c['section_title']}</strong></span>
                    </div>
                    <div style='margin-top: 8px; font-size: 0.9rem; line-height: 1.5;'>{c['content']}</div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# SECTION 5: EVALUATION & BENCHMARKS
# ==========================================
elif menu == "🧪 Evaluation & Benchmarks":
    st.markdown("### 🧪 RAG Evaluation & Benchmark Suite")
    st.write("Evaluate Recall@K, Precision@K, MRR, Answer Faithfulness, and Citation Correctness across **Direct**, **Paraphrased**, **Conflicting**, **Ambiguous**, and **Unanswerable** questions.")

    eval_col1, eval_col2 = st.columns([3, 1])
    with eval_col1:
        st.info("The benchmark evaluates retrieval quality and generation accuracy against ground-truth answers and citations.")
    with eval_col2:
        run_btn = st.button("🚀 Run Full Benchmark", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing benchmark across all 5 test categories..."):
            eval_data = api_evaluate(top_k=3, api_key=st.session_state.gemini_api_key)
            if eval_data:
                st.session_state.eval_results = eval_data
                st.success("Benchmark completed successfully!")
            else:
                st.error("Failed to execute benchmark.")

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        summary = results.get("summary", {})
        cases = results.get("test_results", [])

        st.markdown("#### 📈 Benchmark Summary Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Recall@3", f"{summary.get('mean_recall_at_k', 0.0) * 100:.1f}%")
        with m2:
            st.metric("Precision@3", f"{summary.get('mean_precision_at_k', 0.0) * 100:.1f}%")
        with m3:
            st.metric("MRR", f"{summary.get('mean_mrr', 0.0):.3f}")
        with m4:
            st.metric("Faithfulness", f"{summary.get('mean_faithfulness', 0.0) * 100:.1f}%")
        with m5:
            st.metric("Citation Correctness", f"{summary.get('mean_citation_correctness', 0.0) * 100:.1f}%")

        st.markdown("---")
        st.markdown("#### 📋 Detailed Test Case Breakdown")
        table_rows = []
        for c in cases:
            table_rows.append({
                "Test ID": c["id"],
                "Category": c["category"],
                "Question": c["question"],
                "Recall@3": f"{c['recall_at_k'] * 100:.0f}%",
                "Precision@3": f"{c['precision_at_k'] * 100:.0f}%",
                "MRR": c["mrr"],
                "Faithfulness": f"{c['faithfulness'] * 100:.0f}%",
                "Citation Correct": f"{c['citation_correctness'] * 100:.0f}%",
                "Confidence": c["confidence"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🔍 Inspected Answers & Evidence per Category")
        for c in cases:
            with st.expander(f"**{c['category']}**: *{c['question']}*"):
                st.write(f"**Ground Truth Target:** `{c['ground_truth_doc']}`")
                st.write(f"**Generated Answer:** {c['answer']}")
                st.caption(f"Confidence: {c['confidence']} | Sources Retrieved: {c['sources_count']}")
