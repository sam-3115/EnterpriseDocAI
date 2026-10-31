# EnterpriseDoc AI – HCLTech Hackathon Project

A production-ready, fully functional **Enterprise Documentation Q&A Assistant** built using Retrieval-Augmented Generation (RAG).

---

## 🌟 Technology Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI & Uvicorn
- **Document Processing**: LangChain & Custom Parsers (`pypdf`, `BeautifulSoup4`)
- **Dense Embeddings**: Google Gemini `text-embedding-004`
- **Generative LLM**: Google Gemini Flash
- **Vector Database**: ChromaDB (Persistent storage with cosine distance)
- **Keyword Retrieval**: BM25 (`rank-bm25`)
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with Cross-Encoder contextual reranking
- **Relational Metadata**: SQLite
- **Security & Authentication**: Salted bcrypt password hashing with role-based access control (Admin & User roles)

---

## 📁 Project Architecture & Structure
```text
EnterpriseDocAI/
├── .env.example              # Environment variables template
├── .env                      # Active credentials & configuration
├── requirements.txt          # Python dependencies
├── run_backend.py            # Launch FastAPI backend (port 8000)
├── run_frontend.py           # Launch Streamlit frontend (port 8501)
├── backend/
│   ├── app.py                # FastAPI REST endpoints (auth, docs, chat, evaluate, stats)
│   ├── auth.py               # Bcrypt password hashing & authentication logic
│   ├── chunker.py            # Contextual chunking with metadata preservation
│   ├── config.py             # Global settings, paths, and environment bindings
│   ├── database.py           # SQLite relational models & SHA-256 deduplication
│   ├── evaluation.py         # Evaluation benchmark suite (Recall@K, Precision@K, MRR, Faithfulness)
│   ├── parsers.py            # Document extractors (PDF, HTML, Markdown, TXT)
│   ├── rag_engine.py         # Grounded RAG synthesis, Gemini Flash integration & citations
│   ├── retrieval.py          # BM25 + Vector + Reciprocal Rank Fusion + Reranking
│   └── vector_store.py       # ChromaDB persistent collection & Gemini embeddings
├── frontend/
│   └── app.py                # Streamlit Dashboard with 5 integrated sections
├── sample_docs/              # Multi-format enterprise documents
│   ├── sample_api.html       # HTML specification with headers and tables
│   ├── sample_guide.md       # Markdown onboarding guide with structured sections
│   ├── sample_handbook.pdf   # Multi-page PDF handbook with page numbering
│   └── sample_policy.txt     # Enterprise policy handbook
└── tests/
    ├── test_part1.py         # Part 1 verification suite (Auth & Document Management)
    └── test_part2_and_3.py   # Part 2 & 3 verification suite (RAG, Citations & Benchmark)
```

---

## 🚀 Key Functional Modules

### PART 1: Authentication & Document Management
1. **Bcrypt Authentication & RBAC**:
   - Secure salted password hashing.
   - Strict separation between **Admin** and **User** permissions.
   - Seeded default administrator account: `admin` / `admin123`.
2. **Multi-Format Ingestion**:
   - Clean parsing for **PDF** (page-by-page tracking), **HTML** (DOM cleaning & H1-H4 sectioning), **Markdown** (header preservation), and **TXT** (paragraph normalization).
3. **Contextual Text Chunking**:
   - Recursive text chunking (~700 chars, 120 overlap).
   - Injects document name, page number, and section titles into every chunk.
4. **Duplicate Upload Detection**:
   - SHA-256 checksum check rejects identical re-uploads with `HTTP 409 Conflict`.
5. **Persistent Storage**:
   - Relational metadata in SQLite (`users`, `documents`, `document_chunks`).
   - Vector embeddings persisted in ChromaDB (`data/chroma_db/`).

### PART 2: RAG & Intelligent Question Answering
1. **Hybrid Retrieval**:
   - Dense semantic retrieval via ChromaDB.
   - Sparse keyword matching via BM25Okapi.
   - Reciprocal Rank Fusion (RRF) with constant $k=60$.
   - Cross-Encoder contextual reranker scoring exact phrase matches and token coverage.
2. **Strict Evidence Grounding**:
   - Formats retrieved chunks with explicit source citations.
   - System instructions compel Gemini Flash to answer *strictly* from document context.
   - Generates standardized not-found response when evidence is insufficient:
     *"I could not find sufficient information in the provided enterprise documentation to answer this question."*
3. **Citation & Metadata Transparency**:
   - Inline citation tags (`[Source N]`).
   - Expandable source details with document name, page number, section, and text excerpt.
4. **Contradiction Detection & Confidence Rating**:
   - Scans retrieved passages for conflicting statements and outputs a contradiction warning.
   - Computes evidence-based confidence rating (*High*, *Medium*, *Low*, *None*).
5. **Conversational Dialogue Continuity**:
   - Maintains chat history context across multi-turn follow-up queries.

### PART 3: Frontend, Evaluation & Deployment
1. **Streamlit Multi-Section Dashboard**:
   - 📊 **Dashboard**: System health, document counts, chunk metrics, architecture diagram.
   - 📁 **Knowledge Base**: Admin drag-and-drop document upload, deletion, chunk explorer.
   - 🤖 **AI Assistant**: Conversational chat interface with expandable citations and confidence badges.
   - 📚 **Sources Explorer**: Browse all indexed documents and examine raw extracted passages.
   - 🧪 **Evaluation & Benchmarks**: Interactive benchmarking suite with metrics tables and charts.
2. **Evaluation Metrics**:
   - **Recall@K**, **Precision@K**, and **MRR** (Mean Reciprocal Rank).
   - **Answer Faithfulness** and **Citation Correctness**.
   - Tests across **Direct**, **Paraphrased**, **Conflicting**, **Ambiguous**, and **Unanswerable** questions.

---

## ⚡ Quickstart Instructions

### Step 1: Run the Complete Test Suite
```powershell
cd C:\Users\samar\.gemini\antigravity\scratch\EnterpriseDocAI

# Run Part 1 Test Suite
.\venv\Scripts\python.exe tests\test_part1.py

# Run Part 2 & Part 3 Test Suite
.\venv\Scripts\python.exe tests\test_part2_and_3.py
```

### Step 2: Start the FastAPI Backend
```powershell
.\venv\Scripts\python.exe run_backend.py
```
- API Base: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### Step 3: Start the Streamlit Frontend
In a second terminal:
```powershell
.\venv\Scripts\python.exe run_frontend.py
```
- Web Application: `http://localhost:8501`

### Step 4: Login Credentials & API Key
- **Default Admin Account**:
  - Username: `admin`
  - Password: `admin123`
- **Gemini API Key**:
  - Enter your Gemini API key in the sidebar's **Gemini API Configuration** field or set `GEMINI_API_KEY` in `.env`.
