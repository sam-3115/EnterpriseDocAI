import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import (
    UPLOAD_DIR,
    DEFAULT_ADMIN_USERNAME,
    DEFAULT_ADMIN_PASSWORD,
    GEMINI_API_KEY
)
from backend.database import (
    init_db,
    compute_sha256,
    get_document_by_hash,
    save_document,
    update_document_chunk_count,
    list_documents,
    get_document_by_id,
    delete_document as db_delete_document,
    save_chunks,
    get_chunks_by_doc_id,
    list_users,
    get_stats
)
from backend.auth import hash_password, authenticate_user, register_user
from backend.parsers import parse_document
from backend.chunker import contextual_chunk_document
from backend.vector_store import add_chunks_to_chroma, delete_chunks_by_doc_id, search_vector_store, get_chroma_stats
from backend.retrieval import hybrid_retrieve
from backend.rag_engine import generate_rag_answer
from backend.evaluation import run_evaluation_benchmark

# Initialize FastAPI App
app = FastAPI(
    title="EnterpriseDoc AI Backend",
    description="Backend API for Enterprise Documentation Q&A Assistant using RAG",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request Models
class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[ChatMessage]] = []
    top_k: Optional[int] = 5
    api_key: Optional[str] = None

def setup_system():
    """Ensure database and directories are initialized."""
    admin_hash = hash_password(DEFAULT_ADMIN_PASSWORD)
    init_db(seed_admin=True, admin_username=DEFAULT_ADMIN_USERNAME, admin_password_hash=admin_hash)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Run initialization immediately
setup_system()

@app.on_event("startup")
def on_startup():
    setup_system()
    print("[Startup] Database initialized and default admin ready.")

# --- Authentication Endpoints ---

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """Register a new user account (Admin or User role)."""
    try:
        user = register_user(req.username, req.password, req.role or "user")
        return {"status": "success", "message": "User registered successfully", "user": user}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {e}")

@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Authenticate user and return role information."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return {
        "status": "success",
        "message": "Login successful",
        "user": user
    }

@app.get("/api/auth/users")
def get_users():
    """List all registered users (IDs, usernames, roles, creation dates)."""
    return {"users": list_users()}

# --- Document Management Endpoints ---

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form("admin"),
    role: str = Form("admin"),
    api_key: Optional[str] = Form(None)
):
    """
    Admin-only endpoint to upload documents (PDF, HTML, MD, TXT).
    Validates role, detects duplicates via SHA-256, extracts text,
    creates contextual chunks, and indexes into ChromaDB and SQLite.
    """
    # 1. Role verification: Only Admin can upload
    if role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Only Administrators can upload documents."
        )

    # 2. Check supported extension
    filename = file.filename
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".html", ".htm", ".md", ".markdown", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: PDF, HTML, Markdown, TXT."
        )

    # 3. Read file content and check duplicate
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    file_hash = compute_sha256(content)
    existing_doc = get_document_by_hash(file_hash)
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate document detected: '{filename}' has exact same content as previously uploaded document '{existing_doc['filename']}' (ID: {existing_doc['id']})."
        )

    # 4. Save file to disk
    file_type = ext.replace(".", "")
    saved_path = UPLOAD_DIR / f"{file_hash[:12]}_{filename}"
    with open(saved_path, "wb") as f:
        f.write(content)

    # 5. Insert document metadata in SQLite
    doc_id = save_document(
        filename=filename,
        file_type=file_type,
        file_hash=file_hash,
        file_size=len(content),
        chunk_count=0,
        uploaded_by=uploaded_by
    )

    try:
        # 6. Extract and clean document text
        parsed_segments = parse_document(saved_path, file_type)
        if not parsed_segments:
            raise ValueError(f"Could not extract readable text from '{filename}'.")

        # 7. Contextual text chunking
        chunks = contextual_chunk_document(
            doc_id=doc_id,
            filename=filename,
            parsed_segments=parsed_segments,
            chunk_size=700,
            chunk_overlap=120
        )

        if not chunks:
            raise ValueError("Document yielded no chunks after contextual processing.")

        # 8. Save chunks in SQLite
        save_chunks(chunks)
        update_document_chunk_count(doc_id, len(chunks))

        # 9. Embed and index in ChromaDB
        effective_key = api_key or GEMINI_API_KEY
        add_chunks_to_chroma(chunks, api_key=effective_key)

        return {
            "status": "success",
            "message": f"Document '{filename}' successfully processed and indexed.",
            "document": {
                "id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": len(content),
                "chunk_count": len(chunks),
                "file_hash": file_hash,
                "uploaded_by": uploaded_by
            }
        }
    except Exception as e:
        # Cleanup on failure
        db_delete_document(doc_id)
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/api/documents")
def get_documents():
    """Retrieve list of all uploaded documents with metadata."""
    docs = list_documents()
    return {"documents": docs}

@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    """Retrieve details for a specific document."""
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return {"document": doc}

@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: int):
    """Retrieve all chunks belonging to a document."""
    chunks = get_chunks_by_doc_id(doc_id)
    return {"doc_id": doc_id, "chunks": chunks}

@app.delete("/api/documents/{doc_id}")
def delete_document_endpoint(
    doc_id: int,
    role: str = Query("admin")
):
    """
    Admin-only endpoint to delete a document and its vector embeddings.
    """
    if role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Only Administrators can delete documents."
        )

    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # 1. Delete from ChromaDB
    delete_chunks_by_doc_id(doc_id)

    # 2. Delete from SQLite
    db_delete_document(doc_id)

    # 3. Delete physical file from disk
    file_pattern = f"{doc['file_hash'][:12]}_{doc['filename']}"
    for f in UPLOAD_DIR.glob(f"*{file_pattern}*"):
        try:
            f.unlink()
        except Exception:
            pass

    return {
        "status": "success",
        "message": f"Document '{doc['filename']}' (ID: {doc_id}) successfully deleted."
    }

# --- Search, RAG & Question Answering Endpoints ---

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Part 2 deliverable: RAG & Intelligent Question Answering endpoint.
    Performs hybrid retrieval, reciprocal rank fusion, cross-encoder reranking,
    and strictly-grounded answer generation using Gemini Flash LLM.
    """
    history_dicts = [{"role": m.role, "content": m.content} for m in req.conversation_history] if req.conversation_history else []
    rag_result = generate_rag_answer(
        query=req.query,
        conversation_history=history_dicts,
        top_k=req.top_k or 5,
        api_key=req.api_key
    )
    return rag_result

@app.get("/api/evaluate")
def evaluate_endpoint(
    top_k: int = Query(3, ge=1, le=10),
    api_key: Optional[str] = Query(None)
):
    """
    Part 3 deliverable: Evaluation benchmark endpoint.
    Evaluates Recall@K, Precision@K, MRR, Faithfulness, and Citation Correctness
    across Direct, Paraphrased, Conflicting, Ambiguous, and Unanswerable queries.
    """
    effective_key = api_key or GEMINI_API_KEY
    benchmark_report = run_evaluation_benchmark(api_key=effective_key, top_k=top_k)
    return benchmark_report

@app.get("/api/search")
def search_knowledge_base(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
    api_key: Optional[str] = Query(None)
):
    """
    Part 1 deliverable: Searchable knowledge base semantic retrieval endpoint.
    Retrieves matching document chunks preserving document name, page, and section.
    """
    effective_key = api_key or GEMINI_API_KEY
    results = search_vector_store(query=query, top_k=top_k, api_key=effective_key)
    return {
        "query": query,
        "results_count": len(results),
        "results": results
    }

# --- System Stats & Health ---

@app.get("/api/stats")
def get_system_stats():
    """Return system statistics (docs, chunks, users, ChromaDB status)."""
    db_stats = get_stats()
    chroma_stats = get_chroma_stats()
    return {
        "database": db_stats,
        "chroma": chroma_stats
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "EnterpriseDoc AI"}

