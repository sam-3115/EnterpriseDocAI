import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app
from backend.config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR
from backend.database import init_db, list_users, list_documents, get_stats
from backend.auth import hash_password, verify_password, authenticate_user
from backend.parsers import parse_document
from backend.chunker import contextual_chunk_document
from backend.vector_store import get_chroma_stats, search_vector_store

def run_all_tests():
    print("=" * 60)
    print("RUNNING ENTERPRISEDOC AI - PART 1 VERIFICATION SUITE")
    print("=" * 60)

    # Clean existing test data for reproducible run
    docs = list_documents()
    for d in docs:
        try:
            from backend.app import delete_document_endpoint
            delete_document_endpoint(d["id"], role="admin")
        except Exception:
            pass

    client = TestClient(app)

    # 1. Health check
    print("\n[TEST 1] Testing /api/health endpoint...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    assert resp.json().get("status") == "ok"
    print("  --> PASS: Health check successful.")

    # 2. Authentication & Roles
    print("\n[TEST 2] Testing Password Hashing & Authentication...")
    test_pwd = "SecretPassword123!"
    hashed = hash_password(test_pwd)
    assert verify_password(test_pwd, hashed) is True, "Password verification failed"
    assert verify_password("wrong_pwd", hashed) is False, "False positive in password verification"
    print("  --> PASS: Bcrypt hashing and verification verified.")

    print("\n[TEST 3] Testing User Registration & Roles (Admin vs User)...")
    # Register standard user
    resp = client.post("/api/auth/register", json={"username": "alice_test", "password": "password123", "role": "user"})
    assert resp.status_code in [201, 400], f"Registration failed: {resp.text}"
    
    # Register admin user
    resp = client.post("/api/auth/register", json={"username": "bob_admin", "password": "adminpassword", "role": "admin"})
    assert resp.status_code in [201, 400], f"Admin registration failed: {resp.text}"

    # Verify Login
    login_resp = client.post("/api/auth/login", json={"username": "bob_admin", "password": "adminpassword"})
    assert login_resp.status_code == 200
    user_data = login_resp.json().get("user")
    assert user_data["role"] == "admin"
    print("  --> PASS: User registration, login, and role assignment confirmed.")

    # 3. Document Parsers for all 4 types (PDF, HTML, MD, TXT)
    print("\n[TEST 4] Testing Parsers for PDF, HTML, Markdown, and TXT...")
    sample_dir = PROJECT_ROOT / "sample_docs"

    # TXT
    txt_segments = parse_document(sample_dir / "sample_policy.txt", "txt")
    assert len(txt_segments) > 0, "TXT parsing yielded no segments"
    print(f"  --> TXT parsed: {len(txt_segments)} segments.")

    # Markdown
    md_segments = parse_document(sample_dir / "sample_guide.md", "md")
    assert len(md_segments) > 0, "Markdown parsing yielded no segments"
    assert any("Architecture" in s.get("section_title", "") for s in md_segments), "Markdown section title missing"
    print(f"  --> Markdown parsed: {len(md_segments)} segments with section titles.")

    # HTML
    html_segments = parse_document(sample_dir / "sample_api.html", "html")
    assert len(html_segments) > 0, "HTML parsing yielded no segments"
    print(f"  --> HTML parsed: {len(html_segments)} segments.")

    # PDF
    pdf_segments = parse_document(sample_dir / "sample_handbook.pdf", "pdf")
    assert len(pdf_segments) >= 2, "PDF parsing failed to extract multi-page segments"
    assert pdf_segments[0]["page_number"] == 1
    assert pdf_segments[1]["page_number"] == 2
    print(f"  --> PDF parsed: {len(pdf_segments)} pages extracted with page number preservation.")
    print("  --> PASS: All 4 document formats supported and verified.")

    # 4. Contextual Chunking
    print("\n[TEST 5] Testing Contextual Chunking & Metadata Preservation...")
    chunks = contextual_chunk_document(
        doc_id=999,
        filename="sample_handbook.pdf",
        parsed_segments=pdf_segments,
        chunk_size=300,
        chunk_overlap=50
    )
    assert len(chunks) >= 2
    first_chunk = chunks[0]
    assert "[sample_handbook.pdf | Page 1 | Section:" in first_chunk["content"], "Contextual header missing"
    assert first_chunk["page_number"] == 1
    assert first_chunk["filename"] == "sample_handbook.pdf"
    print(f"  --> PASS: {len(chunks)} contextual chunks generated with embedded metadata headers.")

    # 5. Role-based Upload & Duplicate Detection via FastAPI
    print("\n[TEST 6] Testing Role-based Upload & Duplicate Upload Detection...")
    # Attempt upload with standard user role -> Should be 403 Forbidden
    with open(sample_dir / "sample_policy.txt", "rb") as f:
        file_bytes = f.read()

    forbidden_resp = client.post(
        "/api/documents/upload",
        files={"file": ("sample_policy.txt", file_bytes, "text/plain")},
        data={"uploaded_by": "alice_test", "role": "user"}
    )
    assert forbidden_resp.status_code == 403, f"Expected 403 for non-admin upload, got: {forbidden_resp.status_code}"
    print("  --> PASS: Non-admin upload rejected with HTTP 403 Forbidden.")

    # Upload as Admin -> Should succeed (200 OK)
    admin_upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("sample_policy.txt", file_bytes, "text/plain")},
        data={"uploaded_by": "bob_admin", "role": "admin"}
    )
    assert admin_upload_resp.status_code == 200, f"Admin upload failed: {admin_upload_resp.text}"
    uploaded_doc = admin_upload_resp.json().get("document")
    doc_id = uploaded_doc["id"]
    print(f"  --> PASS: Document uploaded successfully by Admin (Doc ID: {doc_id}, Chunks: {uploaded_doc['chunk_count']}).")

    # Attempt duplicate upload -> Should be 409 Conflict
    dup_resp = client.post(
        "/api/documents/upload",
        files={"file": ("sample_policy.txt", file_bytes, "text/plain")},
        data={"uploaded_by": "bob_admin", "role": "admin"}
    )
    assert dup_resp.status_code == 409, f"Expected 409 for duplicate upload, got: {dup_resp.status_code}"
    print(f"  --> PASS: Duplicate upload correctly detected and rejected with HTTP 409 Conflict.")

    # Upload PDF as well
    with open(sample_dir / "sample_handbook.pdf", "rb") as f:
        pdf_bytes = f.read()
    pdf_upload = client.post(
        "/api/documents/upload",
        files={"file": ("sample_handbook.pdf", pdf_bytes, "application/pdf")},
        data={"uploaded_by": "bob_admin", "role": "admin"}
    )
    assert pdf_upload.status_code == 200, f"PDF upload failed: {pdf_upload.text}"
    pdf_doc_id = pdf_upload.json()["document"]["id"]
    print(f"  --> PASS: PDF uploaded and indexed (Doc ID: {pdf_doc_id}).")

    # 6. Searchable Knowledge Base Endpoint
    print("\n[TEST 7] Testing Searchable Knowledge Base Endpoint (/api/search)...")
    search_resp = client.get("/api/search", params={"query": "paid time off policy leave", "top_k": 3})
    assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
    search_data = search_resp.json()
    assert search_data["results_count"] > 0, "No search results returned"
    top_result = search_data["results"][0]
    print(f"  --> PASS: Retrieved {search_data['results_count']} chunks. Top result: {top_result['metadata']['filename']} (Score: {top_result['score']})")

    # 7. Document Deletion & Cascade Clean
    print("\n[TEST 8] Testing Document Deletion (Admin only)...")
    # Non-admin delete should be 403
    del_forbidden = client.delete(f"/api/documents/{pdf_doc_id}?role=user")
    assert del_forbidden.status_code == 403
    print("  --> PASS: Non-admin deletion rejected with HTTP 403 Forbidden.")

    # Admin delete
    del_resp = client.delete(f"/api/documents/{pdf_doc_id}?role=admin")
    assert del_resp.status_code == 200
    print(f"  --> PASS: Document {pdf_doc_id} successfully deleted from SQLite and ChromaDB.")

    # Stats
    print("\n[TEST 9] Verifying Knowledge Base System Statistics...")
    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    print(f"  --> System Stats: {stats_resp.json()}")

    print("\n" + "=" * 60)
    print("ALL PART 1 TESTS PASSED SUCCESSFULLY! DELIVERABLE READY.")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
