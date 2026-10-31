import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import get_document_by_hash, compute_sha256, save_document, save_chunks, update_document_chunk_count
from backend.parsers import parse_document
from backend.chunker import contextual_chunk_document
from backend.vector_store import add_chunks_to_chroma
from backend.config import UPLOAD_DIR, GEMINI_API_KEY

def seed_all_documents():
    sample_dir = PROJECT_ROOT / "sample_docs"
    files = [
        ("sample_policy.txt", "txt"),
        ("sample_guide.md", "md"),
        ("sample_api.html", "html"),
        ("sample_handbook.pdf", "pdf"),
    ]

    print(f"Uploading and indexing all sample documents into {UPLOAD_DIR}...")

    for fname, ftype in files:
        fpath = sample_dir / fname
        if not fpath.exists():
            print(f"File not found: {fpath}")
            continue

        with open(fpath, "rb") as f:
            content = f.read()

        file_hash = compute_sha256(content)
        existing = get_document_by_hash(file_hash)
        if existing:
            print(f"[SKIP] '{fname}' already exists in database (ID: {existing['id']}).")
            continue

        # Save to data/documents/
        saved_path = UPLOAD_DIR / f"{file_hash[:12]}_{fname}"
        with open(saved_path, "wb") as f:
            f.write(content)

        # SQLite doc record
        doc_id = save_document(
            filename=fname,
            file_type=ftype,
            file_hash=file_hash,
            file_size=len(content),
            chunk_count=0,
            uploaded_by="admin"
        )

        # Parse
        parsed_segments = parse_document(saved_path, ftype)
        # Chunk
        chunks = contextual_chunk_document(
            doc_id=doc_id,
            filename=fname,
            parsed_segments=parsed_segments,
            chunk_size=700,
            chunk_overlap=120
        )

        # Save chunks in SQLite
        save_chunks(chunks)
        update_document_chunk_count(doc_id, len(chunks))

        # Index in ChromaDB
        add_chunks_to_chroma(chunks, api_key=GEMINI_API_KEY)
        print(f"[SUCCESS] Uploaded '{fname}' -> ID: {doc_id}, Chunks: {len(chunks)}, Saved to: {saved_path.name}")

    print("\nAll sample documents are successfully uploaded and indexed!")

if __name__ == "__main__":
    seed_all_documents()

