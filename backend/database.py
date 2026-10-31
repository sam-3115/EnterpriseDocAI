import sqlite3
import hashlib
from typing import Optional, List, Dict, Any
from backend.config import SQLITE_DB_PATH

def get_db_connection() -> sqlite3.Connection:
    """Return a database connection with Row factory enabled."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(seed_admin: bool = True, admin_username: str = "admin", admin_password_hash: str = "") -> None:
    """Initialize SQLite database tables."""
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                uploaded_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Document Chunks Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_id TEXT UNIQUE NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER DEFAULT 1,
                section_title TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """)
        
        # Optional: Seed default admin if table is empty and hash is provided
        if seed_admin and admin_password_hash:
            cursor.execute("SELECT id FROM users WHERE username = ?", (admin_username,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (admin_username, admin_password_hash, "admin")
                )
        conn.commit()

# --- User Management ---

def create_user(username: str, password_hash: str, role: str = "user") -> int:
    """Create a new user with specified role ('admin' or 'user')."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username.strip(), password_hash, role.strip().lower())
        )
        conn.commit()
        return cursor.lastrowid

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record by username."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None

def list_users() -> List[Dict[str, Any]]:
    """List all registered users (excluding sensitive password hashes)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, created_at FROM users ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]

# --- Document Management & Deduplication ---

def compute_sha256(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file content for exact duplicate detection."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_bytes)
    return sha256_hash.hexdigest()

def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if document already exists using content SHA-256 hash."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None

def save_document(
    filename: str,
    file_type: str,
    file_hash: str,
    file_size: int,
    chunk_count: int,
    uploaded_by: str
) -> int:
    """Insert document record into SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (filename, file_type, file_hash, file_size, chunk_count, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, file_type, file_hash, file_size, chunk_count, uploaded_by))
        conn.commit()
        return cursor.lastrowid

def update_document_chunk_count(doc_id: int, chunk_count: int) -> None:
    """Update total chunk count for a document."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, doc_id))
        conn.commit()

def list_documents() -> List[Dict[str, Any]]:
    """List all documents with their metadata."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Get document metadata by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_document(doc_id: int) -> bool:
    """Delete document and its chunks from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0

# --- Chunk Management ---

def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Save batch of document chunks into SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO document_chunks (doc_id, chunk_id, chunk_index, content, page_number, section_title)
            VALUES (:doc_id, :chunk_id, :chunk_index, :content, :page_number, :section_title)
        """, chunks)
        conn.commit()

def get_chunks_by_doc_id(doc_id: int) -> List[Dict[str, Any]]:
    """Retrieve all chunks belonging to a document."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
            (doc_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_all_chunks() -> List[Dict[str, Any]]:
    """Retrieve all chunks across all documents."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, d.filename, d.file_type 
            FROM document_chunks c 
            JOIN documents d ON c.doc_id = d.id 
            ORDER BY c.doc_id, c.chunk_index
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_stats() -> Dict[str, Any]:
    """Return summary statistics of knowledge base."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_users": total_users
        }

