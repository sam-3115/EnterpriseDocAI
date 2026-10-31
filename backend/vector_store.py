import os
import hashlib
import math
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_PERSIST_DIR, GEMINI_API_KEY

COLLECTION_NAME = "enterprise_docs"
EMBEDDING_MODEL_NAME = "text-embedding-004"

_chroma_client: Optional[chromadb.PersistentClient] = None

def get_chroma_client() -> chromadb.PersistentClient:
    """Return persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
    return _chroma_client

def get_collection():
    """Retrieve or create the enterprise_docs ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def _pseudo_hash_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Deterministic fallback embedding for offline testing or prior to API key configuration.
    Produces a normalized pseudo-random vector based on sha256 chunks.
    """
    vec = []
    h = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(dim):
        byte_val = h[i % len(h)]
        val = math.sin((i + 1) * (byte_val + 1))
        vec.append(val)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def generate_embeddings(texts: List[str], task_type: str = "retrieval_document", api_key: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings using official Google GenAI API (text-embedding-004).
    Falls back gracefully to deterministic vectors if API key is missing or offline.
    """
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if effective_api_key and effective_api_key.strip():
        # Try modern google.genai SDK
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key.strip())
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=texts,
            )
            # Response contains embeddings list
            if hasattr(response, "embeddings") and response.embeddings:
                return [emb.values for emb in response.embeddings]
        except Exception as e:
            # Try legacy google.generativeai if available
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=effective_api_key.strip())
                res = legacy_genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type=task_type
                )
                if "embedding" in res:
                    embs = res["embedding"]
                    if embs and isinstance(embs[0], (int, float)):
                        return [embs]
                    return embs
            except Exception as e2:
                print(f"[VectorStore] Gemini embedding call failed: {e}. Falling back to deterministic local vectors.")
            
    # Deterministic fallback when no API key or offline
    return [_pseudo_hash_embedding(t) for t in texts]

def add_chunks_to_chroma(chunks: List[Dict[str, Any]], api_key: Optional[str] = None) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    """
    if not chunks:
        return 0

    collection = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    contents = [c["content"] for c in chunks]
    metadatas = [
        {
            "doc_id": int(c["doc_id"]),
            "filename": str(c.get("filename", "")),
            "page_number": int(c.get("page_number", 1)),
            "section_title": str(c.get("section_title", "General")),
            "chunk_index": int(c.get("chunk_index", 0))
        }
        for c in chunks
    ]

    # Generate embeddings
    embeddings = generate_embeddings(contents, task_type="retrieval_document", api_key=api_key)

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=contents,
        metadatas=metadatas
    )
    return len(ids)

def delete_chunks_by_doc_id(doc_id: int) -> None:
    """
    Remove all chunks associated with doc_id from ChromaDB.
    """
    try:
        collection = get_collection()
        collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        print(f"[VectorStore] Error deleting doc_id {doc_id} from ChromaDB: {e}")

def search_vector_store(
    query: str,
    top_k: int = 5,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for most semantically relevant chunks.
    """
    if not query.strip():
        return []

    collection = get_collection()
    query_embeddings = generate_embeddings([query], task_type="retrieval_query", api_key=api_key)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted = []
    if results and "ids" in results and results["ids"]:
        ids = results["ids"][0]
        docs = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            score = max(0.0, 1.0 - dist) if dist is not None else 0.0
            formatted.append({
                "chunk_id": cid,
                "content": doc,
                "metadata": meta,
                "score": round(score, 4)
            })

    return formatted

def get_chroma_stats() -> Dict[str, Any]:
    """Return collection count and status."""
    try:
        collection = get_collection()
        count = collection.count()
        return {"status": "connected", "total_embeddings": count, "collection": COLLECTION_NAME}
    except Exception as e:
        return {"status": "error", "error": str(e), "total_embeddings": 0}
