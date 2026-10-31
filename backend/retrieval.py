import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from backend.database import get_all_chunks
from backend.vector_store import search_vector_store

def tokenize_text(text: str) -> List[str]:
    """Tokenize and lower-case text for BM25 keyword matching."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if len(t) > 1]

class BM25Retriever:
    """Keyword-based retriever using BM25Okapi over SQLite document chunks."""
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._corpus_tokenized: List[List[str]] = []
        self.refresh()

    def refresh(self):
        """Reload chunks from SQLite database and index with BM25."""
        self.chunks = get_all_chunks()
        if self.chunks:
            self._corpus_tokenized = [
                tokenize_text(c["content"]) for c in self.chunks
            ]
            self.bm25 = BM25Okapi(self._corpus_tokenized)
        else:
            self.bm25 = None
            self._corpus_tokenized = []

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search chunks using BM25 scores."""
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs = list(enumerate(scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_pairs[:top_k]:
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": {
                    "doc_id": chunk["doc_id"],
                    "filename": chunk.get("filename", ""),
                    "page_number": chunk.get("page_number", 1),
                    "section_title": chunk.get("section_title", "General"),
                    "chunk_index": chunk.get("chunk_index", 0)
                },
                "score": float(score),
                "retriever": "bm25"
            })
        return results

# Singleton retriever instance
_bm25_retriever: Optional[BM25Retriever] = None

def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever

def reciprocal_rank_fusion(
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine semantic and keyword results using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum(1 / (k + rank(d)))
    """
    rrf_scores: Dict[str, float] = {}
    items_by_id: Dict[str, Dict[str, Any]] = {}

    # 1. Score semantic results
    for rank, item in enumerate(semantic_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
        if cid not in items_by_id:
            items_by_id[cid] = item

    # 2. Score keyword (BM25) results
    for rank, item in enumerate(keyword_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
        if cid not in items_by_id:
            items_by_id[cid] = item

    # 3. Sort by fused RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_results = []
    for cid in sorted_ids:
        chunk_obj = dict(items_by_id[cid])
        chunk_obj["rrf_score"] = round(rrf_scores[cid], 5)
        chunk_obj["score"] = chunk_obj["rrf_score"]
        fused_results.append(chunk_obj)

    return fused_results

def cross_encoder_rerank(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Cross-Encoder / Contextual Reranker:
    Scores candidates by combining semantic alignment, term coverage ratio,
    and exact phrase matching within the passage.
    """
    if not candidate_chunks:
        return []

    q_tokens = set(tokenize_text(query))
    if not q_tokens:
        return candidate_chunks[:top_k]

    q_clean = query.lower().strip()
    reranked = []

    for item in candidate_chunks:
        content = item.get("content", "").lower()
        
        # 1. Exact phrase match bonus
        exact_match_score = 0.35 if q_clean in content else 0.0

        # 2. Query token coverage ratio in document
        c_tokens = set(tokenize_text(content))
        matching_tokens = q_tokens.intersection(c_tokens)
        coverage_ratio = len(matching_tokens) / len(q_tokens) if q_tokens else 0.0

        # 3. Existing RRF / semantic base score
        base_score = float(item.get("rrf_score", item.get("score", 0.0)))

        # Composite rerank score
        rerank_score = (0.45 * base_score) + (0.35 * coverage_ratio) + exact_match_score

        item_copy = dict(item)
        item_copy["rerank_score"] = round(rerank_score, 4)
        item_copy["token_coverage"] = round(coverage_ratio, 2)
        reranked.append(item_copy)

    # Sort descending by rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:
    1. Semantic retrieval with ChromaDB (vector)
    2. Keyword retrieval with BM25 (sparse)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-scoring reranker
    """
    # 1. Semantic retrieval
    semantic_results = search_vector_store(query=query, top_k=top_k * 2, api_key=api_key)

    # 2. BM25 keyword retrieval
    bm25 = get_bm25_retriever()
    bm25.refresh()
    keyword_results = bm25.search(query=query, top_k=top_k * 2)

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(semantic_results, keyword_results, k=60)

    # 4. Cross-Encoder Reranking
    final_candidates = cross_encoder_rerank(query=query, candidate_chunks=fused, top_k=top_k)

    return final_candidates

