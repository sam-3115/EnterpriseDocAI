import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app
from backend.retrieval import get_bm25_retriever, reciprocal_rank_fusion, cross_encoder_rerank, hybrid_retrieve
from backend.rag_engine import generate_rag_answer
from backend.evaluation import run_evaluation_benchmark

def run_part2_and_3_tests():
    print("=" * 65)
    print("RUNNING ENTERPRISEDOC AI - PART 2 & 3 END-TO-END VERIFICATION")
    print("=" * 65)

    client = TestClient(app)

    # 0. Ensure sample documents are indexed for evaluation and retrieval
    sample_dir = PROJECT_ROOT / "sample_docs"
    with open(sample_dir / "sample_policy.txt", "rb") as f:
        client.post(
            "/api/documents/upload",
            files={"file": ("sample_policy.txt", f.read(), "text/plain")},
            data={"uploaded_by": "admin", "role": "admin"}
        )
    with open(sample_dir / "sample_guide.md", "rb") as f:
        client.post(
            "/api/documents/upload",
            files={"file": ("sample_guide.md", f.read(), "text/markdown")},
            data={"uploaded_by": "admin", "role": "admin"}
        )

    # 1. Test BM25 Retrieval
    print("\n[TEST 1] Testing BM25 Keyword Retrieval...")
    bm25 = get_bm25_retriever()
    bm25.refresh()
    bm25_results = bm25.search("paid time off leave request", top_k=3)
    assert len(bm25_results) > 0, "BM25 returned no results for relevant keywords"
    print(f"  --> PASS: BM25 retrieved {len(bm25_results)} keyword-matched chunks.")
    print(f"      Top Match: {bm25_results[0]['metadata']['filename']} (Score: {bm25_results[0]['score']:.2f})")

    # 2. Test Reciprocal Rank Fusion (RRF)
    print("\n[TEST 2] Testing Reciprocal Rank Fusion (RRF)...")
    mock_semantic = [
        {"chunk_id": "c1", "content": "Paid time off and leave policies", "score": 0.85, "metadata": {"filename": "policy.txt"}},
        {"chunk_id": "c2", "content": "Security protocols and authentication", "score": 0.72, "metadata": {"filename": "security.txt"}}
    ]
    mock_keyword = [
        {"chunk_id": "c2", "content": "Security protocols and authentication", "score": 4.5, "metadata": {"filename": "security.txt"}},
        {"chunk_id": "c1", "content": "Paid time off and leave policies", "score": 3.2, "metadata": {"filename": "policy.txt"}}
    ]
    fused = reciprocal_rank_fusion(mock_semantic, mock_keyword, k=60)
    assert len(fused) == 2, "RRF failed to fuse items"
    assert "rrf_score" in fused[0], "Missing rrf_score in fused item"
    print(f"  --> PASS: RRF merged ranks successfully. Top RRF score: {fused[0]['rrf_score']}")

    # 3. Test Cross-Encoder / Contextual Reranking
    print("\n[TEST 3] Testing Cross-Encoder Contextual Reranker...")
    reranked = cross_encoder_rerank(
        query="paid time off",
        candidate_chunks=fused,
        top_k=2
    )
    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    # c1 should be reranked to top because of exact token coverage for 'paid time off'
    assert reranked[0]["chunk_id"] == "c1"
    print("  --> PASS: Cross-encoder reranked exact query-document alignment to rank #1.")

    # 4. Test Hybrid Retrieval Pipeline
    print("\n[TEST 4] Testing Full Hybrid Retrieval Pipeline...")
    candidates = hybrid_retrieve(query="remote work core hours", top_k=3)
    assert len(candidates) > 0, "Hybrid retrieval returned no candidates"
    top_cand = candidates[0]
    print(f"  --> PASS: Hybrid retrieval returned {len(candidates)} candidates.")
    print(f"      Top Document: {top_cand['metadata']['filename']} (Rerank Score: {top_cand.get('rerank_score', 0)})")

    # 5. Test RAG Answering with Strict Grounding & Citation Handling
    print("\n[TEST 5] Testing RAG Engine Grounded Answer & Citations...")
    rag_resp = generate_rag_answer(
        query="How many days of paid time off do employees get?",
        top_k=3
    )
    assert "answer" in rag_resp
    assert "sources" in rag_resp
    assert len(rag_resp["sources"]) > 0
    top_source = rag_resp["sources"][0]
    assert top_source["filename"] != ""
    assert "page_number" in top_source
    assert "section_title" in top_source
    print(f"  --> PASS: RAG generated answer with {len(rag_resp['sources'])} structured source citations.")
    print(f"      Source: {top_source['filename']}, Page {top_source['page_number']}, Section: {top_source['section_title']}")

    # 6. Test Unanswerable Query Detection (Strict Grounding & None Confidence)
    print("\n[TEST 6] Testing Unanswerable Query Handling...")
    unanswerable_resp = generate_rag_answer(
        query="What is the employee stock discount rate for purchasing company shares?",
        top_k=3
    )
    assert "could not find sufficient information" in unanswerable_resp["answer"].lower() or unanswerable_resp["confidence"] in ["None", "Low"]
    print("  --> PASS: Assistant correctly flagged unanswerable question without hallucination.")

    # 7. Test Conversational Follow-up Support
    print("\n[TEST 7] Testing Conversational Follow-up Support...")
    history = [
        {"role": "user", "content": "What is the policy on paid time off?"},
        {"role": "assistant", "content": "Employees get 20 days of paid time off per calendar year."}
    ]
    follow_up_resp = client.post("/api/chat", json={
        "query": "Can unused days be carried forward?",
        "conversation_history": history,
        "top_k": 3
    })
    assert follow_up_resp.status_code == 200, f"Chat endpoint failed: {follow_up_resp.text}"
    chat_data = follow_up_resp.json()
    assert "answer" in chat_data
    assert len(chat_data["sources"]) > 0
    print("  --> PASS: Conversational follow-up processed with dialogue history context.")

    # 8. Test Part 3 Evaluation Benchmark (/api/evaluate)
    print("\n[TEST 8] Running Benchmark Suite (/api/evaluate)...")
    eval_resp = client.get("/api/evaluate?top_k=3")
    assert eval_resp.status_code == 200, f"Evaluation benchmark failed: {eval_resp.text}"
    benchmark = eval_resp.json()
    summary = benchmark["summary"]
    cases = benchmark["test_results"]
    
    assert summary["total_test_cases"] == 5, "Expected 5 evaluation test categories"
    assert summary["mean_recall_at_k"] > 0.0, "Recall@K must be positive"
    assert summary["mean_faithfulness"] > 0.0, "Faithfulness must be positive"

    print("  --> PASS: Evaluation benchmark executed across all 5 categories:")
    print(f"      • Mean Recall@3: {summary['mean_recall_at_k'] * 100:.1f}%")
    print(f"      • Mean Precision@3: {summary['mean_precision_at_k'] * 100:.1f}%")
    print(f"      • Mean MRR: {summary['mean_mrr']:.3f}")
    print(f"      • Answer Faithfulness: {summary['mean_faithfulness'] * 100:.1f}%")
    print(f"      • Citation Correctness: {summary['mean_citation_correctness'] * 100:.1f}%")

    print("\n" + "=" * 65)
    print("ALL PART 2 & PART 3 TESTS PASSED! FULL APPLICATION READY.")
    print("=" * 65)

if __name__ == "__main__":
    run_part2_and_3_tests()
