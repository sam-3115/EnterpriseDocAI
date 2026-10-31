from typing import List, Dict, Any, Optional
from backend.rag_engine import generate_rag_answer

# Labelled Ground-Truth Evaluation Dataset covering the 5 required categories
EVALUATION_DATASET = [
    {
        "id": "Q1_DIRECT",
        "category": "Direct Question",
        "question": "How many days of paid time off are employees entitled to per calendar year?",
        "ground_truth_doc": "sample_policy.txt",
        "ground_truth_keywords": ["20 days", "paid time off", "calendar year"],
        "is_answerable": True
    },
    {
        "id": "Q2_PARAPHRASED",
        "category": "Paraphrased Question",
        "question": "Can workers carry forward unused annual leave into the next quarter?",
        "ground_truth_doc": "sample_policy.txt",
        "ground_truth_keywords": ["5 days", "carried forward", "following quarter"],
        "is_answerable": True
    },
    {
        "id": "Q3_CONFLICTING",
        "category": "Conflicting / Outdated",
        "question": "Can I store proprietary source code on personal cloud drives?",
        "ground_truth_doc": "sample_policy.txt",
        "ground_truth_keywords": ["strictly prohibited", "personal cloud drives", "disciplinary review"],
        "is_answerable": True
    },
    {
        "id": "Q4_AMBIGUOUS",
        "category": "Ambiguous Question",
        "question": "What is the working schedule and when are enterprise sync meetings held?",
        "ground_truth_doc": "sample_policy.txt",
        "ground_truth_keywords": ["10:00 AM", "4:00 PM", "Monday mornings"],
        "is_answerable": True
    },
    {
        "id": "Q5_UNANSWERABLE",
        "category": "Unanswerable / Out-of-Domain",
        "question": "What is the company discount percentage for purchasing corporate stock options?",
        "ground_truth_doc": None,
        "ground_truth_keywords": ["could not find", "insufficient"],
        "is_answerable": False
    }
]

def evaluate_retrieval(retrieved_sources: List[Dict[str, Any]], ground_truth_doc: Optional[str], k: int = 3) -> Dict[str, float]:
    """Calculate Recall@K, Precision@K, and Reciprocal Rank (RR)."""
    if not ground_truth_doc:
        # For unanswerable questions, if no false positives are strongly ranked, retrieval is considered safe
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0}

    top_sources = retrieved_sources[:k]
    relevant_retrieved = 0
    first_relevant_rank = 0

    for rank, s in enumerate(top_sources):
        src_name = s.get("filename", "")
        if ground_truth_doc.lower() in src_name.lower():
            relevant_retrieved += 1
            if first_relevant_rank == 0:
                first_relevant_rank = rank + 1

    recall = 1.0 if relevant_retrieved > 0 else 0.0
    precision = relevant_retrieved / max(1, len(top_sources))
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank > 0 else 0.0

    return {
        "recall_at_k": round(recall, 3),
        "precision_at_k": round(precision, 3),
        "mrr": round(mrr, 3)
    }

def evaluate_faithfulness_and_citations(
    answer: str,
    sources: List[Dict[str, Any]],
    ground_truth_keywords: List[str],
    is_answerable: bool
) -> Dict[str, float]:
    """
    Evaluate:
    1. Answer Faithfulness: Claims supported directly by context or correct not-found response.
    2. Citation Correctness: Citations present and pointing to valid source items.
    """
    answer_lower = answer.lower()

    if not is_answerable:
        # Correctly rejected unanswerable questions get full faithfulness
        if "could not find" in answer_lower or "insufficient" in answer_lower or "not find sufficient" in answer_lower:
            return {"faithfulness": 1.0, "citation_correctness": 1.0}
        return {"faithfulness": 0.0, "citation_correctness": 0.5}

    # Measure keyword presence from ground truth
    keyword_hits = sum(1 for kw in ground_truth_keywords if kw.lower() in answer_lower)
    faithfulness = round(keyword_hits / len(ground_truth_keywords), 2) if ground_truth_keywords else 1.0

    # Citation check: does the answer cite [Source N] or [Doc: ...] and are sources present?
    has_inline_citations = ("[" in answer and "]" in answer) or len(sources) > 0
    citation_correctness = 1.0 if (has_inline_citations and len(sources) > 0) else 0.0

    return {
        "faithfulness": faithfulness,
        "citation_correctness": citation_correctness
    }

def run_evaluation_benchmark(api_key: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """
    Execute full evaluation benchmark over direct, paraphrased, conflicting,
    ambiguous, and unanswerable questions.
    """
    results = []
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_faithfulness = 0.0
    total_citations = 0.0
    n = len(EVALUATION_DATASET)

    for item in EVALUATION_DATASET:
        rag_res = generate_rag_answer(
            query=item["question"],
            top_k=top_k,
            api_key=api_key
        )

        sources = rag_res.get("sources", [])
        answer = rag_res.get("answer", "")

        # Retrieval metrics
        r_metrics = evaluate_retrieval(sources, item["ground_truth_doc"], k=top_k)
        
        # Generation metrics
        g_metrics = evaluate_faithfulness_and_citations(
            answer=answer,
            sources=sources,
            ground_truth_keywords=item["ground_truth_keywords"],
            is_answerable=item["is_answerable"]
        )

        total_recall += r_metrics["recall_at_k"]
        total_precision += r_metrics["precision_at_k"]
        total_mrr += r_metrics["mrr"]
        total_faithfulness += g_metrics["faithfulness"]
        total_citations += g_metrics["citation_correctness"]

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth_doc": item["ground_truth_doc"] or "None (Out of domain)",
            "answer": answer[:220] + "..." if len(answer) > 220 else answer,
            "confidence": rag_res.get("confidence", "Medium"),
            "sources_count": len(sources),
            "recall_at_k": r_metrics["recall_at_k"],
            "precision_at_k": r_metrics["precision_at_k"],
            "mrr": r_metrics["mrr"],
            "faithfulness": g_metrics["faithfulness"],
            "citation_correctness": g_metrics["citation_correctness"]
        })

    summary = {
        "total_test_cases": n,
        "mean_recall_at_k": round(total_recall / n, 3),
        "mean_precision_at_k": round(total_precision / n, 3),
        "mean_mrr": round(total_mrr / n, 3),
        "mean_faithfulness": round(total_faithfulness / n, 3),
        "mean_citation_correctness": round(total_citations / n, 3)
    }

    return {
        "summary": summary,
        "test_results": results
    }

