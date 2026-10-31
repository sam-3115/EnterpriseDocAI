import os
from typing import List, Dict, Any, Optional
from backend.config import GEMINI_API_KEY
from backend.retrieval import hybrid_retrieve

LLM_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are EnterpriseDoc AI, an intelligent, authoritative Enterprise Documentation Assistant.

CRITICAL INSTRUCTIONS:
1. STRICT EVIDENCE GROUNDING: Answer the question using ONLY the retrieved document passages provided in the Context below. Do NOT use external knowledge, speculate, extrapolate, or invent facts.
2. NOT FOUND RESPONSE: If the answer cannot be directly determined from the provided Context passages, you MUST respond with:
"I could not find sufficient information in the provided enterprise documentation to answer this question."
Do NOT attempt to fabricate an answer.
3. CITATIONS: Include precise inline citations in your response using format [Source N] or [Doc: <filename>, Page <page>]. Every factual claim must be backed by a source.
4. CONTRADICTION & OUTDATED INFO DETECTION: Carefully inspect the retrieved passages. If two documents or sections state conflicting or contradictory rules, policies, or dates, you MUST add a dedicated section at the bottom titled:
"⚠️ Contradiction / Conflict Warning:" explaining the conflicting statements and citing the conflicting sources.
5. CONFIDENCE RATING: At the very end of your response, output on a new line:
Confidence: High | Medium | Low | None
- High: Context directly, unambiguously answers the question.
- Medium: Context partially answers the question or requires minor inference from stated facts.
- Low: Context mentions keywords but is missing critical specifics.
- None: Question is unanswerable from the context.
"""

def format_context_passages(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into structured context blocks with metadata."""
    if not retrieved_chunks:
        return "No relevant context passages found in knowledge base."

    blocks = []
    for idx, c in enumerate(retrieved_chunks):
        meta = c.get("metadata", {})
        fname = meta.get("filename", "Unknown Document")
        pnum = meta.get("page_number", 1)
        sec = meta.get("section_title", "General")
        content = c.get("content", "").strip()

        block = f"[Source {idx + 1}]\nDocument: {fname}\nPage: {pnum}\nSection: {sec}\nContent:\n{content}"
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)

def parse_confidence_and_contradiction(answer_text: str) -> Dict[str, Any]:
    """Parse out confidence level and detect contradiction warning from model output."""
    confidence = "Medium"
    has_contradiction = False
    clean_answer = answer_text

    # Detect confidence
    for line in answer_text.splitlines():
        if "confidence:" in line.lower():
            if "high" in line.lower():
                confidence = "High"
            elif "medium" in line.lower():
                confidence = "Medium"
            elif "low" in line.lower():
                confidence = "Low"
            elif "none" in line.lower():
                confidence = "None"

    # Detect contradiction warning
    if "contradiction / conflict warning" in answer_text.lower() or "conflict warning" in answer_text.lower():
        has_contradiction = True

    # If model gave not-found answer, force confidence to None or Low
    if "could not find sufficient information" in answer_text.lower():
        confidence = "None"

    return {
        "confidence": confidence,
        "has_contradiction": has_contradiction
    }

def generate_rag_answer(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute end-to-end RAG pipeline:
    1. Hybrid retrieval (Semantic + BM25 + RRF + Cross-Encoder Reranking)
    2. Context assembly with citation references
    3. Conversational history integration
    4. Gemini Flash generation strictly grounded in evidence
    5. Contradiction & Confidence metadata extraction
    """
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY

    # 1. Retrieve most relevant passages using Hybrid Pipeline
    retrieved_chunks = hybrid_retrieve(query=query, top_k=top_k, api_key=effective_api_key)

    if not retrieved_chunks:
        return {
            "query": query,
            "answer": "I could not find sufficient information in the provided enterprise documentation to answer this question.",
            "confidence": "None",
            "has_contradiction": False,
            "sources": [],
            "retrieved_count": 0
        }

    # Format context and sources list
    context_str = format_context_passages(retrieved_chunks)
    sources = []
    for idx, c in enumerate(retrieved_chunks):
        meta = c.get("metadata", {})
        sources.append({
            "source_id": idx + 1,
            "chunk_id": c.get("chunk_id"),
            "filename": meta.get("filename", "Unknown"),
            "page_number": meta.get("page_number", 1),
            "section_title": meta.get("section_title", "General"),
            "score": c.get("rerank_score", c.get("score", 0.0)),
            "excerpt": c.get("content", "")[:280] + "..." if len(c.get("content", "")) > 280 else c.get("content", "")
        })

    # Build prompt with conversation history for conversational follow-ups
    history_str = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-4:]: # take last 4 turns for context
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_str = "Prior Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    prompt_content = f"""{SYSTEM_PROMPT}

{history_str}CONTEXT PASSAGES:
{context_str}

USER QUESTION:
{query}

ANSWER (Strictly based on above context passages with inline citations):"""

    # 2. Call Gemini Flash LLM
    answer_text = ""
    
    # Check query content-token overlap with retrieved passages
    from backend.retrieval import tokenize_text
    q_toks = set(tokenize_text(query))
    GENERIC_TERMS = {
        "what", "is", "the", "are", "for", "and", "in", "to", "of", "how",
        "many", "can", "employee", "employees", "company", "corporate", "enterprise",
        "about", "with", "from", "at", "by", "this", "that"
    }
    content_q_toks = q_toks - GENERIC_TERMS
    if not content_q_toks:
        content_q_toks = q_toks

    matched_toks = set()
    for c in retrieved_chunks:
        matched_toks.update(set(tokenize_text(c.get("content", ""))))
    overlap = content_q_toks.intersection(matched_toks)

    # If key content terms are missing from documents (unanswerable / out-of-domain)
    if not overlap or (len(overlap) / len(content_q_toks) < 0.25):
        answer_text = (
            "I could not find sufficient information in the provided enterprise documentation to answer this question.\n\n"
            "Confidence: None"
        )
    elif effective_api_key and effective_api_key.strip():
        # Try modern google.genai SDK
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key.strip())
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt_content,
            )
            answer_text = response.text or ""
        except Exception as e:
            # Fallback to legacy google.generativeai if available
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=effective_api_key.strip())
                model = legacy_genai.GenerativeModel(LLM_MODEL)
                res = model.generate_content(prompt_content)
                answer_text = res.text or ""
            except Exception as e2:
                answer_text = f"Error communicating with Gemini Flash API ({e}). Retrieved sources are displayed below."
    else:
        # Inform user to configure Gemini API Key while providing evidence synthesis
        answer_text = (
            "[API Key Note]: Please configure your GEMINI_API_KEY in the sidebar or .env file to generate live Gemini Flash answers.\n\n"
            f"Evidence Retrieved ({len(retrieved_chunks)} relevant passage(s) found):\n"
            f"The knowledge base identified top matches from {sources[0]['filename']} (Section: {sources[0]['section_title']}, Page {sources[0]['page_number']}).\n\n"
            "Confidence: Medium"
        )

    # 3. Parse confidence and contradictions
    parsed_meta = parse_confidence_and_contradiction(answer_text)

    return {
        "query": query,
        "answer": answer_text,
        "confidence": parsed_meta["confidence"],
        "has_contradiction": parsed_meta["has_contradiction"],
        "sources": sources,
        "retrieved_count": len(sources)
    }
