import os
import re
from typing import List, Dict, Any, Optional
from backend.config import GEMINI_API_KEY
from backend.retrieval import hybrid_retrieve

LLM_MODEL = "gemini-2.5-flash"

CONVERSATIONAL_PROMPT = """You are EnterpriseDoc AI, a polite, helpful Enterprise Documentation Assistant.
The user is sending a conversational greeting, pleasantry, gratitude, self-introduction, or general courtesy message.
Respond in a friendly, professional, and courteous manner (e.g., greet them, welcome them, acknowledge their name or thanks, and politely offer help with any enterprise documentation, policies, or technical guidelines).
Keep it natural, polite, and concise (1-3 sentences).
Do NOT include citations, document references, or confidence ratings for conversational messages."""

SYSTEM_PROMPT = """You are EnterpriseDoc AI, an intelligent, authoritative Enterprise Documentation Assistant.

CRITICAL INSTRUCTIONS:
1. STRICT EVIDENCE GROUNDING: Answer the question using ONLY the retrieved document passages provided in the Context below. Do NOT use external knowledge, speculate, extrapolate, or invent facts.
2. NOT FOUND / UNANSWERABLE RESPONSE: If the question cannot be answered from the provided Context passages (e.g. out-of-domain, non-existent policy, or missing topics), you MUST respond with:
"I could not find sufficient information in the provided enterprise documentation to answer this question."
Do NOT attempt to fabricate an answer or assume company policies not stated.
3. AMBIGUOUS QUESTIONS: If the user's question is vague, incomplete, or ambiguous (e.g., asking about "the policy", "the rules", "how to submit", without specifying which policy, department, or procedure):
- Clearly state what information is ambiguous or missing.
- Polite ask the user for specific context (e.g., "Could you please specify which policy or procedure you are referring to, such as Paid Time Off, Remote Work, or Information Security?").
- You may briefly mention the relevant topics found in the documentation to help them clarify.
4. CITATIONS: Include precise inline citations in your response using format [Source N] or [Doc: <filename>, Page <page>]. Every factual claim must be backed by a source.
5. CONTRADICTION & OUTDATED INFO DETECTION: Carefully inspect the retrieved passages. If two documents or sections state conflicting or contradictory rules, policies, or dates, you MUST add a dedicated section at the bottom titled:
"⚠️ Contradiction / Conflict Warning:" explaining the conflicting statements and citing the conflicting sources.
6. CONFIDENCE RATING: At the very end of your response, output on a new line:
Confidence: High | Medium | Low | None
- High: Context directly, unambiguously answers the question.
- Medium: Context partially answers the question or requires clarification/disambiguation.
- Low: Context mentions keywords but is missing critical specifics.
- None: Question is unanswerable from the context.
"""

def is_conversational_query(query: str) -> bool:
    """
    Determine whether a query is conversational (greeting, small talk, gratitude, introduction, apology)
    and does NOT require RAG retrieval from enterprise documentation.
    """
    q = query.strip().lower()
    
    # Strip common punctuation
    cleaned = re.sub(r'[^\w\s]', '', q).strip()
    words = cleaned.split()
    
    # Quick exact matches or short conversational phrases
    conversational_exact = {
        "hi", "hello", "hey", "hiya", "howdy", "hola",
        "good morning", "good afternoon", "good evening", "good night", "good day",
        "thanks", "thank you", "thank you so much", "thanks a lot", "thx", "ty",
        "sorry", "im sorry", "i am sorry", "my apologies", "apologies",
        "bye", "goodbye", "see you", "see ya", "talk to you later",
        "who are you", "what are you", "what can you do", "help", "help me",
        "how are you", "how are you doing", "hows it going", "how is it going",
        "whats up", "what is up", "sup", "ok", "okay", "sure", "cool", "great", "nice"
    }
    
    if cleaned in conversational_exact:
        return True

    # Check for intro patterns like "my name is...", "i am...", "call me..."
    intro_patterns = [
        r'^(hi|hello|hey)?\s*(my name is|i am|im|this is|call me)\s+[a-zA-Z\s]+$',
        r'^(hi|hello|hey|greetings|dear assistant)[,!\.\s]*$',
        r'^(thank you|thanks|many thanks|much appreciated)[,!\.\s]*(for.*)?$',
        r'^(sorry|im sorry|i apologize)[,!\.\s]*(for.*)?$'
    ]
    for pattern in intro_patterns:
        if re.match(pattern, cleaned):
            return True

    # If query is very short (1-3 words) and made purely of polite/greeting words
    polite_tokens = {
        "hi", "hello", "hey", "there", "good", "morning", "afternoon", "evening",
        "thanks", "thank", "you", "so", "much", "sorry", "please", "welcome",
        "bye", "goodbye", "ok", "okay", "alright", "name", "is", "my", "i", "am", "im"
    }
    if 0 < len(words) <= 3 and all(w in polite_tokens for w in words):
        return True

    return False

def generate_conversational_response(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None
) -> str:
    """Generate a friendly conversational response without invoking document retrieval."""
    cleaned = re.sub(r'[^\w\s]', '', query.strip().lower()).strip()
    
    # Fast deterministic fallbacks for common pleasantries
    name_match = re.search(r'(?:my name is|i am|im|this is)\s+([a-zA-Z]+)', query, re.IGNORECASE)
    user_name = name_match.group(1).capitalize() if name_match else None

    # Try live Gemini Flash if API key available for natural personality
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if effective_api_key and effective_api_key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key.strip())
            prompt = f"{CONVERSATIONAL_PROMPT}\n\nUser message: {query}\nAssistant response:"
            resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            if resp.text and resp.text.strip():
                return resp.text.strip()
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=effective_api_key.strip())
                model = legacy_genai.GenerativeModel(LLM_MODEL)
                res = model.generate_content(f"{CONVERSATIONAL_PROMPT}\n\nUser message: {query}\nAssistant response:")
                if res.text and res.text.strip():
                    return res.text.strip()
            except Exception:
                pass

    # High quality fallback responses
    if user_name:
        return f"Hello {user_name}! Happy to meet you. How can I assist you with enterprise documentation today?"
    if any(w in cleaned for w in ["thank", "thanks", "thx", "ty"]):
        return "You're very welcome! Happy to help. Let me know if you have any other questions regarding company policies or documentation."
    if any(w in cleaned for w in ["sorry", "apologize", "apologies"]):
        return "No worries at all! How can I assist you today?"
    if any(w in cleaned for w in ["bye", "goodbye"]):
        return "Goodbye! Have a great day ahead. Feel free to return whenever you need documentation assistance."
    if any(w in cleaned for w in ["how are you", "hows it going"]):
        return "Hello! I'm doing great, thank you. I'm ready to help you navigate through all enterprise documents and policies. What would you like to know?"
    
    return "Hello! I am your Enterprise Documentation Assistant. Happy to help! Feel free to ask any question about company policies, architecture, APIs, or guidelines."

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

    # 0. Intent routing: Check if query is conversational (greeting/pleasantry/gratitude)
    # If conversational, bypass document retrieval entirely!
    if is_conversational_query(query):
        conv_answer = generate_conversational_response(
            query=query,
            conversation_history=conversation_history,
            api_key=effective_api_key
        )
        return {
            "query": query,
            "answer": conv_answer,
            "confidence": "High",
            "has_contradiction": False,
            "sources": [],
            "retrieved_count": 0,
            "is_conversational": True
        }

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
        "about", "with", "from", "at", "by", "this", "that", "tell", "me", "does",
        "do", "we", "have", "a", "an", "policy", "policies", "procedure", "procedures"
    }
    content_q_toks = q_toks - GENERIC_TERMS
    if not content_q_toks:
        content_q_toks = q_toks

    matched_toks = set()
    for c in retrieved_chunks:
        matched_toks.update(set(tokenize_text(c.get("content", ""))))
    overlap = content_q_toks.intersection(matched_toks)

    # Check for ambiguous query patterns (vague queries lacking a specific subject)
    AMBIGUOUS_PATTERNS = [
        r'^(what is the policy|tell me about the policy|explain the rules|what are the requirements|how do i apply|how to submit|what is the process|what are the guidelines)[\?\.]*$',
        r'^(can i do this|is this allowed|what should i do|who do i contact)[\?\.]*$'
    ]
    is_ambiguous = any(re.match(p, query.strip().lower()) for p in AMBIGUOUS_PATTERNS)

    if is_ambiguous:
        # Prompt user to clarify context
        sample_topics = list({c.get("metadata", {}).get("section_title", "") for c in retrieved_chunks if c.get("metadata", {}).get("section_title")})[:3]
        topic_hint = f" (for example: {', '.join(sample_topics)})" if sample_topics else ""
        answer_text = (
            f"Your question appears broad or ambiguous. Could you please specify which topic or policy you are referring to{topic_hint}? "
            "Please provide additional details so I can give you an accurate, evidence-backed answer.\n\n"
            "Confidence: Medium"
        )
    # If key content terms are completely missing from documents (unanswerable / out-of-domain)
    elif not overlap or (len(overlap) / len(content_q_toks) < 0.20):
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

    # 4. Strict Citation Filtering: Only keep sources that were actually used/cited in the answer
    cited_ids = set()
    # Match patterns like [Source 1], [Source 2], Source 1, [1], etc.
    for match in re.finditer(r'\[Source\s+(\d+)\]', answer_text, re.IGNORECASE):
        cited_ids.add(int(match.group(1)))
    for match in re.finditer(r'(?:^|[^\w])Source\s+(\d+)(?:[^\w]|$)', answer_text, re.IGNORECASE):
        cited_ids.add(int(match.group(1)))
    for match in re.finditer(r'\[(\d+)\]', answer_text):
        cited_ids.add(int(match.group(1)))

    # Also match explicit mentions of document filenames in the generated text
    cited_filenames = set()
    for s in sources:
        fname = s.get("filename", "")
        if fname and fname.lower() in answer_text.lower():
            cited_filenames.add(fname.lower())
    for match in re.finditer(r'\[Doc:\s*([^,\]]+)', answer_text, re.IGNORECASE):
        cited_filenames.add(match.group(1).strip().lower())

    if cited_ids or cited_filenames:
        cited_sources = [
            s for s in sources
            if s["source_id"] in cited_ids
            or s.get("filename", "").lower() in cited_filenames
        ]
        if cited_sources:
            sources = cited_sources
    else:
        # If the answer is unanswerable or no sources were cited, show NO sources
        # (Never dump all retrieved chunks if none was referenced)
        if (
            parsed_meta["confidence"] == "None" 
            or "could not find sufficient information" in answer_text.lower()
            or not cited_ids
        ):
            # If the model produced an answer without formal citations, keep only top 1 relevant source as reference
            if "could not find sufficient information" in answer_text.lower() or parsed_meta["confidence"] == "None":
                sources = []
            else:
                sources = sources[:1] if sources else []

    return {
        "query": query,
        "answer": answer_text,
        "confidence": parsed_meta["confidence"],
        "has_contradiction": parsed_meta["has_contradiction"],
        "sources": sources,
        "retrieved_count": len(sources)
    }
