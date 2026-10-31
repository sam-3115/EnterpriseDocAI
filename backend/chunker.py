import re
from typing import List, Dict, Any

def split_text_into_chunks(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Split text into chunks with overlap, respecting sentence and paragraph boundaries.
    """
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Try to find paragraph break near the end
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + (chunk_size // 3):
            # Try sentence break (. followed by space or newline)
            sentence_matches = list(re.finditer(r"\.\s+|\?\s+|\!\s+|\n", text[start:end]))
            if sentence_matches:
                last_match = sentence_matches[-1]
                boundary = start + last_match.end()
            else:
                # Try word break
                space_boundary = text.rfind(" ", start, end)
                if space_boundary > start + (chunk_size // 3):
                    boundary = space_boundary
                else:
                    boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start with overlap
        start = max(start + 1, boundary - chunk_overlap)

    return chunks

def contextual_chunk_document(
    doc_id: int,
    filename: str,
    parsed_segments: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Produce contextual chunks preserving document name, page number, and section metadata.
    Each chunk is contextualized with a header to improve semantic retrieval accuracy.
    """
    chunks = []
    chunk_idx = 0

    for segment in parsed_segments:
        seg_text = segment.get("text", "").strip()
        page_num = segment.get("page_number", 1)
        section = segment.get("section_title", "General").strip()

        if not seg_text:
            continue

        raw_chunks = split_text_into_chunks(seg_text, chunk_size, chunk_overlap)

        for raw_c in raw_chunks:
            # Contextual header embedded into chunk content for retrieval fidelity
            contextual_header = f"[{filename} | Page {page_num} | Section: {section}]\n"
            content_with_context = f"{contextual_header}{raw_c}"

            chunk_id = f"doc_{doc_id}_c{chunk_idx}"
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "content": content_with_context,
                "raw_content": raw_c,
                "page_number": page_num,
                "section_title": section,
                "filename": filename
            })
            chunk_idx += 1

    return chunks

