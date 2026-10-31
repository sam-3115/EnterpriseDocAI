import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    # Normalize unicode characters and replace multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove control characters except standard whitespace
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (32 <= ord(ch) <= 126) or ord(ch) > 127)
    return text.strip()

def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page by page.
    Preserves page numbers and identifies candidate section headings.
    """
    reader = PdfReader(str(file_path))
    segments = []
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
            
        # Detect section title if first non-empty line looks like a header
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        section_title = f"Page {page_num}"
        if lines:
            first_line = lines[0]
            if len(first_line) < 80 and not first_line.endswith("."):
                section_title = first_line
                
        segments.append({
            "text": cleaned,
            "page_number": page_num,
            "section_title": section_title
        })
        
    return segments

def parse_html(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract and clean HTML content.
    Tracks heading hierarchy (H1, H2, H3) as section metadata.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script, style, nav, and other noisy non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.decompose()

    segments = []
    current_section = "Overview"
    current_buffer = []
    
    # Process body tags or root tags
    body = soup.body or soup
    for elem in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"]):
        tag_name = elem.name.lower()
        elem_text = clean_text(elem.get_text(separator=" ", strip=True))
        
        if not elem_text:
            continue
            
        if tag_name in ["h1", "h2", "h3", "h4"]:
            if current_buffer:
                full_text = clean_text("\n".join(current_buffer))
                if full_text:
                    segments.append({
                        "text": full_text,
                        "page_number": 1,
                        "section_title": current_section
                    })
                current_buffer = []
            current_section = elem_text
        elif tag_name in ["p", "li"]:
            current_buffer.append(elem_text)
        elif tag_name == "div" and not elem.find(["p", "h1", "h2", "h3", "li"]):
            current_buffer.append(elem_text)
            
    if current_buffer:
        full_text = clean_text("\n".join(current_buffer))
        if full_text:
            segments.append({
                "text": full_text,
                "page_number": 1,
                "section_title": current_section
            })
            
    if not segments:
        raw_text = clean_text(soup.get_text(separator="\n", strip=True))
        if raw_text:
            segments.append({
                "text": raw_text,
                "page_number": 1,
                "section_title": "Document Content"
            })
            
    return segments

def parse_markdown(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract Markdown content while recognizing header (#, ##, ###) sections.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    segments = []
    current_section = "Introduction"
    current_lines = []
    
    for line in lines:
        stripped = line.strip()
        header_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if header_match:
            if current_lines:
                section_text = clean_text("".join(current_lines))
                if section_text:
                    segments.append({
                        "text": section_text,
                        "page_number": 1,
                        "section_title": current_section
                    })
                current_lines = []
            current_section = header_match.group(2).strip()
        else:
            current_lines.append(line)
            
    if current_lines:
        section_text = clean_text("".join(current_lines))
        if section_text:
            segments.append({
                "text": section_text,
                "page_number": 1,
                "section_title": current_section
            })
            
    return segments

def parse_txt(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract Plain Text content.
    Splits into logical sections based on double newlines and header markers.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_content = f.read()

    cleaned = clean_text(raw_content)
    if not cleaned:
        return []
        
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    segments = []
    current_section = "General"
    buffer = []
    
    for p in paragraphs:
        # Detect simple header line: short line without trailing period
        if len(p) < 60 and ("\n" not in p) and not p.endswith("."):
            if buffer:
                segments.append({
                    "text": clean_text("\n\n".join(buffer)),
                    "page_number": 1,
                    "section_title": current_section
                })
                buffer = []
            current_section = p
        else:
            buffer.append(p)
            
    if buffer:
        segments.append({
            "text": clean_text("\n\n".join(buffer)),
            "page_number": 1,
            "section_title": current_section
        })
        
    return segments

def parse_document(file_path: Path, file_type: str) -> List[Dict[str, Any]]:
    """
    Dispatch parser according to file type: PDF, HTML, Markdown, or TXT.
    """
    ft = file_type.lower().replace(".", "")
    if ft == "pdf":
        return parse_pdf(file_path)
    elif ft in ["html", "htm"]:
        return parse_html(file_path)
    elif ft in ["md", "markdown"]:
        return parse_markdown(file_path)
    elif ft in ["txt", "text"]:
        return parse_txt(file_path)
    else:
        # Fallback to TXT parser
        return parse_txt(file_path)

