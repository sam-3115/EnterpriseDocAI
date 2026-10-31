import zlib
from pathlib import Path

def create_simple_pdf(output_path: Path):
    """Generate a clean, valid multi-page PDF for testing text extraction and page numbering."""
    content_page1 = (
        "BT\n"
        "/F1 18 Tf\n"
        "50 720 Td\n"
        "(Enterprise Security and Compliance Manual) Tj\n"
        "/F1 12 Tf\n"
        "0 -30 Td\n"
        "(Section 1: Access Control and Identification) Tj\n"
        "0 -20 Td\n"
        "(All personnel must wear identification badges at all times within secure facilities.) Tj\n"
        "0 -20 Td\n"
        "(Access to database clusters requires explicit authorization from security leads.) Tj\n"
        "ET\n"
    ).encode("latin1")

    content_page2 = (
        "BT\n"
        "/F1 14 Tf\n"
        "50 720 Td\n"
        "(Section 2: Incident Response Protocol) Tj\n"
        "/F1 12 Tf\n"
        "0 -25 Td\n"
        "(Security incidents must be reported to soc@enterprisedoc.ai within 15 minutes.) Tj\n"
        "0 -20 Td\n"
        "(The incident response team initiates triage and isolation procedures immediately.) Tj\n"
        "ET\n"
    ).encode("latin1")

    stream1 = content_page1
    stream2 = content_page2

    # Assemble PDF objects
    objs = []
    
    # 1: Catalog
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: Pages
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n")
    # 3: Page 1
    objs.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    # 4: Page 2
    objs.append(b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    # 5: Stream 1
    objs.append(f"5 0 obj\n<< /Length {len(stream1)} >>\nstream\n".encode("latin1") + stream1 + b"\nendstream\nendobj\n")
    # 6: Stream 2
    objs.append(f"6 0 obj\n<< /Length {len(stream2)} >>\nstream\n".encode("latin1") + stream2 + b"\nendstream\nendobj\n")
    # 7: Font
    objs.append(b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Calculate offsets
    pdf_header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    
    current_offset = len(pdf_header)
    for obj in objs:
        offsets.append(current_offset)
        body += obj
        current_offset += len(obj)

    xref_offset = current_offset + len(pdf_header) # offset from start of file
    xref_offset = len(pdf_header) + len(body)
    
    xref = f"xref\n0 {len(objs) + 1}\n".encode("latin1")
    xref += b"0000000000 65535 f \n"
    for o in offsets[1:]:
        xref += f"{o:010d} 00000 n \n".encode("latin1")

    trailer = f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_header + body + xref + trailer)
    print(f"Created sample PDF at {output_path}")

if __name__ == "__main__":
    out = Path("sample_docs/sample_handbook.pdf")
    create_simple_pdf(out)

