# Lists which PDF pages mention given keywords (used to debug retrieval).
import sys
from pathlib import Path
from pypdf import PdfReader

PDF = Path(__file__).resolve().parent.parent / "data" / "Ebook-Agentic-AI.pdf"
words = [w.lower() for w in (sys.argv[1:] or ["challenge", "limitation", "risk", "barrier", "concern"])]
for num, page in enumerate(PdfReader(PDF).pages, start=1):
    text = (page.extract_text() or "").lower()
    hits = {w: text.count(w) for w in words if w in text}
    if hits:
        print(f"page {num}: {hits}")