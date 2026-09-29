# One-off sanity check: confirms the eBook has extractable text (not scanned images).
from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "Ebook-Agentic-AI.pdf"
r = PdfReader(PDF_PATH)
print("Pages:", len(r.pages))
lens = [len((p.extract_text() or "").strip()) for p in r.pages]
print("Chars per page:", lens)
print("Empty pages:", sum(1 for l in lens if l < 20))
print("--- Sample from page 3 ---")
print(r.pages[2].extract_text()[:500])
