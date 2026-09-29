# Debug helper: shows where the eBook mentions a topic, and what Pinecone returns for different phrasings.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader

from src import config

KEYWORDS = ["challenge", "limitation", "risk", "barrier", "concern"]

print("=== Where the eBook mentions challenges ===")
for num, page in enumerate(PdfReader(config.PDF_PATH).pages, start=1):
    text = " ".join((page.extract_text() or "").split())
    low = text.lower()
    for kw in KEYWORDS:
        i = low.find(kw)
        if i != -1:
            print(f"p{num} [{kw}]: ...{text[max(0, i - 80):i + 140]}...")
            break

print("\n=== What Pinecone returns for different phrasings ===")
store = PineconeVectorStore(
    index_name=config.PINECONE_INDEX_NAME,
    embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
)
queries = sys.argv[1:] or [
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "challenges, risks, barriers, limitations, concerns",
    "challenges organizations face when implementing and adopting AI agents",
]
for q in queries:
    print(f"\nQ: {q}")
    for doc, score in store.similarity_search_with_score(q, k=8):
        snippet = " ".join(doc.page_content.split())[:90]
        print(f"  p{doc.metadata.get('page'):<3} {score:.3f}  {snippet}")