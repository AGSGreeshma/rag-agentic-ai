"""
Ingestion pipeline: PDF -> cleaned pages -> chunks -> OpenAI embeddings -> Pinecone.

Run from the project root:
    python -m src.ingestion           # create index if needed, then upsert chunks
    python -m src.ingestion --reset   # delete the index first, then re-ingest
"""
import argparse
import re
import time

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)     # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse large blank gaps
    return text.strip()


def load_pages(pdf_path) -> list[Document]:
    """One Document per page, cleaned, with near-empty pages removed."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if len(text) < config.MIN_PAGE_CHARS:
            continue  # cover, title and section-divider pages
        pages.append(Document(
            page_content=text,
            metadata={"source": pdf_path.name, "page": page_num},
        ))
    print(f"Loaded {len(reader.pages)} pages, kept {len(pages)} with real text")
    return pages
def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"Split into {len(chunks)} chunks (avg {avg} chars)")
    return chunks


def get_or_create_index(pc: Pinecone, reset: bool):
    name = config.PINECONE_INDEX_NAME
    existing = pc.list_indexes().names()

    if name in existing and reset:
        print(f"Deleting existing index '{name}'...")
        pc.delete_index(name)
        existing = []

    if name not in existing:
        print(f"Creating index '{name}' (dim={config.EMBEDDING_DIM}, cosine)...")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    else:
        dim = pc.describe_index(name).dimension
        if dim != config.EMBEDDING_DIM:
            raise ValueError(
                f"Index '{name}' has dimension {dim}, expected {config.EMBEDDING_DIM}. "
                "Run with --reset to recreate it."
            )
    return pc.Index(name)
def run_ingestion(reset: bool = False) -> None:
    config.validate()

    pages = load_pages(config.PDF_PATH)
    chunks = split_pages(pages)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = get_or_create_index(pc, reset)

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Deterministic IDs: re-running overwrites the same vectors instead of duplicating them
    ids = [f"p{c.metadata['page']}-c{c.metadata['chunk_id']}" for c in chunks]
    print(f"Embedding and upserting {len(chunks)} chunks...")
    vector_store.add_documents(chunks, ids=ids)

    time.sleep(5)  # Pinecone stats update a few seconds after writes
    total = index.describe_index_stats().total_vector_count
    print(f"Done. Index '{config.PINECONE_INDEX_NAME}' holds {total} vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index first")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)