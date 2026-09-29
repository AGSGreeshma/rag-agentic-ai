import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MIN_PAGE_CHARS = 100          
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
GDRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

# Chunking / retrieval
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5


def validate():
    missing = [k for k, v in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}. Check your .env file.")