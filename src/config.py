"""Central configuration: environment variables, paths, model names and tuning constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
GDRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

# --- API keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# --- Pinecone serverless (the free tier supports aws / us-east-1) ---
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

# --- Ingestion / chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 100          # skip cover/title/divider pages

# --- Retrieval ---
TOP_K = 5

# --- Graph tuning ---
MIN_SIMILARITY = 0.25          # chunks below this cosine similarity are ignored
GROUNDING_THRESHOLD = 0.7      # min grounding score to accept an answer
MAX_GENERATION_ATTEMPTS = 2    # generate once, retry once if not grounded
RETRIEVAL_WEIGHT = 0.4         # confidence = 0.4*retrieval + 0.6*grounding


def validate():
    """Fail early with a clear message if required keys are missing."""
    missing = [k for k, v in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}. Check your .env file.")