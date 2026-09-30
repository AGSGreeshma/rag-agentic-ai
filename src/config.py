"""Central configuration: environment variables, paths, model names and tuning constants."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Secrets: .env locally, Streamlit Secrets when deployed ---
# .env is gitignored, so it does not exist on Streamlit Community Cloud; there the keys come from
# the app's Settings -> Secrets panel instead. Copy anything found there into os.environ *before*
# the constants below are read: langchain-openai and langchain-pinecone build their own clients
# straight from the environment, so setting only the module-level names would not be enough.
SECRET_KEYS = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
               "RAG_API_URL", "RAG_API_MODE")


def _load_streamlit_secrets() -> None:
    """Fill in missing env vars from st.secrets. A no-op outside a Streamlit process.

    Only consulted when streamlit is already imported - under uvicorn or pytest it never is, and
    importing it there would cost a second of startup to read secrets that .env already supplied.
    Real env vars and .env win over st.secrets, so a local override still takes effect.
    """
    st = sys.modules.get("streamlit")
    if st is None:
        return
    for key in SECRET_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = st.secrets[key]
        except Exception:
            continue  # no secrets.toml configured, or this key is not in it
        if value:
            os.environ[key] = str(value)


_load_streamlit_secrets()

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
RETRIEVAL_WEIGHT = 0.4        # confidence = 0.4*retrieval + 0.6*grounding
MIN_RELEVANT_CHUNKS = 2        # fewer than this -> rewrite the query and retrieve again
MAX_QUERY_REWRITES = 1         # at most one rewrite per question


def validate():
    """Fail early with a clear message if required keys are missing."""
    missing = [k for k, v in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing env vars: {', '.join(missing)}. Set them in .env locally, or in "
            "Streamlit Cloud under Settings -> Secrets when deployed."
        )