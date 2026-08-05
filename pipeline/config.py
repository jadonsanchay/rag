from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf_files"
TEXT_DIR = DATA_DIR / "text_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OPENAI_MODEL = "gpt-4o-mini"
