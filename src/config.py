import os
import getpass
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or getpass.getpass("Groq API Key: ")
    FAST_API_URL = os.getenv("API_URL") or getpass.getpass("FastAPI URL: ")
    INDEX_NAME = "invoice-analysis"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "llama3-70b-8192"
    LLM_TEMPERATURE = 0.3
    SEARCH_CONFIG = {"k": 1, "score_threshold": 0.5}
    VECTOR_STORE_DIR = "./vectorDB"
    DB_NAME = "invoice_analysis_report"

