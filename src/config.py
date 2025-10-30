"""
Konfigürasyon ayarları
"""
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Model ayarları
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

# Chroma DB ayarları
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "elasticsearch_docs"

# Döküman ayarları
DOCS_PATH = "./data/elasticsearch_docs.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval ayarları
TOP_K_DOCS = 3

# Web search ayarları
MAX_SEARCH_RESULTS = 3

# Debug modu
DEBUG = True