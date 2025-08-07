from pathlib import Path
import torch

class Config:

    def __init__(self):
        self.PDF_PATH = Path("machine_learning.pdf")
        self.CACHE_DIR = Path("cache/cache")

        self.DOCS_PICKLE_PATH = self.CACHE_DIR / "documents.pkl"
        self.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.LLM_MODEL_NAME = "google/flan-t5-large"
        # self.LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
        self.VECTOR_DB_NAME = "vectorstore"
        self.VECTOR_DB_PATH = self.CACHE_DIR / f"{self.VECTOR_DB_NAME}.faiss"
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_content_tokens = 600
        self.MODEL_FOLDER_NAME = "google_flan_t5_large"
        # self.MODEL_FOLDER_NAME = "mistral_model"