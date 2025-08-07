from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Load_And_Dwnld_Embed:

    def __init__(self, embed_path):
        self.embed_path = embed_path
        self.EMBEDDING_MODEL_CACHE_DIR = self.embed_path / "embedding_model"
        self.EMBEDDING_MODEL_CACHE_DIR.mkdir(exist_ok=True)
    
    def load_embed(self):
        logging.info("Embedding are already cached, loading them!")
        self.embeddings = HuggingFaceEmbeddings(model_name=str(self.EMBEDDING_MODEL_CACHE_DIR))
        return self.embeddings
    
    def dwnld_and_cache_embed(self, embed_model):
        logging.info("No Cached embeddings found. Downloading and caching them")
        self.model = SentenceTransformer(embed_model)
        self.model.save(str(self.EMBEDDING_MODEL_CACHE_DIR))
        self.embeddings = HuggingFaceEmbeddings(model_name=str(self.EMBEDDING_MODEL_CACHE_DIR))
        return self.embeddings