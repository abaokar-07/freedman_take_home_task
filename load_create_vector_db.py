from langchain.vectorstores import FAISS
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Load_Create_Vector_DB:

    def __init__(self, cache_path, db_name, embed):
        self.cache_path = cache_path
        self.db_name = db_name
        self.embed = embed
    
    def load_db(self):
        logging.info("Found the vector database. Loading it!")
        self.vector_store = FAISS.load_local(
            folder_path=str(self.cache_path),
            index_name=self.db_name,
            embeddings=self.embed,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store
    
    def create_db(self, docs):
        logging.info("No Caching of the database detected. Creating and then caching it")
        self.vector_store = FAISS.from_documents(docs, self.embed)
        self.vector_store.save_local(str(self.cache_path), index_name=self.db_name)
        return self.vector_store