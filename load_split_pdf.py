from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Load_And_Split_PDF:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def load(self):
        logging.info("Loading the PDF from: %s", self.pdf_path)
        with open(self.pdf_path, "rb") as f:
            self.documents = pickle.load(f)
        return self.documents
    
    def create_and_dump(self):
        logging.info("Document caching not present. Splitting and caching the document from : %s", self.pdf_path)
        self.loader = PyPDFLoader(str(self.pdf_path))
        self.docs = self.loader.load()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.documents = self.splitter.split_documents(self.docs)
        with open(self.pdf_path, "wb") as f:
            pickle.dump(self.documents, f)
        return self.documents