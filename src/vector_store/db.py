from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import Config
from typing import List, Optional, Dict
from langchain_core.documents import Document
from src.logger import logging as log
from src.exception import CustomException


config = Config()

class VectorStore:
    def __init__(self, db_path: str = config.VECTOR_STORE_DIR) -> None:
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            self.vector_store = Chroma(
                collection_name=config.DB_NAME,
                embedding_function=self.embeddings,
                persist_directory=db_path
            )
        except Exception as e:
            raise CustomException(f"VectorStore initialization failed: {e}", e)


    def add_documents(self, documents: List[Document]) -> None:
        """Add documents with metadata to the vector store"""
        log.info(f"Adding docs to chromaDB::length={len(documents)}")
        try:
            log.info(f"Adding {len(documents)} documents to ChromaDB.")
            self.vector_store.add_documents(documents)
        except Exception as e:
            raise CustomException(f"Error while adding documents: {e}", e)
        
        
    def similarity_search(self, query: str, k: int = 4, metadata_filter: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            log.info(f"Performing similarity search with query: {query}")
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter
            )
        except Exception as e:
            raise CustomException(f"Similarity search failed: {e}", e)
        

    def as_retriever(self, search_type: str = "mmr", k: int = 1, fetch_k: int = 5):
        """Create a retriever with specified search parameters"""
        try:
            return self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k, "fetch_k": fetch_k}
            )
        except Exception as e:
            raise CustomException(f"Failed to create retriever: {e}", e)