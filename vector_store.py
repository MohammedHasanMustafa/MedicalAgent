# vector_store.py
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from typing import List, Dict
import os

class VectorStoreManager:
    def __init__(self, embeddings, config):
        self.embeddings = embeddings
        self.config = config
        self.vector_store = None
    
    def create_vector_store(self, all_documents: Dict[str, List[Document]]):
        """Create vector store from all documents"""
        # Combine all documents
        all_docs = []
        for dataset_name, documents in all_documents.items():
            all_docs.extend(documents)
        
        print(f"Creating vector store with {len(all_docs)} total documents...")
        
        if self.config.VECTOR_STORE_TYPE == "faiss":
            self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
            self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
        else:
            self.vector_store = Chroma.from_documents(
                all_docs, 
                self.embeddings, 
                persist_directory=self.config.VECTOR_STORE_PATH
            )
        
        print("✓ Vector store created successfully!")
        return self.vector_store
    
    def load_vector_store(self):
        """Load existing vector store"""
        if self.config.VECTOR_STORE_TYPE == "faiss":
            self.vector_store = FAISS.load_local(
                self.config.VECTOR_STORE_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.config.VECTOR_STORE_PATH,
                embedding_function=self.embeddings
            )
        return self.vector_store
