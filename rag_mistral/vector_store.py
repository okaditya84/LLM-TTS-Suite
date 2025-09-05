import os
import pickle
import logging
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_model_name: str, vector_db_path: str):
        self.embedding_model_name = embedding_model_name
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (using HNSW for speed)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50
        
        self.documents = []
        self.doc_embeddings = []
        
        # Try to load existing index
        self.load_index()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings with batching for efficiency"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.doc_embeddings.extend(embeddings)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def save_index(self):
        """Save the FAISS index and documents"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.doc_embeddings,
                'model_name': self.embedding_model_name
            }, f)
        
        logger.info(f"Vector store saved to {self.vector_db_path}")
    
    def load_index(self):
        """Load existing FAISS index and documents"""
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load documents
                with open(docs_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.doc_embeddings = data['embeddings']
                
                logger.info(f"Loaded {len(self.documents)} documents from existing vector store")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                self.documents = []
                self.doc_embeddings = []
    
    def clear_index(self):
        """Clear the vector store"""
        self.index.reset()
        self.documents = []
        self.doc_embeddings = []
        logger.info("Vector store cleared")
