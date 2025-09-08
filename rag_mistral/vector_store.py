import os
import pickle
import logging
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from functools import lru_cache
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFAISSVectorStore:
    """Heavily optimized FAISS vector store for sub-second retrieval"""
    
    def __init__(self, embedding_model_name: str, vector_db_path: str, device: str = "cuda"):
        self.embedding_model_name = embedding_model_name
        self.vector_db_path = vector_db_path
        self.device = device
        
        # Initialize optimized embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Move to GPU if available
        if device == "cuda" and torch.cuda.is_available():
            self.embedding_model = self.embedding_model.cuda()
            logger.info(f"Embedding model moved to {device}")
        
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize optimized FAISS index
        self._init_optimized_index()
        
        self.documents = []
        self.doc_embeddings = []
        
        # Performance optimizations
        self.query_cache = {}
        self.batch_size = 64  # Larger batch size for GPU
        self.max_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Load existing index
        self.load_index()
        
        # Pre-compute normalized embeddings for faster search
        self._precompute_optimizations()
    
    def _init_optimized_index(self):
        """Initialize optimized FAISS index for maximum performance"""
        
        # Use GPU index if available
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            try:
                # GPU-accelerated index
                self.gpu_resource = faiss.StandardGpuResources()
                
                # Create CPU index first
                cpu_index = faiss.IndexHNSWFlat(self.dimension, 32)
                cpu_index.hnsw.efConstruction = 200
                cpu_index.hnsw.efSearch = 50
                
                # Move to GPU
                self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)
                logger.info("Using GPU-accelerated FAISS index")
                
            except Exception as e:
                logger.warning(f"GPU FAISS initialization failed: {e}, falling back to CPU")
                self._init_cpu_index()
        else:
            self._init_cpu_index()
    
    def _init_cpu_index(self):
        """Initialize optimized CPU FAISS index"""
        # Use IVF index for large datasets (faster search)
        if hasattr(self, 'documents') and len(self.documents) > 10000:
            # IVF index for large datasets
            nlist = min(4096, max(1, len(self.documents) // 39))
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.nprobe = min(nlist, 10)  # Number of clusters to search
        else:
            # HNSW for smaller datasets (better quality)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100  # Increased for better recall
        
        logger.info("Using optimized CPU FAISS index")
    
    def _precompute_optimizations(self):
        """Pre-compute optimizations after loading"""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained and len(self.documents) > 0:
            # Train IVF index if needed
            if hasattr(self, 'doc_embeddings') and len(self.doc_embeddings) > 0:
                embeddings_array = np.array(self.doc_embeddings).astype('float32')
                self.index.train(embeddings_array)
                logger.info("FAISS index trained")
    
    def encode_texts_optimized(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Optimized batch encoding with GPU acceleration"""
        if batch_size is None:
            batch_size = self.batch_size
        
        embeddings = []
        
        # Process in optimized batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            
            # Encode with optimizations
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(batch),  # Process full batch
                device=self.device if self.device == "cuda" else None
            )
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    @lru_cache(maxsize=1000)
    def _get_cached_query_embedding(self, query_hash: str) -> Optional[np.ndarray]:
        """LRU cache for query embeddings"""
        return None  # LRU cache handles the storage
    
    def _cache_query_embedding(self, query: str, embedding: np.ndarray):
        """Cache query embedding"""
        query_hash = hash(query)
        self.query_cache[query_hash] = embedding
        # Trigger LRU cache
        self._get_cached_query_embedding(query_hash)
    
    def encode_query_optimized(self, query: str, use_cache: bool = True) -> np.ndarray:
        """Optimized single query encoding with caching"""
        if use_cache:
            query_hash = hash(query)
            if query_hash in self.query_cache:
                return self.query_cache[query_hash]
        
        # Encode single query
        embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device if self.device == "cuda" else None
        )[0]
        
        if use_cache:
            self._cache_query_embedding(query, embedding)
        
        return embedding.astype(np.float32)
    
    def add_documents(self, documents: List[Document]):
        """Optimized document addition with batch processing"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        start_time = time.time()
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in optimized batches
        embeddings = self.encode_texts_optimized(texts)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.doc_embeddings.extend(embeddings)
        
        # Re-optimize index if needed
        self._precompute_optimizations()
        
        duration = time.time() - start_time
        logger.info(f"Added {len(documents)} documents in {duration:.3f} seconds")
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search_optimized(self, query: str, k: int = 5, use_cache: bool = True) -> List[Document]:
        """Optimized similarity search with caching"""
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        start_time = time.time()
        
        # Encode query with caching
        query_embedding = self.encode_query_optimized(query, use_cache)
        
        # Search with optimized parameters
        k_search = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k_search)
        
        # Return documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        
        search_time = time.time() - start_time
        logger.debug(f"Search completed in {search_time:.4f} seconds")
        
        return results
    
    async def similarity_search_async(self, query: str, k: int = 5, use_cache: bool = True) -> List[Document]:
        """Async similarity search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.similarity_search_optimized,
            query,
            k,
            use_cache
        )
    
    async def batch_similarity_search_async(self, queries: List[str], k: int = 5, use_cache: bool = True) -> List[List[Document]]:
        """Optimized batch similarity search for parallel queries"""
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return [[] for _ in queries]
        
        start_time = time.time()
        
        # Encode all queries in batch
        query_embeddings = []
        for query in queries:
            if use_cache:
                query_hash = hash(query)
                if query_hash in self.query_cache:
                    query_embeddings.append(self.query_cache[query_hash])
                    continue
            
            embedding = self.encode_query_optimized(query, use_cache)
            query_embeddings.append(embedding)
        
        # Convert to batch array
        query_batch = np.array(query_embeddings, dtype=np.float32)
        
        # Batch search
        k_search = min(k, len(self.documents))
        distances, indices = self.index.search(query_batch, k_search)
        
        # Prepare results
        results = []
        for query_indices in indices:
            query_results = []
            for idx in query_indices:
                if 0 <= idx < len(self.documents):
                    query_results.append(self.documents[idx])
            results.append(query_results)
        
        search_time = time.time() - start_time
        logger.info(f"Batch search of {len(queries)} queries completed in {search_time:.4f} seconds")
        
        return results
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Standard similarity search (wrapper for backward compatibility)"""
        return self.similarity_search_optimized(query, k)
    
    def save_index(self):
        """Save the optimized FAISS index and documents"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Save FAISS index (handle GPU index)
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        
        if hasattr(self, 'gpu_resource'):
            # Move GPU index to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        embeddings_path = os.path.join(self.vector_db_path, "embeddings.npy")
        
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'model_name': self.embedding_model_name,
                'dimension': self.dimension,
                'index_type': type(self.index).__name__
            }, f)
        
        # Save embeddings separately for faster loading
        if self.doc_embeddings:
            np.save(embeddings_path, np.array(self.doc_embeddings))
        
        logger.info(f"Optimized vector store saved to {self.vector_db_path}")
    
    def load_index(self):
        """Load existing optimized FAISS index and documents"""
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        embeddings_path = os.path.join(self.vector_db_path, "embeddings.npy")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                start_time = time.time()
                
                # Load FAISS index
                cpu_index = faiss.read_index(index_path)
                
                # Move to GPU if available
                if hasattr(self, 'gpu_resource'):
                    try:
                        self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)
                    except Exception as e:
                        logger.warning(f"Failed to move index to GPU: {e}")
                        self.index = cpu_index
                else:
                    self.index = cpu_index
                
                # Load documents
                with open(docs_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                
                # Load embeddings
                if os.path.exists(embeddings_path):
                    self.doc_embeddings = np.load(embeddings_path).tolist()
                else:
                    # Fallback to pickle format
                    self.doc_embeddings = data.get('embeddings', [])
                
                load_time = time.time() - start_time
                logger.info(f"Loaded {len(self.documents)} documents in {load_time:.3f} seconds")
                
                # Pre-compute optimizations
                self._precompute_optimizations()
                
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                self.documents = []
                self.doc_embeddings = []
                self._init_optimized_index()
    
    def clear_index(self):
        """Clear the vector store"""
        self.index.reset()
        self.documents = []
        self.doc_embeddings = []
        self.query_cache.clear()
        self._get_cached_query_embedding.cache_clear()
        logger.info("Optimized vector store cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_documents": len(self.documents),
            "index_type": type(self.index).__name__,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "device": self.device,
            "cache_size": len(self.query_cache),
            "batch_size": self.batch_size,
            "gpu_available": torch.cuda.is_available(),
            "using_gpu_index": hasattr(self, 'gpu_resource'),
            "index_trained": getattr(self.index, 'is_trained', True)
        }
    
    def optimize_for_queries(self):
        """Additional optimizations for query performance"""
        if hasattr(self.index, 'hnsw'):
            # Optimize HNSW parameters for faster search
            self.index.hnsw.efSearch = 50  # Balance between speed and quality
        
        if hasattr(self.index, 'nprobe'):
            # Optimize IVF parameters
            self.index.nprobe = min(self.index.nlist, 20)
        
        logger.info("Index optimized for query performance")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'gpu_resource'):
            try:
                del self.gpu_resource
            except:
                pass

# Alias for backward compatibility
FAISSVectorStore = OptimizedFAISSVectorStore
