#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG System with Gemma-7B and Comprehensive System Monitoring
Author: AI Engineering Expert
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

# Core ML libraries
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer

# Vector database and document processing
import faiss
import numpy as np
from PyPDF2 import PdfReader
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# System monitoring
import psutil
import GPUtil
import threading
import queue
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStats:
    """System statistics dataclass for comprehensive monitoring"""
    timestamp: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization: float
    gpu_temperature: float
    step_name: str
    duration_seconds: float = 0.0

@dataclass
class Document:
    """Document representation with metadata"""
    content: str
    source: str
    page_number: int
    chunk_id: int
    embedding: Optional[np.ndarray] = None

class SystemMonitor:
    """Advanced system monitoring with real-time statistics"""
    
    def __init__(self):
        self.stats_queue = queue.Queue()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                stats = self._get_current_stats("Continuous Monitoring")
                self.stats_queue.put(stats)
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
    
    def _get_current_stats(self, step_name: str) -> SystemStats:
        """Get current system statistics"""
        # CPU and RAM stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        
        # GPU stats
        gpu_memory_used_mb = 0
        gpu_memory_total_mb = 0
        gpu_utilization = 0
        gpu_temperature = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU
                gpu_memory_used_mb = gpu.memoryUsed
                gpu_memory_total_mb = gpu.memoryTotal
                gpu_utilization = gpu.load * 100
                gpu_temperature = gpu.temperature
        except Exception as e:
            logger.warning(f"GPU monitoring error: {e}")
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            step_name=step_name
        )
    
    @contextmanager
    def monitor_step(self, step_name: str):
        """Context manager for monitoring specific steps"""
        start_time = time.time()
        start_stats = self._get_current_stats(f"{step_name} - Start")
        
        logger.info(f"\n[STEP] [{step_name}] Starting...")
        logger.info(f"   CPU: {start_stats.cpu_percent:.1f}%")
        logger.info(f"   RAM: {start_stats.ram_used_gb:.2f}/{start_stats.ram_total_gb:.2f} GB")
        logger.info(f"   GPU Memory: {start_stats.gpu_memory_used_mb:.0f}/{start_stats.gpu_memory_total_mb:.0f} MB")
        logger.info(f"   GPU Utilization: {start_stats.gpu_utilization:.1f}%")
        
        try:
            yield start_stats
        finally:
            end_time = time.time()
            duration = end_time - start_time
            end_stats = self._get_current_stats(f"{step_name} - End")
            end_stats.duration_seconds = duration
            
            logger.info(f"[DONE] [{step_name}] Completed in {duration:.2f}s")
            logger.info(f"   Memory Delta: {end_stats.ram_used_gb - start_stats.ram_used_gb:.2f} GB")
            logger.info(f"   GPU Memory Delta: {end_stats.gpu_memory_used_mb - start_stats.gpu_memory_used_mb:.0f} MB")

class AdvancedPDFProcessor:
    """High-performance PDF processing with intelligent chunking"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.monitor = SystemMonitor()
    
    def process_pdf_directory(self, pdf_dir: str) -> List[Document]:
        """Process all PDFs in a directory with comprehensive error handling"""
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        
        with self.monitor.monitor_step("PDF Processing"):
            for pdf_file in pdf_files:
                try:
                    documents = self._process_single_pdf(pdf_file)
                    all_documents.extend(documents)
                    logger.info(f"Processed {pdf_file.name}: {len(documents)} chunks")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file.name}: {e}")
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
    
    def _process_single_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF file with advanced chunking"""
        documents = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks = self._intelligent_chunk_text(text)
                        for chunk_id, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 50:  # Filter very short chunks
                                documents.append(Document(
                                    content=chunk.strip(),
                                    source=pdf_path.name,
                                    page_number=page_num + 1,
                                    chunk_id=chunk_id
                                ))
        except Exception as e:
            logger.error(f"PDF parsing error for {pdf_path.name}: {e}")
            raise
        
        return documents
    
    def _intelligent_chunk_text(self, text: str) -> List[str]:
        """Intelligent text chunking with sentence boundary preservation"""
        # Split by sentences first
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Handle overlap by keeping last part of current chunk
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap//10:] if len(words) > self.chunk_overlap//10 else []
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ('. ' if current_chunk else '') + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class VectorDatabase:
    """High-performance vector database with FAISS optimization"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.monitor = SystemMonitor()
        self._embedding_dim = 384  # Default for all-MiniLM-L6-v2
    
    def initialize_embedding_model(self):
        """Initialize the embedding model with optimization"""
        with self.monitor.monitor_step("Embedding Model Loading"):
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self._embedding_dim}")
    
    def build_vector_database(self, documents: List[Document], save_path: str = "vector_db"):
        """Build optimized vector database with batch processing"""
        if not self.embedding_model:
            self.initialize_embedding_model()
        
        with self.monitor.monitor_step("Vector Database Creation"):
            logger.info(f"Creating embeddings for {len(documents)} documents")
            
            # Batch processing for efficiency
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_texts = [doc.content for doc in batch_docs]
                
                with torch.no_grad():
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=True if i == 0 else False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                embeddings.extend(batch_embeddings)
                
                # Update progress
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
            
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Build FAISS index with optimization
            logger.info("Building FAISS index...")
            if len(documents) > 1000:
                # Use IVF index for large datasets
                nlist = min(100, len(documents) // 10)
                quantizer = faiss.IndexFlatIP(self._embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self._embedding_dim, nlist)
                self.index.train(embeddings_array)
            else:
                # Use flat index for smaller datasets
                self.index = faiss.IndexFlatIP(self._embedding_dim)
            
            self.index.add(embeddings_array)
            self.documents = documents
            
            # Save the database
            self._save_database(save_path)
            logger.info(f"Vector database saved to {save_path}")
    
    def load_database(self, load_path: str):
        """Load pre-built vector database"""
        with self.monitor.monitor_step("Loading Vector Database"):
            db_path = Path(load_path)
            
            # Load FAISS index
            index_path = db_path / "faiss.index"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            else:
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
            # Load documents
            docs_path = db_path / "documents.pkl"
            if docs_path.exists():
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                raise FileNotFoundError(f"Documents file not found: {docs_path}")
            
            # Initialize embedding model
            if not self.embedding_model:
                self.initialize_embedding_model()
            
            logger.info(f"Database loaded: {len(self.documents)} documents")
    
    def _save_database(self, save_path: str):
        """Save vector database components"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # Save documents
        with open(save_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self._embedding_dim,
            "num_documents": len(self.documents),
            "created_at": datetime.now().isoformat()
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Tuple[Document, float]]:
        """Advanced similarity search with score filtering"""
        if not self.embedding_model or not self.index:
            raise ValueError("Database not initialized")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results to filter
        
        # Filter by score threshold and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results[:k]

class GemmaRAGSystem:
    """Advanced RAG system with Gemma-7B integration"""
    
    def __init__(self, model_path: str, vector_db_path: str = None):
        self.model_path = Path(model_path)
        self.vector_db_path = vector_db_path
        self.monitor = SystemMonitor()
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.text_generator = None
        
        # Vector database
        self.vector_db = VectorDatabase()
        
        # Configuration
        self.max_context_length = 2048
        self.generation_config = {
            "max_new_tokens": 1024,  # Increased for complete responses
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,  # Will be set after tokenizer loading
            "eos_token_id": None,  # Will be set after tokenizer loading
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "early_stopping": True,
            "num_beams": 1
        }
    
    def initialize_model(self):
        """Initialize Gemma-7B model with optimizations"""
        with self.monitor.monitor_step("Model Loading"):
            logger.info(f"Loading Gemma-7B from: {self.model_path}")
            
            # Check if model files exist
            required_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
            for file in required_files:
                file_path = self.model_path / file
                if not file_path.exists():
                    raise FileNotFoundError(f"Required model file not found: {file_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=True,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            # Configure quantization for GPU memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("Model loaded successfully")
    
    def initialize_vector_database(self, pdf_directory: str = None):
        """Initialize or load vector database"""
        if self.vector_db_path and Path(self.vector_db_path).exists():
            # Load existing database
            self.vector_db.load_database(self.vector_db_path)
        elif pdf_directory:
            # Build new database
            processor = AdvancedPDFProcessor()
            documents = processor.process_pdf_directory(pdf_directory)
            self.vector_db.build_vector_database(documents, self.vector_db_path or "vector_db")
        else:
            raise ValueError("Either provide existing vector_db_path or pdf_directory to build new database")
    
    def _create_rag_prompt(self, query: str, context_docs: List[Tuple[Document, float]]) -> str:
        """Create optimized RAG prompt for Gemma"""
        context_text = ""
        for i, (doc, score) in enumerate(context_docs, 1):
            context_text += f"[Document {i}] (Source: {doc.source}, Page: {doc.page_number}, Relevance: {score:.3f})\n"
            context_text += f"{doc.content}\n\n"
        
        prompt = f"""Below are relevant documents that contain information to answer the user's question.

Context Documents:
{context_text}

Question: {query}

Based on the provided documents, please provide a comprehensive and detailed answer to the question. Make sure to:
1. Use information from the documents above
2. Explain concepts clearly and thoroughly
3. Include specific details and examples when available
4. Cite the relevant documents when appropriate
5. Provide a complete response that fully addresses the question

Answer:"""
        
        return prompt
    
    def query(self, user_query: str, k_documents: int = 5) -> Dict[str, Any]:
        """Process user query with RAG pipeline"""
        with self.monitor.monitor_step("RAG Query Processing"):
            start_time = time.time()
            
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving documents for: {user_query[:100]}...")
            relevant_docs = self.vector_db.similarity_search(user_query, k=k_documents)
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return {
                    "query": user_query,
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            # Step 2: Create RAG prompt
            rag_prompt = self._create_rag_prompt(user_query, relevant_docs)
            logger.info(f"Generated prompt length: {len(rag_prompt)} characters")
            
            # Step 3: Generate response
            logger.info("Generating response...")
            with torch.no_grad():
                try:
                    # Generate response with improved parameters
                    response = self.text_generator(
                        rag_prompt,
                        max_new_tokens=self.generation_config["max_new_tokens"],
                        temperature=self.generation_config["temperature"],
                        top_p=self.generation_config["top_p"],
                        do_sample=self.generation_config["do_sample"],
                        pad_token_id=self.generation_config["pad_token_id"],
                        eos_token_id=self.generation_config["eos_token_id"],
                        repetition_penalty=self.generation_config["repetition_penalty"],
                        length_penalty=self.generation_config["length_penalty"],
                        return_full_text=False,
                        clean_up_tokenization_spaces=True,
                        num_return_sequences=1
                    )
                    
                    generated_text = response[0]['generated_text'].strip()
                    logger.info(f"Generated response length: {len(generated_text)} characters")
                    
                    # Check if response is empty or too short
                    if not generated_text or len(generated_text.strip()) < 10:
                        logger.warning("Generated response is empty or too short, trying alternative approach")
                        
                        # Try with different parameters
                        alternative_response = self.text_generator(
                            rag_prompt,
                            max_new_tokens=512,
                            temperature=0.8,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=self.generation_config["pad_token_id"],
                            return_full_text=False,
                            num_return_sequences=1
                        )
                        generated_text = alternative_response[0]['generated_text'].strip()
                    
                    # If still empty, provide a fallback based on context
                    if not generated_text or len(generated_text.strip()) < 10:
                        logger.warning("Model failed to generate response, providing context-based fallback")
                        generated_text = self._create_fallback_response(user_query, relevant_docs)
                        
                except Exception as e:
                    logger.error(f"Error during text generation: {e}")
                    generated_text = self._create_fallback_response(user_query, relevant_docs)
            
            # Ensure response is complete by checking if it ends properly
            if generated_text and not generated_text.endswith(('.', '!', '?', ':')):
                # Try to find the last complete sentence
                sentences = generated_text.split('.')
                if len(sentences) > 1:
                    # Keep all complete sentences except the potentially incomplete last one
                    generated_text = '.'.join(sentences[:-1]) + '.'
            
            # Step 4: Prepare response
            sources = [
                {
                    "source": doc.source,
                    "page": doc.page_number,
                    "chunk_id": doc.chunk_id,
                    "relevance_score": score,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                }
                for doc, score in relevant_docs
            ]
            
            result = {
                "query": user_query,
                "answer": generated_text,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "model_info": {
                    "model": "Gemma-7B",
                    "retrieved_documents": len(relevant_docs),
                    "generation_config": self.generation_config
                }
            }
            
            logger.info(f"Query processed in {result['processing_time']:.2f}s")
            return result

    def _create_fallback_response(self, query: str, context_docs: List[Tuple[Document, float]]) -> str:
        """Create a fallback response when model fails to generate output"""
        if not context_docs:
            return "I apologize, but I couldn't find relevant information to answer your question."
        
        # Create a simple response based on the retrieved documents
        response_parts = [
            f"Based on the retrieved documents, here's what I found regarding '{query}':\n"
        ]
        
        for i, (doc, score) in enumerate(context_docs[:3], 1):  # Use top 3 documents
            content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            response_parts.append(
                f"From {doc.source} (Page {doc.page_number}):\n{content_preview}\n"
            )
        
        response_parts.append(
            "Please note: This response was generated using a fallback method due to model generation issues. "
            "For more detailed information, please refer to the source documents listed above."
        )
        
        return "\n".join(response_parts)

class InteractiveCLI:
    """Simple CLI interface for the RAG system."""

    def __init__(self, rag_system: GemmaRAGSystem):
        self.rag_system = rag_system
        self.monitor = rag_system.monitor
        self.session_history: List[Dict[str, Any]] = []
        self.session_start_time = datetime.now()
        self.session_stats: Dict[str, Any] = {
            "session_id": f"session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}",
            "start_time": self.session_start_time.isoformat(),
            "end_time": None,
            "total_duration_seconds": 0,
            "total_queries": 0,
            "total_processing_time": 0,
            "total_documents_retrieved": 0,
            "average_processing_time": 0,
            "system_stats_timeline": [],
            "model_info": self.rag_system.generation_config,
            "vector_db_info": {
                "embedding_model": self.rag_system.vector_db.embedding_model_name,
                "total_documents": len(self.rag_system.vector_db.documents)
            }
        }

    def run(self):
        """Run an interactive CLI loop."""
        print("\n" + "=" * 80)
        print("ADVANCED RAG SYSTEM WITH GEMMA-7B")
        print("=" * 80)
        print("Interactive Document Q&A System")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 80 + "\n")

        self.monitor.start_monitoring()

        try:
            while True:
                try:
                    user_input = input("\nYour Question: ").strip()
                    if not user_input:
                        continue

                    cmd = user_input.lower()
                    if cmd in ("quit", "exit", "q"):
                        print("Goodbye! Session ended.")
                        break
                    if cmd == "help":
                        self._show_help(); continue
                    if cmd == "stats":
                        self._show_system_stats(); continue
                    if cmd == "history":
                        self._show_session_history(); continue
                    if cmd == "save":
                        self._save_session_data(); continue
                    if cmd == "clear":
                        os.system('clear' if os.name == 'posix' else 'cls'); continue

                    result = self.rag_system.query(user_input)
                    self._display_result(result)
                    self.session_history.append(result)
                    
                    # Update session statistics
                    self._update_session_stats(result)

                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'quit' to exit.")
                    continue
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print(f"Error: {e}")
                    continue
        finally:
            self.monitor.stop_monitoring()
            # Save session data when conversation ends
            self._save_session_data()

    def _display_result(self, result: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("AI RESPONSE")
        print("=" * 80)
        print(result.get('answer', ''))
        print("\nPROCESSING STATS")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"Documents Retrieved: {len(result.get('sources', []))}")
        print("\nSOURCE DOCUMENTS")
        print("-" * 50)
        for i, src in enumerate(result.get('sources', []), 1):
            print(f"{i}. {src.get('source')} (Page {src.get('page')}) - Score: {src.get('relevance_score'):.3f}")
            print(f"   Preview: {src.get('content_preview')}")

    def _show_help(self):
        print("""
AVAILABLE COMMANDS:
  - Ask a question directly
  - help      Show this message
  - stats     Show current system statistics
  - history   Show session history
  - save      Save current session data to JSON file
  - clear     Clear the terminal screen
  - quit/exit Exit the application
        """)

    def _show_system_stats(self):
        stats = self.monitor._get_current_stats("Manual Check")
        print(f"\nCURRENT SYSTEM STATISTICS:\n  CPU Usage: {stats.cpu_percent:.1f}%\n  RAM: {stats.ram_used_gb:.2f}/{stats.ram_total_gb:.2f} GB ({stats.ram_used_gb/stats.ram_total_gb*100:.1f}%)\n  GPU Memory: {stats.gpu_memory_used_mb:.0f}/{stats.gpu_memory_total_mb:.0f} MB\n  GPU Utilization: {stats.gpu_utilization:.1f}%\n  GPU Temperature: {stats.gpu_temperature:.1f} C\n")

    def _show_session_history(self):
        if not self.session_history:
            print("No queries in this session yet.")
            return
        print(f"\nSESSION HISTORY ({len(self.session_history)} queries):")
        print("-" * 60)
        for i, r in enumerate(self.session_history, 1):
            q = r.get('query', '')
            print(f"{i}. {q[:60]}{'...' if len(q) > 60 else ''} | {r.get('processing_time', 0):.2f}s | {len(r.get('sources', []))} docs")

    def _update_session_stats(self, result: Dict[str, Any]):
        """Update session statistics with new query result"""
        self.session_stats["total_queries"] += 1
        self.session_stats["total_processing_time"] += result.get('processing_time', 0)
        self.session_stats["total_documents_retrieved"] += len(result.get('sources', []))
        
        if self.session_stats["total_queries"] > 0:
            self.session_stats["average_processing_time"] = (
                self.session_stats["total_processing_time"] / self.session_stats["total_queries"]
            )
        
        # Capture current system stats
        current_stats = self.monitor._get_current_stats("Query Complete")
        self.session_stats["system_stats_timeline"].append({
            "timestamp": current_stats.timestamp,
            "query_number": self.session_stats["total_queries"],
            "cpu_percent": current_stats.cpu_percent,
            "ram_used_gb": current_stats.ram_used_gb,
            "ram_total_gb": current_stats.ram_total_gb,
            "gpu_memory_used_mb": current_stats.gpu_memory_used_mb,
            "gpu_memory_total_mb": current_stats.gpu_memory_total_mb,
            "gpu_utilization": current_stats.gpu_utilization,
            "gpu_temperature": current_stats.gpu_temperature
        })

    def _save_session_data(self):
        """Save comprehensive session data to JSON file"""
        try:
            session_end_time = datetime.now()
            self.session_stats["end_time"] = session_end_time.isoformat()
            self.session_stats["total_duration_seconds"] = (
                session_end_time - self.session_start_time
            ).total_seconds()
            
            # Prepare comprehensive session data
            session_data = {
                "session_metadata": self.session_stats,
                "conversation_history": self.session_history,
                "system_performance": {
                    "peak_cpu": max([s["cpu_percent"] for s in self.session_stats["system_stats_timeline"]], default=0),
                    "peak_ram_gb": max([s["ram_used_gb"] for s in self.session_stats["system_stats_timeline"]], default=0),
                    "peak_gpu_memory_mb": max([s["gpu_memory_used_mb"] for s in self.session_stats["system_stats_timeline"]], default=0),
                    "peak_gpu_utilization": max([s["gpu_utilization"] for s in self.session_stats["system_stats_timeline"]], default=0),
                    "max_gpu_temperature": max([s["gpu_temperature"] for s in self.session_stats["system_stats_timeline"]], default=0),
                },
                "conversation_summary": {
                    "unique_sources_accessed": len(set([
                        src["source"] for result in self.session_history 
                        for src in result.get("sources", [])
                    ])),
                    "total_response_length": sum([
                        len(result.get("answer", "")) for result in self.session_history
                    ]),
                    "average_response_length": (
                        sum([len(result.get("answer", "")) for result in self.session_history]) / 
                        len(self.session_history) if self.session_history else 0
                    ),
                    "queries_by_length": {
                        "short_queries": len([r for r in self.session_history if len(r.get("query", "")) < 50]),
                        "medium_queries": len([r for r in self.session_history if 50 <= len(r.get("query", "")) < 100]),
                        "long_queries": len([r for r in self.session_history if len(r.get("query", "")) >= 100])
                    }
                }
            }
            
            # Generate filename with timestamp
            filename = f"rag_session_{self.session_stats['session_id']}.json"
            filepath = Path(filename)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n?? Session data saved to: {filepath.absolute()}")
            print(f"Total queries: {self.session_stats['total_queries']}")
            print(f"Session duration: {self.session_stats['total_duration_seconds']:.1f} seconds")
            print(f"Average processing time: {self.session_stats['average_processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            print(f"Warning: Could not save session data - {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Advanced RAG System with Gemma-7B")
    parser.add_argument("--model-path", required=True, help="Path to Gemma-7B model directory")
    parser.add_argument("--pdf-dir", help="Directory containing PDF documents")
    parser.add_argument("--vector-db", help="Path to existing vector database")
    parser.add_argument("--build-db", action="store_true", help="Force rebuild of vector database")
    
    args = parser.parse_args()
    
    try:
        print("Initializing Advanced RAG System...")
        
        # Initialize RAG system
        rag_system = GemmaRAGSystem(args.model_path, args.vector_db)
        
        # Load model
        rag_system.initialize_model()
        
        # Initialize vector database
        if args.build_db or not (args.vector_db and Path(args.vector_db).exists()):
            if not args.pdf_dir:
                raise ValueError("PDF directory required for building vector database")
            rag_system.initialize_vector_database(args.pdf_dir)
        else:
            rag_system.initialize_vector_database()
        
        # Start interactive CLI
        cli = InteractiveCLI(rag_system)
        cli.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()