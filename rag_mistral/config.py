import os
from pathlib import Path

# Model Configuration - Optimized for A100
# Update this path to your actual Mistral model location
MISTRAL_MODEL_PATH = "Mistral-7B-Instruct-v0.2"  # Will download automatically
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, high-quality embeddings

# RAG Configuration - Optimized for speed
CHUNK_SIZE = 800  # Reduced for faster processing
CHUNK_OVERLAP = 150  # Reduced overlap
TOP_K_RETRIEVAL = 4  # Reduced for faster LLM processing
MAX_CONTEXT_LENGTH = 3000  # Reduced for faster generation

# VLLM Configuration for A100 - Maximum Performance
VLLM_CONFIG = {
    "tensor_parallel_size": 1,  # Adjust based on your A100 setup
    "gpu_memory_utilization": 0.95,  # Aggressive memory usage
    "max_model_len": 16384,  # Reduced for speed
    "quantization": None,  # Keep None for A100
    "dtype": "float16",  # Half precision for speed
    "trust_remote_code": True,
    "max_num_batched_tokens": 8192,  # Larger batch size
    "max_num_seqs": 16,  # Multiple sequences in parallel
    "swap_space": 4,  # GB of swap space
    "disable_log_stats": True,  # Reduce logging overhead
    "enforce_eager": False,  # Use CUDA graphs when possible
}

# Performance Optimizations
USE_OPTIMIZED_BACKENDS = True
ENABLE_RESPONSE_CACHING = True
ENABLE_QUERY_CACHING = True
ASYNC_PROCESSING = True
BATCH_SIZE_EMBEDDINGS = 64
BATCH_SIZE_LLM = 8
MAX_PARALLEL_WORKERS = 4

# GPU Optimizations
USE_GPU_EMBEDDINGS = True
USE_GPU_FAISS = True
ENABLE_MIXED_PRECISION = True
COMPILE_MODEL = True  # PyTorch 2.0+ compilation

# File Paths
COURSEWORK_FOLDER = "coursework"
VECTOR_DB_PATH = "vector_db_optimized"
CACHE_DIR = "cache_optimized"

# Create directories
for path in [COURSEWORK_FOLDER, VECTOR_DB_PATH, CACHE_DIR]:
    Path(path).mkdir(exist_ok=True)

# Logging Configuration
ENABLE_PERFORMANCE_LOGGING = True
LOG_LEVEL = "INFO"
