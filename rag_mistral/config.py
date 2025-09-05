import os
from pathlib import Path

# Model Configuration
MISTRAL_MODEL_PATH = "/path/to/your/mistral-7b-instruct-v0.2"  # Update this path
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, high-quality embeddings

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
MAX_CONTEXT_LENGTH = 4000

# VLLM Configuration for A100
VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "quantization": None,  # Keep None for A100, use "awq" if you have quantized model
}

# File Paths
COURSEWORK_FOLDER = "coursework"
VECTOR_DB_PATH = "vector_db"
CACHE_DIR = "cache"

# Create directories
for path in [COURSEWORK_FOLDER, VECTOR_DB_PATH, CACHE_DIR]:
    Path(path).mkdir(exist_ok=True)
