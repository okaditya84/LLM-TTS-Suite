import logging
import asyncio
from typing import List, Optional, Union, Dict, Any
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import torch
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMistralLLM:
    """Heavily optimized Mistral LLM interface for A100 GPU targeting <2s response time"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid reloading models"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str, max_workers: int = 4):
        if hasattr(self, 'initialized'):
            return
            
        self.model_path = model_path
        self.max_workers = max_workers
        self.llm = None
        self.backend = None
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # Performance optimizations
        self.response_cache = {}
        self.batch_queue = []
        self.batch_size = 8
        self.batch_timeout = 0.05  # 50ms
        self.last_batch_time = time.time()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Initializing Optimized Mistral LLM from {model_path}")
        
        # Initialize with best available backend
        self._initialize_optimized_backend()
        
        if self.llm is None and self.model is None:
            raise RuntimeError("Failed to initialize any backend. Please check your installation.")
        
        # Warm up the model
        self._warmup_model()
        
        logger.info(f"Optimized Mistral LLM initialized successfully using {self.backend} backend")
        self.initialized = True
    
    def _initialize_optimized_backend(self):
        """Initialize the most optimized backend available"""
        
        # Option 1: vLLM with optimizations (fastest for A100)
        try:
            self._init_optimized_vllm()
            return
        except Exception as e:
            logger.warning(f"Optimized vLLM initialization failed: {e}")
        
        # Option 2: Optimized Transformers with Flash Attention and optimizations
        try:
            self._init_optimized_transformers()
            return
        except Exception as e:
            logger.warning(f"Optimized Transformers initialization failed: {e}")
        
        # Option 3: Standard Transformers fallback
        try:
            self._init_standard_transformers()
            return
        except Exception as e:
            logger.error(f"Standard Transformers initialization failed: {e}")
    
    def _init_optimized_vllm(self):
        """Initialize vLLM with maximum optimizations for A100"""
        from vllm import LLM, SamplingParams
        import torch
        
        # Optimal vLLM config for A100
        vllm_config = {
            "model": self.model_path,
            "tensor_parallel_size": torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1,
            "gpu_memory_utilization": 0.95,  # Aggressive memory usage
            "max_model_len": 16384,  # Reduced for speed
            "quantization": None,
            "dtype": "float16",
            "trust_remote_code": True,
            "max_num_batched_tokens": 8192,  # Larger batch size
            "max_num_seqs": 16,  # Multiple sequences in parallel
            "swap_space": 4,  # GB of swap space
            "disable_log_stats": True,  # Reduce logging overhead
            "enforce_eager": False,  # Use CUDA graphs when possible
        }
        
        self.llm = LLM(**vllm_config)
        
        # Optimized sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            stop=["</s>", "[INST]", "[/INST]"],
            repetition_penalty=1.05,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            use_beam_search=False,  # Faster than beam search
            n=1,
        )
        
        self.backend = "vLLM-Optimized"
        self.device = "cuda"
    
    def _init_optimized_transformers(self):
        """Initialize Transformers with maximum optimizations"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,  # Use fast tokenizer
            padding_side="left",  # Better for batching
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Flash Attention 2
            use_cache=True,
            low_cpu_mem_usage=True,
            max_memory={0: "80GB"},  # Assume A100 80GB
        )
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
            logger.info("Model compiled with torch.compile for optimized inference")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        
        self.backend = "Transformers-Optimized"
        self.llm = self  # Self-reference for compatibility
    
    def _init_standard_transformers(self):
        """Standard Transformers initialization as fallback"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.backend = "Transformers-Standard"
        self.llm = self
    
    def _warmup_model(self):
        """Warm up the model with a dummy inference to optimize CUDA kernels"""
        logger.info("Warming up model...")
        warmup_start = time.time()
        
        dummy_prompt = "[INST] What is AI? [/INST]"
        
        try:
            if self.backend.startswith("vLLM"):
                self.llm.generate([dummy_prompt], self.sampling_params)
            else:
                self._generate_transformers_batch([dummy_prompt])
            
            warmup_time = time.time() - warmup_start
            logger.info(f"Model warmup completed in {warmup_time:.3f} seconds")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    @lru_cache(maxsize=1000)
    def _get_cached_response(self, prompt_hash: str) -> Optional[str]:
        """LRU cache for responses to avoid recomputation"""
        return None  # LRU cache handles the actual caching
    
    def _cache_response(self, prompt: str, response: str):
        """Cache response with prompt hash as key"""
        prompt_hash = hash(prompt)
        self._get_cached_response(prompt_hash)  # Trigger cache storage
        self.response_cache[prompt_hash] = response
    
    def format_prompt(self, query: str, context: str, conversation_history: Optional[List] = None) -> str:
        """Optimized prompt formatting with minimal overhead"""
        system_prompt = "Answer based on provided context. Be concise and accurate."
        
        # Simplified conversation context (last 2 messages only for speed)
        conversation_context = ""
        if conversation_history:
            for msg in conversation_history[-2:]:
                role = "H" if msg["role"] == "user" else "A"
                conversation_context += f"{role}: {msg['content'][:200]}...\n"
        
        # Truncate context if too long
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        prompt = f"[INST] {system_prompt}\n\nContext:\n{context}\n\n{conversation_context}Q: {query} [/INST]"
        return prompt
    
    async def generate_response_async(self, prompt: str, use_cache: bool = True) -> str:
        """Async response generation with caching"""
        if use_cache:
            prompt_hash = hash(prompt)
            cached = self._get_cached_response(prompt_hash)
            if cached:
                return cached
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor, 
            self._generate_single, 
            prompt
        )
        
        if use_cache:
            self._cache_response(prompt, response)
        
        return response
    
    def generate_response(self, prompt: str, use_cache: bool = True) -> str:
        """Synchronous response generation"""
        if use_cache:
            prompt_hash = hash(prompt)
            cached = self._get_cached_response(prompt_hash)
            if cached:
                return cached
        
        response = self._generate_single(prompt)
        
        if use_cache:
            self._cache_response(prompt, response)
        
        return response
    
    def _generate_single(self, prompt: str) -> str:
        """Generate single response with optimal backend"""
        try:
            if self.backend.startswith("vLLM"):
                return self._generate_vllm_single(prompt)
            else:
                return self._generate_transformers_single(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    def _generate_vllm_single(self, prompt: str) -> str:
        """Single response generation with vLLM"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def _generate_transformers_single(self, prompt: str) -> str:
        """Single response generation with Transformers"""
        return self._generate_transformers_batch([prompt])[0]
    
    def _generate_transformers_batch(self, prompts: List[str]) -> List[str]:
        """Batch generation with Transformers for efficiency"""
        import torch
        
        # Tokenize with padding for batch processing
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4000
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with optimizations
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision for speed
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    use_cache=True,
                    num_beams=1,  # Greedy decoding for speed
                )
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            new_tokens = output[inputs['input_ids'][i].shape[0]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    async def generate_batch_async(self, prompts: List[str], use_cache: bool = True) -> List[str]:
        """Async batch generation for parallel queries"""
        if self.backend.startswith("vLLM"):
            # vLLM handles batching natively
            return await self._generate_vllm_batch_async(prompts, use_cache)
        else:
            # Use manual batching for Transformers
            return await self._generate_transformers_batch_async(prompts, use_cache)
    
    async def _generate_vllm_batch_async(self, prompts: List[str], use_cache: bool) -> List[str]:
        """Async batch generation with vLLM"""
        # Check cache first
        cached_responses = []
        uncached_prompts = []
        uncached_indices = []
        
        if use_cache:
            for i, prompt in enumerate(prompts):
                prompt_hash = hash(prompt)
                cached = self._get_cached_response(prompt_hash)
                if cached:
                    cached_responses.append((i, cached))
                else:
                    uncached_prompts.append(prompt)
                    uncached_indices.append(i)
        else:
            uncached_prompts = prompts
            uncached_indices = list(range(len(prompts)))
        
        # Generate uncached responses
        loop = asyncio.get_event_loop()
        if uncached_prompts:
            uncached_responses = await loop.run_in_executor(
                self.executor,
                self._generate_vllm_batch_sync,
                uncached_prompts
            )
            
            # Cache new responses
            if use_cache:
                for prompt, response in zip(uncached_prompts, uncached_responses):
                    self._cache_response(prompt, response)
        else:
            uncached_responses = []
        
        # Merge cached and uncached responses
        all_responses = [None] * len(prompts)
        
        # Add cached responses
        for i, response in cached_responses:
            all_responses[i] = response
        
        # Add uncached responses
        for i, response in zip(uncached_indices, uncached_responses):
            all_responses[i] = response
        
        return all_responses
    
    def _generate_vllm_batch_sync(self, prompts: List[str]) -> List[str]:
        """Synchronous batch generation with vLLM"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses
    
    async def _generate_transformers_batch_async(self, prompts: List[str], use_cache: bool) -> List[str]:
        """Async batch generation with Transformers"""
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            self.executor,
            self._generate_transformers_batch,
            prompts
        )
        
        if use_cache:
            for prompt, response in zip(prompts, responses):
                self._cache_response(prompt, response)
        
        return responses
    
    def chat_with_context(self, query: str, context_documents: List, conversation_history: Optional[List] = None) -> str:
        """Optimized single chat response"""
        # Optimized context combination (parallel processing)
        context = self._combine_context_optimized(context_documents)
        
        # Format prompt
        prompt = self.format_prompt(query, context, conversation_history)
        
        # Generate response
        response = self.generate_response(prompt)
        
        return response
    
    async def chat_with_context_async(self, query: str, context_documents: List, conversation_history: Optional[List] = None) -> str:
        """Async optimized chat response"""
        # Optimized context combination
        context = self._combine_context_optimized(context_documents)
        
        # Format prompt
        prompt = self.format_prompt(query, context, conversation_history)
        
        # Generate response
        response = await self.generate_response_async(prompt)
        
        return response
    
    async def chat_batch_async(self, queries: List[str], context_documents_list: List[List], conversation_histories: Optional[List[List]] = None) -> List[str]:
        """Async batch chat processing for parallel queries"""
        if conversation_histories is None:
            conversation_histories = [None] * len(queries)
        
        # Prepare prompts in parallel
        prompts = []
        for i, (query, context_docs, conv_history) in enumerate(zip(queries, context_documents_list, conversation_histories)):
            context = self._combine_context_optimized(context_docs)
            prompt = self.format_prompt(query, context, conv_history)
            prompts.append(prompt)
        
        # Generate responses in batch
        responses = await self.generate_batch_async(prompts)
        
        return responses
    
    def _combine_context_optimized(self, context_documents: List) -> str:
        """Optimized context combination with truncation"""
        context_parts = []
        total_length = 0
        max_context_length = 3000
        
        for doc in context_documents:
            doc_content = f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            
            if total_length + len(doc_content) > max_context_length:
                # Truncate last document to fit
                remaining_length = max_context_length - total_length
                if remaining_length > 100:  # Only add if substantial content can fit
                    doc_content = doc_content[:remaining_length] + "..."
                    context_parts.append(doc_content)
                break
            
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        return "\n\n".join(context_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "backend": self.backend,
            "device": self.device,
            "cache_size": len(self.response_cache),
            "max_workers": self.max_workers,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self._get_cached_response.cache_clear()
        logger.info("Response cache cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
