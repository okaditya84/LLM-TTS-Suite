# chat_with_books.py - Interactive chat using fine-tuned Sarvam-1 + FAISS retrieval
# Requirements: pip install psutil GPUtil (for system metrics)
import json
import numpy as np
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import os
import sys
import time
import threading
from datetime import datetime
from collections import defaultdict

# Optional dependencies for enhanced metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("⚠️ psutil not available - system metrics will be limited")
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    print("⚠️ GPUtil not available - GPU metrics will be limited")
    GPUTIL_AVAILABLE = False

# Configuration
BASE_MODEL_PATH = "/nlsasfs/home/ledgerptf/ashsa/models/sarvam1"
FINETUNED_MODEL_PATH = "fine_tuned_sarvam_books"
INDEX_FILE = "faiss_index.ivf"
META_FILE = "chunks_meta.jsonl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SVD_FILE = "svd_model.pkl"
CHUNKS_FILE = "data_books/chunks.jsonl"

# Chat parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K_RETRIEVAL = 3  # Number of relevant chunks to retrieve

# Metrics configuration
METRICS_LOG_FILE = "chat_metrics.jsonl"

class PerformanceMetrics:
    """Class to track comprehensive performance metrics for production demo with thread safety"""
    
    def __init__(self):
        import threading
        self.lock = threading.Lock()  # Thread safety for metrics
        self.reset_session_metrics()
        
    def reset_session_metrics(self):
        """Reset metrics for a new session"""
        with self.lock:
            self.session_start_time = time.time()
            self.total_queries = 0
            self.total_tokens_generated = 0
            self.cumulative_times = defaultdict(float)
            self.query_metrics = []
        
    def start_query_timing(self):
        """Start timing a new query"""
        return {
            'query_start': time.time(),
            'retrieval_start': None,
            'retrieval_end': None,
            'generation_start': None,
            'generation_end': None,
            'query_end': None,
            'system_metrics': self.get_system_metrics()
        }
    
    def get_system_metrics(self):
        """Get current system resource usage with improved GPU detection"""
        metrics = {"timestamp": time.time()}
        
        # CPU and Memory metrics
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                metrics.update({
                    "cpu_percent": cpu_percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "memory_percent": memory.percent,
                })
            except Exception as e:
                metrics["psutil_error"] = str(e)
        else:
            metrics["psutil_available"] = False
        
        # Enhanced GPU metrics
        gpu_metrics = {"gpu_available": False}
        if torch.cuda.is_available():
            try:
                gpu_metrics["gpu_available"] = True
                gpu_metrics["gpu_count"] = torch.cuda.device_count()
                gpu_metrics["gpu_current_device"] = torch.cuda.current_device()
                
                # PyTorch GPU memory info
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
                
                gpu_metrics.update({
                    "gpu_torch_memory_allocated_gb": gpu_memory_allocated,
                    "gpu_torch_memory_reserved_gb": gpu_memory_reserved,
                    "gpu_torch_memory_cached_gb": gpu_memory_cached,
                })
                
                # Try to get more detailed GPU info
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_metrics.update({
                                "gpu_utilization": gpu.load * 100,
                                "gpu_memory_used_mb": gpu.memoryUsed,
                                "gpu_memory_total_mb": gpu.memoryTotal,
                                "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                "gpu_temperature": gpu.temperature,
                                "gpu_name": gpu.name
                            })
                    except Exception as e:
                        gpu_metrics["gputil_error"] = str(e)
                else:
                    # Use nvidia-ml-py if available
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        gpu_metrics.update({
                            "gpu_utilization": util.gpu,
                            "gpu_memory_used_mb": info.used / (1024**2),
                            "gpu_memory_total_mb": info.total / (1024**2),
                            "gpu_memory_percent": (info.used / info.total) * 100,
                            "gpu_temperature": temp,
                        })
                    except:
                        gpu_metrics["detailed_gpu_available"] = False
            except Exception as e:
                gpu_metrics["gpu_error"] = str(e)
        
        metrics.update(gpu_metrics)
        return metrics
    
    def log_retrieval_timing(self, timing_data, query, chunks_found, retrieval_time):
        """Log retrieval phase metrics"""
        timing_data['retrieval_start'] = time.time() - retrieval_time
        timing_data['retrieval_end'] = time.time()
        timing_data['retrieval_duration'] = retrieval_time
        timing_data['chunks_retrieved'] = chunks_found
        
    def log_generation_timing(self, timing_data, tokens_generated, generation_time):
        """Log generation phase metrics"""
        timing_data['generation_end'] = time.time()
        timing_data['generation_duration'] = generation_time
        timing_data['tokens_generated'] = tokens_generated
        timing_data['tokens_per_second'] = tokens_generated / generation_time if generation_time > 0 else 0
        
    def finalize_query_metrics(self, timing_data, query, response):
        """Finalize and store query metrics with thread safety"""
        timing_data['query_end'] = time.time()
        timing_data['total_duration'] = timing_data['query_end'] - timing_data['query_start']
        timing_data['timestamp'] = datetime.now().isoformat()
        
        # Add query metadata
        timing_data['query_length'] = len(query)
        timing_data['response_length'] = len(response)
        
        with self.lock:  # Thread-safe metrics update
            timing_data['query_id'] = self.total_queries + 1
            
            # Store query metrics
            self.query_metrics.append(timing_data)
            self.total_queries += 1
            self.total_tokens_generated += timing_data.get('tokens_generated', 0)
            
            # Update cumulative times
            for key in ['retrieval_duration', 'generation_duration', 'total_duration']:
                if key in timing_data:
                    self.cumulative_times[key] += timing_data[key]
        
        # Log to file
        self.log_metrics_to_file(timing_data)
        
        return timing_data
    
    def log_metrics_to_file(self, metrics_data):
        """Log metrics to JSONL file for analysis"""
        try:
            with open(METRICS_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_data, default=str) + '\n')
        except Exception as e:
            print(f"Warning: Could not log metrics to file: {e}")
    
    def get_session_summary(self):
        """Get summary of session metrics with thread safety"""
        with self.lock:
            if self.total_queries == 0:
                return {"message": "No queries processed yet"}
            
            session_duration = time.time() - self.session_start_time
            avg_retrieval = self.cumulative_times['retrieval_duration'] / self.total_queries
            avg_generation = self.cumulative_times['generation_duration'] / self.total_queries
            avg_total = self.cumulative_times['total_duration'] / self.total_queries
            avg_tokens_per_second = self.total_tokens_generated / self.cumulative_times['generation_duration'] if self.cumulative_times['generation_duration'] > 0 else 0
        
        return {
            "session_duration": session_duration,
            "total_queries": self.total_queries,
            "total_tokens_generated": self.total_tokens_generated,
            "average_retrieval_time": avg_retrieval,
            "average_generation_time": avg_generation,
            "average_total_time": avg_total,
            "average_tokens_per_second": avg_tokens_per_second,
            "queries_per_minute": (self.total_queries / session_duration) * 60,
            "current_system_metrics": self.get_system_metrics()
        }
    
    def print_query_summary(self, timing_data):
        """Print a formatted summary of the last query metrics"""
        print(f"\n📊 Query Performance Metrics:")
        print(f"   ⏱️  Total Time: {timing_data['total_duration']:.3f}s")
        print(f"   🔍 Retrieval: {timing_data.get('retrieval_duration', 0):.3f}s")
        print(f"   🤖 Generation: {timing_data.get('generation_duration', 0):.3f}s")
        print(f"   📝 Tokens Generated: {timing_data.get('tokens_generated', 0)}")
        print(f"   ⚡ Tokens/Second: {timing_data.get('tokens_per_second', 0):.1f}")
        print(f"   💾 Memory Usage: {timing_data['system_metrics'].get('memory_percent', 0):.1f}%")
        if timing_data['system_metrics'].get('gpu_available'):
            print(f"   🎮 GPU Usage: {timing_data['system_metrics'].get('gpu_memory_percent', 0):.1f}%")

class StreamingTextGenerator:
    """Optimized streaming text generator with proper formatting and GPU utilization"""
    
    def __init__(self, tokenizer, model, max_new_tokens=512, temperature=0.7, top_p=0.9):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(model.parameters()).device
        
    def stream_generate(self, input_text, print_callback=None):
        """Generate text with optimized streaming and proper formatting"""
        # Tokenize input efficiently
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1500,  # Reduced for better performance
            padding=False,
            return_token_type_ids=False
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        generated_text = ""
        tokens_generated = 0
        start_time = time.time()
        
        # Get the original input length to extract only new tokens
        original_length = input_ids.shape[1]
        
        with torch.no_grad():
            # Generate with optimized parameters for speed
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(self.max_new_tokens, 256),  # Reduced for speed
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Reduced for speed
                use_cache=True,
                num_beams=1,  # No beam search for speed
                early_stopping=True,
            )
            
            # Extract only the newly generated tokens
            new_tokens = output_ids[0][original_length:]
            tokens_generated = len(new_tokens)
            
            # Decode with proper formatting
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up the text
            if generated_text:
                generated_text = self.clean_and_format_text(generated_text)
                
                # Output directly without character-by-character streaming for speed
                if print_callback:
                    print_callback(generated_text)
                else:
                    print(generated_text, end='', flush=True)
        
        generation_time = time.time() - start_time
        return generated_text, tokens_generated, generation_time
    
    def clean_and_format_text(self, text):
        """Clean and format the generated text for better readability"""
        import re
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Add proper spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)
        
        # Fix common formatting issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Space between letters and numbers
        
        # Ensure sentences start with capital letters
        sentences = text.split('. ')
        formatted_sentences = []
        for sentence in sentences:
            if sentence:
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    formatted_sentences.append(sentence)
        
        text = '. '.join(formatted_sentences)
        
        # Add proper paragraph breaks for better readability
        text = re.sub(r'(\. )([A-Z][a-z]+ [A-Z][a-z]+)', r'\1\n\n\2', text)
        
        return text

class BookChatBot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.faiss_index = None
        self.metadata = None
        self.vectorizer = None
        self.svd = None
        self.chunks_with_text = {}
        self.streaming_generator = None
        self.metrics = PerformanceMetrics()
        
        # Thread safety locks
        import threading
        self.model_lock = threading.Lock()  # Protect model inference
        self.faiss_lock = threading.Lock()  # Protect FAISS operations
        self.metrics_lock = threading.Lock()  # Protect metrics updates
        
        print("🚀 BookChatBot initialized with performance tracking and thread safety")
        
    def load_models(self):
        """Load the fine-tuned model and tokenizer"""
        print("🤖 Loading fine-tuned Sarvam-1 model...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH, 
                local_files_only=True, 
                use_fast=True
            )
            print("✅ Loaded tokenizer via AutoTokenizer.")
        except Exception as e:
            print("⚠️ AutoTokenizer failed, using PreTrainedTokenizerFast fallback...")
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(BASE_MODEL_PATH, "tokenizer.json"),
                use_fast=True,
            )
            print("✅ Loaded tokenizer via PreTrainedTokenizerFast.")
        
        # Set pad token
        if self.tokenizer.pad_token_id is None:
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
                self.tokenizer.pad_token_id = vocab_size - 1
        
        # Load base model
        print("📂 Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load fine-tuned adapter
        if os.path.exists(FINETUNED_MODEL_PATH):
            print("🎯 Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(self.model, FINETUNED_MODEL_PATH)
            print("✅ Fine-tuned model loaded successfully!")
        else:
            print("⚠️ No fine-tuned model found, using base model only.")
        
        self.model.eval()
        
        # Initialize streaming text generator
        self.streaming_generator = StreamingTextGenerator(
            self.tokenizer, 
            self.model, 
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        print("✅ Streaming text generator initialized")
        
    def load_retrieval_system(self):
        """Load FAISS index and related components"""
        print("🔍 Loading FAISS retrieval system...")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(INDEX_FILE)
        
        # Load metadata
        self.metadata = []
        with open(META_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        # Load chunks with text content
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks_with_text[chunk["id"]] = chunk["text"]
        
        # Load vectorizer and SVD
        with open(VECTORIZER_FILE, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(SVD_FILE, 'rb') as f:
            self.svd = pickle.load(f)
        
        print(f"✅ Loaded index with {self.faiss_index.ntotal} vectors")
        
    def retrieve_relevant_chunks(self, query, timing_data, k=TOP_K_RETRIEVAL):
        """Retrieve relevant text chunks for the query with timing and thread safety"""
        retrieval_start = time.time()
        timing_data['retrieval_start'] = retrieval_start
        
        with self.faiss_lock:  # Thread-safe FAISS operations
            # Transform query using the same pipeline
            query_tfidf = self.vectorizer.transform([query])
            query_vec = self.svd.transform(query_tfidf).astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vec)
            
            # Search
            scores, indices = self.faiss_index.search(query_vec, k)
            
            # Get relevant chunks
            relevant_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.1:  # Filter low relevance
                    meta = self.metadata[idx]
                    chunk_id = meta["id"]
                    text_content = self.chunks_with_text.get(chunk_id, "")
                    relevant_chunks.append({
                        "score": float(score),
                        "text": text_content,
                        "source": meta.get("source", "unknown"),
                        "page": meta.get("page", "unknown")
                    })
        
        retrieval_time = time.time() - retrieval_start
        timing_data['retrieval_end'] = time.time()
        timing_data['retrieval_duration'] = retrieval_time
        timing_data['chunks_retrieved'] = len(relevant_chunks)
        
        return relevant_chunks
    
    def format_prompt(self, user_question, relevant_chunks):
        """Create a prompt with context and question - optimized for better generation"""
        context_text = ""
        if relevant_chunks:
            context_text = "RELEVANT INFORMATION FROM BOOKS:\n"
            for i, chunk in enumerate(relevant_chunks[:3], 1):  # Limit to top 3 chunks for focus
                # Clean and format chunk text
                clean_text = chunk['text'].strip()
                # Ensure proper sentence endings
                if clean_text and not clean_text.endswith(('.', '!', '?')):
                    clean_text += '.'
                    
                context_text += f"[{i}] From {chunk['source']} (Page {chunk['page']}):\n"
                context_text += f"{clean_text[:500]}\n\n"  # Slightly longer context for better understanding
        
        prompt = f"""You are an expert assistant helping with questions about remote sensing and related topics. Use the provided book excerpts to answer the question accurately and comprehensively.

{context_text}

QUESTION: {user_question}

INSTRUCTION: Provide a clear, well-structured answer based on the information above. Use proper paragraph breaks and ensure your response flows naturally.

ANSWER:"""
        
        return prompt
    
    def generate_response(self, prompt, timing_data):
        """Generate response using the fine-tuned model with streaming and thread safety"""
        print("\n🤖 Assistant: ", end='', flush=True)
        
        # Mark generation start
        timing_data['generation_start'] = time.time()
        
        # Thread-safe model access
        with self.model_lock:
            # Use streaming generator
            response, tokens_generated, generation_time = self.streaming_generator.stream_generate(prompt)
        
        # Update timing data
        timing_data['generation_end'] = time.time()
        timing_data['generation_duration'] = generation_time
        timing_data['tokens_generated'] = tokens_generated
        timing_data['tokens_per_second'] = tokens_generated / generation_time if generation_time > 0 else 0
        
        return response.strip()
    
    def chat(self, question):
        """Main chat function with comprehensive metrics"""
        # Start timing for this query
        timing_data = self.metrics.start_query_timing()
        
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching for relevant information...")
        
        try:
            # Retrieve relevant chunks with timing
            relevant_chunks = self.retrieve_relevant_chunks(question, timing_data)
            
            if not relevant_chunks:
                print("⚠️ No relevant information found in the books.")
                response = "I couldn't find specific information about that topic in the available books. Could you try rephrasing your question or asking about a different aspect?"
                tokens_generated = len(self.tokenizer.encode(response))
                timing_data.update({
                    'generation_start': time.time(),
                    'generation_end': time.time(),
                    'generation_duration': 0.001,
                    'tokens_generated': tokens_generated,
                    'tokens_per_second': 0
                })
            else:
                print(f"✅ Found {len(relevant_chunks)} relevant chunks")
                
                # Show sources
                print("📚 Sources:")
                for i, chunk in enumerate(relevant_chunks, 1):
                    print(f"   {i}. {chunk['source']} (Page {chunk['page']}) - Relevance: {chunk['score']:.3f}")
                
                # Create prompt and generate response
                prompt = self.format_prompt(question, relevant_chunks)
                response = self.generate_response(prompt, timing_data)
            
            print()  # Add newline after streaming response
            
            # Finalize metrics and display summary
            final_metrics = self.metrics.finalize_query_metrics(timing_data, question, response)
            self.metrics.print_query_summary(final_metrics)
            
            return response
            
        except Exception as e:
            print(f"\n❌ Error during chat: {e}")
            return f"I encountered an error while processing your question: {e}"
    
    def chat_once(self, question):
        """Run a single chat query for testing purposes"""
        print(f"\n👤 User: {question}")
        
        # Start timing
        timing_data = self.metrics.start_query_timing()
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, timing_data)
        
        # Format prompt
        prompt = self.format_prompt(question, relevant_chunks)
        
        # Generate response
        response = self.generate_response(prompt, timing_data)
        
        # Record metrics
        self.metrics.finalize_query_metrics(timing_data, question, response)
        
        return response

def main():
    print("📖 Initializing Book Chat System...")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        bot = BookChatBot()
        
        # Load models and retrieval system
        bot.load_models()
        bot.load_retrieval_system()
        
        print("\n🎉 Book Chat System Ready!")
        print("=" * 60)
        print("💬 You can now ask questions about your remote sensing books.")
        print("💡 Examples:")
        print("   - 'What are the basic principles of remote sensing?'")
        print("   - 'Explain electromagnetic spectrum in remote sensing'")
        print("   - 'How does satellite imagery work?'")
        print("📝 Special commands:")
        print("   - 'metrics' - Show session performance summary")
        print("   - 'reset' - Reset performance metrics")
        print("   - 'quit' or 'exit' - End conversation")
        print("=" * 60)
        
        # Interactive chat loop
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    # Show final session summary
                    print("\n� Final Session Summary:")
                    print("=" * 60)
                    summary = bot.metrics.get_session_summary()
                    for key, value in summary.items():
                        if isinstance(value, float):
                            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                    print("=" * 60)
                    print("�👋 Goodbye! Happy learning!")
                    break
                
                if question.lower() == 'metrics':
                    print("\n📊 Session Performance Summary:")
                    print("=" * 60)
                    summary = bot.metrics.get_session_summary()
                    for key, value in summary.items():
                        if key == 'current_system_metrics':
                            print("   System Resources:")
                            for skey, svalue in value.items():
                                if isinstance(svalue, float):
                                    print(f"     {skey.replace('_', ' ').title()}: {svalue:.2f}")
                                else:
                                    print(f"     {skey.replace('_', ' ').title()}: {svalue}")
                        elif isinstance(value, float):
                            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                    print("=" * 60)
                    continue
                
                if question.lower() == 'reset':
                    bot.metrics.reset_session_metrics()
                    print("✅ Performance metrics reset!")
                    continue
                
                if not question:
                    print("⚠️ Please enter a question.")
                    continue
                
                # Process the question
                bot.chat(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Failed to initialize chat system: {e}")
        print("💡 Make sure you have:")
        print("   1. Run 'python prepare_data.py' to create chunks")
        print("   2. Run 'python build_faiss.py' to create search index")
        print("   3. Run 'python finetune_qlora.py' to create fine-tuned model")

if __name__ == "__main__":
    main()
