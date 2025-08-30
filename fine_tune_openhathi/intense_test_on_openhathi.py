"""
Ultra-fast concurrent testing script for fine-tuned OpenHathi model.
Optimized for speed: batched inference + proper GPU utilization.
Target: <20 seconds for 10 queries.
"""

import time
import json
import torch
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import psutil

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
import rag_utils as rag_utils

class FastFineTunedTester:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Fast tester initializing on {self.device}")
        # Load RAG components once
        self.rag = rag_utils.load_faiss_components('.')
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load model optimized for speed"""
        print("📝 Loading tokenizer...")
        # Use left padding for decoder-only models to avoid generation warnings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side='left'  # Left padding for proper decoder-only generation
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("🧠 Loading base model with optimized settings...")
        
        # Try quantized loading first
        try:
            # Lighter quantization for speed
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,  # Faster loading
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # Faster than bfloat16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("✅ Loaded with 4-bit quantization")
            
        except Exception as e:
            print(f"⚠️ Quantized loading failed: {e}")
            print("🔄 Falling back to FP16 loading...")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("✅ Loaded with FP16")

        # After loading base_model (either quantized or fp16), apply LoRA adapter if available
        print("🔧 Loading LoRA adapter...")
        try:
            from peft import PeftModel as _PeftModel
        except Exception:
            _PeftModel = None

        if _PeftModel is not None:
            try:
                self.model = _PeftModel.from_pretrained(base_model, self.adapter_path)
                print("✅ LoRA adapter applied")
            except Exception as e:
                print(f"⚠️ Failed to apply LoRA adapter: {e}")
                self.model = base_model
        else:
            print("⚠️ peft not installed; using base model without LoRA")
            self.model = base_model
        self.model.eval()
        
        # Optimize for inference
        torch.backends.cudnn.benchmark = True
        
        print(f"✅ Model ready! Memory: {self._get_gpu_memory():.1f}GB")
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def _json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _cleanup_response(self, resp: str) -> str:
        """Enhanced cleanup for generated responses to fix garbled outputs and unicode issues."""
        if not resp:
            return resp
            
        # Step 1: Aggressive unicode and artifact removal
        resp = re.sub(r'\(cid:\d+\)', '', resp)  # Remove (cid:123) patterns
        resp = re.sub(r'cid:\d+', '', resp)  # Remove standalone cid:123
        resp = re.sub(r'cid[:\[\]\(\)h\-\d]*', '', resp)  # Remove cid variants
        resp = re.sub(r'uid:\w+', '', resp)  # Remove uid patterns
        resp = re.sub(r'\\[0-9a-fA-F]{2,4}', '', resp)  # Remove hex escape sequences
        
        # Step 2: Remove problematic patterns and artifacts
        resp = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\%\&\$]', ' ', resp)  # Keep essential punctuation
        resp = re.sub(r'\s+', ' ', resp)  # Normalize whitespace
        resp = resp.strip()
        
        # Step 3: Remove common prefixes and artifacts
        prefixes_to_remove = [
            "Answer:", "A:", "Q:", "Question:", "Context:", "Reference:", 
            "You are a technical expert.", "Read the context carefully", 
            "Instructions:", "- Answer using", "- Be precise"
        ]
        for prefix in prefixes_to_remove:
            if resp.startswith(prefix):
                resp = resp[len(prefix):].strip()
                
        # Step 4: Handle responses that start with colons or fragments
        if resp.startswith(':'):
            resp = resp[1:].strip()
            
        # Step 5: Remove question repetition if present
        if '?' in resp:
            parts = resp.split('?')
            # If first part looks like a question repetition, remove it
            if len(parts) > 1 and len(parts[0].split()) < 25:
                remaining = '?'.join(parts[1:]).strip()
                if len(remaining) > 20:  # Only if substantial content remains
                    resp = remaining
        
        # Step 6: Fix responses that start mid-sentence
        sentences = re.split(r'[.!?]+', resp)
        if sentences and len(sentences) > 1:
            first_sentence = sentences[0].strip()
            # If first sentence is very short, starts lowercase, or looks fragmented
            if (len(first_sentence) < 15 or 
                (first_sentence and first_sentence[0].islower()) or
                first_sentence.count(' ') < 3):
                # Try to start from second sentence
                if len(sentences) > 1:
                    remaining_sentences = [s.strip() for s in sentences[1:] if s.strip()]
                    if remaining_sentences:
                        resp = '. '.join(remaining_sentences)
        
        # Step 7: Remove trailing incomplete fragments
        resp = re.sub(r'\s+[A-Z][^.!?]*$', '', resp)  # Remove trailing incomplete sentences
        resp = re.sub(r'\s+\w{1,3}$', '', resp)  # Remove trailing short words
        
        # Step 8: Ensure proper capitalization
        if resp and resp[0].islower():
            resp = resp[0].upper() + resp[1:] if len(resp) > 1 else resp.upper()
        
        # Step 9: Final cleanup - remove empty parentheses, brackets, etc.
        resp = re.sub(r'\(\s*\)', '', resp)
        resp = re.sub(r'\[\s*\]', '', resp)
        resp = re.sub(r'\s+', ' ', resp)
        resp = resp.strip()
                
        return resp
    
    def batched_inference(self, queries: List[str], batch_size: int = None, max_new_tokens: int = 200) -> List[Dict[str, Any]]:
        """Truly parallel batched inference - process all queries simultaneously.

        This version processes ALL queries in a single batch for maximum parallelism.
        """
        total_queries = len(queries)
        print(f"🚀 Processing {total_queries} queries in TRUE PARALLEL mode")
        
        all_results = []
        total_start = time.time()

        # Skip warmup - it's not needed for production runs and slows us down
        
        # Process ALL queries in a single batch for maximum parallelism
        try:
            batch_start = time.time()
            
            # Pre-build all prompts in parallel (vectorized operations where possible)
            print(f"⚡ Building RAG prompts for all {total_queries} queries...")
            prompt_build_start = time.time()
            
            prompts = []
            for q in queries:
                q_emb = rag_utils.embed_query(self.rag, q, model_name=self.rag.get('embed_info', {}).get('model', 'all-mpnet-base-v2'))
                retrieved = []
                if q_emb is not None:
                    retrieved = rag_utils.search_rag(self.rag, query=q, query_emb=q_emb, k=4)  # Reduced k for speed
                else:
                    retrieved = rag_utils.search_rag(self.rag, query=q, query_emb=None, k=4)
                
                prompt = rag_utils.assemble_prompt(retrieved, q, max_context_chars=1200)  # Smaller context for stability
                prompts.append(prompt)
            
            prompt_time = time.time() - prompt_build_start
            print(f"📝 Prompt building completed in {prompt_time:.2f}s")

            # Tokenize all prompts in one batch
            tokenize_start = time.time()
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1500  # Smaller for better stability and speed
            ).to(self.device)
            
            input_lengths = inputs.attention_mask.sum(dim=1).cpu().numpy()
            tokenize_time = time.time() - tokenize_start
            print(f"🔢 Tokenization completed in {tokenize_time:.2f}s")

            # Single GPU synchronization before generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            gen_start = time.time()
            print(f"⚡ Starting parallel generation for {total_queries} queries...")
            
            # Optimized generation parameters for speed + quality - explicit parameter control
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Deterministic for consistency
                "num_beams": 3,  # Slightly higher for better quality
                "repetition_penalty": 1.2,  # Stronger repetition penalty
                "early_stopping": True,
                "no_repeat_ngram_size": 3,  # Prevent repetition
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            generation_time = time.time() - gen_start
            batch_time = time.time() - batch_start
            
            print(f"⚡ Parallel generation completed in {generation_time:.2f}s (total batch: {batch_time:.2f}s)")

            # Process all results in parallel - NO individual retries to maintain speed
            decode_start = time.time()
            all_results = []
            
            for j, (query, output, input_len) in enumerate(zip(queries, outputs, input_lengths)):
                total_length = len(output)
                generated_tokens = output[input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                response = self._cleanup_response(response)

                output_tokens = len(generated_tokens)
                total_tokens = total_length
                
                # Calculate per-query metrics (all queries processed in parallel)
                per_query_time = batch_time  # Same for all since they're processed in parallel
                tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

                result = {
                    "query_index": j,
                    "query": query,
                    "response": response,
                    "input_tokens": self._json_serializable(int(input_len)),
                    "output_tokens": self._json_serializable(int(output_tokens)),
                    "total_tokens": self._json_serializable(int(total_tokens)),
                    "inference_time": per_query_time,
                    "tokens_per_second": self._json_serializable(tokens_per_second),
                    "batch_time": batch_time,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result)
                print(f"  ✓ Query {j+1}: {len(response)} chars, {output_tokens} tokens, {tokens_per_second:.1f} tok/s")
            
            decode_time = time.time() - decode_start
            print(f"📖 Decoding completed in {decode_time:.2f}s")

        except RuntimeError as e:
            print(f"❌ Failed to process all queries in parallel: {e}")
            print("🔄 Falling back to smaller batches...")
            # Fallback to smaller batches if OOM
            return self._fallback_batched_inference(queries, max_new_tokens)

        total_time = time.time() - total_start
        actual_per_query_time = total_time / total_queries  # Fix calculation
        print(f"🏁 All {total_queries} queries completed in {total_time:.2f}s (avg: {actual_per_query_time:.2f}s per query)")
        
        return all_results, total_time
    
    def _fallback_batched_inference(self, queries: List[str], max_new_tokens: int = 200) -> List[Dict[str, Any]]:
        """Fallback method when full batch fails - use smaller batches"""
        print("🔄 Using fallback batching strategy...")
        all_results = []
        total_start = time.time()
        
        # Try with half batch size
        batch_size = max(1, len(queries) // 2)
        i = 0
        while i < len(queries):
            end = min(i + batch_size, len(queries))
            batch_queries = queries[i:end]
            
            try:
                print(f"⚡ Processing queries {i+1}-{end} (fallback batch)")
                batch_start = time.time()
                
                prompts = []
                for q in batch_queries:
                    q_emb = rag_utils.embed_query(self.rag, q, model_name=self.rag.get('embed_info', {}).get('model', 'all-mpnet-base-v2'))
                    retrieved = rag_utils.search_rag(self.rag, query=q, query_emb=q_emb, k=4) if q_emb else []
                    prompt = rag_utils.assemble_prompt(retrieved, q, max_context_chars=1500)
                    prompts.append(prompt)

                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(self.device)
                input_lengths = inputs.attention_mask.sum(dim=1).cpu().numpy()

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=2,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                batch_time = time.time() - batch_start
                
                for j, (query, output, input_len) in enumerate(zip(batch_queries, outputs, input_lengths)):
                    generated_tokens = output[input_len:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    response = self._cleanup_response(response)
                    
                    result = {
                        "query_index": i + j,
                        "query": query,
                        "response": response,
                        "input_tokens": self._json_serializable(int(input_len)),
                        "output_tokens": self._json_serializable(len(generated_tokens)),
                        "total_tokens": self._json_serializable(len(output)),
                        "inference_time": batch_time,
                        "tokens_per_second": self._json_serializable(len(generated_tokens) / batch_time),
                        "batch_time": batch_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    all_results.append(result)
                    
                i = end
                
            except Exception as e:
                print(f"❌ Fallback batch failed: {e}")
                # Process individually as last resort
                for q in batch_queries:
                    try:
                        response = "Error: Could not process query"
                        result = {
                            "query_index": i,
                            "query": q,
                            "response": response,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "inference_time": 0,
                            "tokens_per_second": 0,
                            "batch_time": 0,
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(result)
                        i += 1
                    except:
                        i += 1
                        continue
        
        total_time = time.time() - total_start
        return all_results, total_time
    
    def calculate_fast_metrics(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Calculate metrics with corrected timing calculations"""
        
        # Fix: Use actual total_time instead of individual batch times for per-query average
        total_queries = len(results)
        actual_avg_query_time = total_time / total_queries if total_queries > 0 else 0
        
        tokens_per_second = [r["tokens_per_second"] for r in results]
        input_tokens = [r["input_tokens"] for r in results]
        output_tokens = [r["output_tokens"] for r in results]
        
        metrics = {
            "summary": {
                "total_queries": total_queries,
                "total_execution_time": self._json_serializable(total_time),
                "average_query_time": self._json_serializable(actual_avg_query_time),  # Corrected calculation
                "throughput_qps": self._json_serializable(total_queries / total_time if total_time > 0 else 0),
                "total_tokens_generated": self._json_serializable(np.sum(output_tokens)),
            },
            "performance": {
                "fastest_query_time": self._json_serializable(actual_avg_query_time),  # All queries processed in parallel
                "slowest_query_time": self._json_serializable(actual_avg_query_time),   # All queries processed in parallel
                "avg_tokens_per_second": self._json_serializable(np.mean(tokens_per_second)),
                "max_tokens_per_second": self._json_serializable(np.max(tokens_per_second)),
                "avg_response_length": self._json_serializable(np.mean(output_tokens)),
            },
            "model_info": {
                "base_model": self.base_model_path,
                "adapter_path": self.adapter_path,
                "device": self.device,
                "gpu_memory_gb": self._json_serializable(self._get_gpu_memory()),
            }
        }
        
        return metrics

def main():
    # Configuration
    BASE_MODEL_PATH = "/nlsasfs/home/ledgerptf/ashsa/models/openhathi/OpenHathi-7B-Hi-v0.1-Base/"
    ADAPTER_PATH = "/nlsasfs/home/ledgerptf/ashsa/models/openhathi/fine_tuned_openhathi/"
    OUTPUT_FILE = "fast_test_results.json"
    
    # Better focused test queries for improved response quality
    test_queries = [
        "What is the electromagnetic spectrum and how do wavelength and frequency relate to photon energy?",
        "Compare geostationary, medium Earth, and low Earth orbits in terms of altitude, velocity, and applications.",
        "Explain the basic principles of synthetic aperture radar imaging and how it improves resolution.",
        "How does atmospheric interference affect remote sensing measurements at different wavelengths?",
        "What are the key differences between active and passive remote sensing systems?",
        "Describe the applications of hyperspectral imaging in precision agriculture and environmental monitoring.",
        "What is geometric correction in satellite imagery and why is it important?",
        "How do microwave remote sensing instruments work and what can they measure?",
        "Define spatial and spectral resolution and explain their importance in remote sensing.",
        "What are the advantages and limitations of LiDAR technology for mapping applications?"
    ]
    
    print("="*80)
    print("⚡ ULTRA-FAST FINE-TUNED OPENHATHI TESTING")
    print("="*80)
    
    # Initialize tester
    tester = FastFineTunedTester(
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH
    )
    
    # Run truly parallel batched inference - ALL queries at once
    print("🚀 Starting TRULY PARALLEL inference (all 10 queries simultaneously)...")
    results, total_time = tester.batched_inference(
        test_queries, 
        batch_size=len(test_queries),  # Process ALL queries in one batch
        max_new_tokens=200  # Maintain quality with sufficient tokens
    )
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = tester.calculate_fast_metrics(results, total_time)
    
    # Prepare output
    output_data = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "fast_batched_inference",
            "model_type": "fine_tuned_openhathi_lora",
            "optimization": "batched_inference_fp16"
        },
        "results": results,
        "metrics": metrics
    }
    
    # Save results with JSON serialization fix
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=tester._json_serializable)
        print(f"💾 Results saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        # Fallback: save metrics only
        with open("metrics_only.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=tester._json_serializable)
    
    # Display results
    print("\n" + "="*80)
    print("⚡ FAST TEST RESULTS")
    print("="*80)
    print(f"✅ Queries: {metrics['summary']['total_queries']}")
    print(f"⏱️  Total Time: {metrics['summary']['total_execution_time']:.2f}s")
    print(f"⚡ Avg per Query: {metrics['summary']['average_query_time']:.2f}s")
    print(f"🚀 Throughput: {metrics['summary']['throughput_qps']:.2f} queries/sec")
    print(f"📝 Avg Tokens/sec: {metrics['performance']['avg_tokens_per_second']:.1f}")
    print(f"💾 GPU Memory: {metrics['model_info']['gpu_memory_gb']:.1f}GB")
    
    # Show fastest vs slowest
    print(f"\n⚡ Fastest Query: {metrics['performance']['fastest_query_time']:.2f}s")
    print(f"🐌 Slowest Query: {metrics['performance']['slowest_query_time']:.2f}s")
    
    # Sample responses
    print(f"\n📝 Sample Responses:")
    print("-" * 60)
    for i, result in enumerate(results[:3]):
        print(f"\n🔸 Q{i+1}: {result['query']}")
        print(f"🔹 A: {result['response'][:150]}...")
        print(f"   ⏱️ {result['inference_time']:.2f}s | 🚀 {result['tokens_per_second']:.1f} tok/s")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()