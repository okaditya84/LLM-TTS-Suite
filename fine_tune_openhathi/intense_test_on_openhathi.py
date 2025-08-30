"""
Ultra-fast concurrent testing script for fine-tuned OpenHathi model.
Optimized for speed: batched inference + proper GPU utilization.
Target: <20 seconds for 10 queries.
"""

import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import psutil

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

class FastFineTunedTester:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Fast tester initializing on {self.device}")
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load model optimized for speed"""
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True,
            padding_side='left'  # For batched inference
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
        
        print("🔧 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
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
    
    def batched_inference(self, queries: List[str], batch_size: int = None, max_new_tokens: int = 150) -> List[Dict[str, Any]]:
        """Fast batched inference.

        Behavior:
        - Try to run a single full-batch generate (all queries at once). If that OOMs,
          progressively reduce batch size (half) until it succeeds.
        - Warm up the model with a tiny generation before measuring to reduce one-time overhead.
        - Measure timings with torch.cuda.synchronize for accuracy.
        """
        total_queries = len(queries)
        if batch_size is None:
            # attempt full batch first
            batch_size = total_queries

        all_results = []
        total_start = time.time()

        # Warm-up: run a tiny generation to reduce first-call overhead
        try:
            warm_q = queries[0:1]
            warm_inputs = self.tokenizer(warm_q, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                _ = self.model.generate(warm_inputs.input_ids, attention_mask=warm_inputs.attention_mask, max_new_tokens=1)
            # small sleep to let CUDA settle
            torch.cuda.synchronize()
        except Exception:
            pass

        # Try batches; prefer a single batch if possible
        current_bs = batch_size
        i = 0
        while i < total_queries:
            # Determine end index for this attempt
            end = min(i + current_bs, total_queries)
            batch_queries = queries[i:end]

            # Attempt generate and handle OOM by reducing batch size
            succeeded = False
            attempt_bs = current_bs
            while not succeeded:
                try:
                    print(f"⚡ Attempting batch generate for queries {i+1}-{end} (batch_size={attempt_bs})")
                    batch_start = time.time()

                    inputs = self.tokenizer(
                        batch_queries,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    input_lengths = inputs.attention_mask.sum(dim=1).cpu().numpy()

                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            top_k=40,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            num_beams=1,
                            use_cache=True,
                        )
                    torch.cuda.synchronize()

                    batch_time = time.time() - batch_start

                    # Decode and record
                    for j, (query, output, input_len) in enumerate(zip(batch_queries, outputs, input_lengths)):
                        generated_tokens = output[input_len:]
                        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        output_tokens = len(generated_tokens)
                        total_tokens = len(output)
                        # For parallel batched generation the per-query latency equals batch_time (since tokens are produced in parallel steps)
                        per_query_time = batch_time
                        tokens_per_second = output_tokens / per_query_time if per_query_time > 0 else 0

                        result = {
                            "query_index": i + j,
                            "query": query,
                            "response": response,
                            "input_tokens": self._json_serializable(int(input_len)),
                            "output_tokens": self._json_serializable(int(output_tokens)),
                            "total_tokens": self._json_serializable(int(total_tokens)),
                            "inference_time": per_query_time,
                            "tokens_per_second": self._json_serializable(tokens_per_second),
                            "batch_time": batch_time,
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(result)
                        print(f"  ✓ Query {i+j+1}: {len(response)} chars, {output_tokens} tokens, {tokens_per_second:.1f} tok/s (per-query {per_query_time:.2f}s)")

                    succeeded = True
                except RuntimeError as e:
                    # Likely OOM; reduce batch size and retry
                    msg = str(e)
                    print(f"⚠️ RuntimeError during generate: {msg}")
                    # free and reduce
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if attempt_bs <= 1:
                        raise
                    attempt_bs = max(1, attempt_bs // 2)
                    # update end and batch_queries according to new attempt_bs
                    end = min(i + attempt_bs, total_queries)
                    batch_queries = queries[i:end]
                    print(f"🔁 Retrying with smaller batch_size={attempt_bs}")

            # Advance i by number of items processed
            i = end

        total_time = time.time() - total_start
        print(f"🏁 All queries completed in {total_time:.2f}s")
        return all_results, total_time
    
    def calculate_fast_metrics(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Calculate metrics with JSON serialization fix"""
        
        inference_times = [r["inference_time"] for r in results]
        tokens_per_second = [r["tokens_per_second"] for r in results]
        input_tokens = [r["input_tokens"] for r in results]
        output_tokens = [r["output_tokens"] for r in results]
        
        metrics = {
            "summary": {
                "total_queries": len(results),
                "total_execution_time": self._json_serializable(total_time),
                "average_query_time": self._json_serializable(np.mean(inference_times)),
                "throughput_qps": self._json_serializable(len(results) / total_time),
                "total_tokens_generated": self._json_serializable(np.sum(output_tokens)),
            },
            "performance": {
                "fastest_query_time": self._json_serializable(np.min(inference_times)),
                "slowest_query_time": self._json_serializable(np.max(inference_times)),
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
    
    # Shorter, focused test queries for speed
    test_queries = [
        "What is electromagnetic spectrum?",
        "Explain satellite orbits briefly.",
        "What is SAR imaging?",
        "How does atmospheric interference affect remote sensing?",
        "Difference between active and passive remote sensing?",
        "Applications of hyperspectral imaging?",
        "What is geometric correction?",
        "How do microwave sensors work?",
        "Define spatial and spectral resolution.",
        "Advantages of LiDAR technology?"
    ]
    
    print("="*80)
    print("⚡ ULTRA-FAST FINE-TUNED OPENHATHI TESTING")
    print("="*80)
    
    # Initialize tester
    tester = FastFineTunedTester(
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH
    )
    
    # Run fast batched inference
    results, total_time = tester.batched_inference(
        test_queries, 
        batch_size=5,  # Process 5 at a time for optimal speed
        max_new_tokens=100  # Shorter responses for speed
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