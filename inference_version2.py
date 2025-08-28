#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

class OptimizedParlerTTS:
    def __init__(self, model_path=".", device="cuda"):
        self.device = device
        self.model = None
        self.prompt_tok = None
        self.desc_tok = None
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load model once during initialization"""
        print("Loading model (one-time setup)...")
        
        # Optimize CUDA settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Load model with optimizations
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # Move to device and optimize
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizers
        self.prompt_tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.desc_tok = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path, local_files_only=True
        )
        
        # Compile model (one-time cost)
        print("Compiling model...")
        self.model = torch.compile(self.model, mode="max-autotune")
        
        # Warmup inference to avoid first-call overhead
        self._warmup()
        
    def _warmup(self):
        """Warmup the model with a dummy inference"""
        print("Warming up model...")
        dummy_prompt = "test"
        dummy_desc = "A speaker"
        
        with torch.inference_mode():
            p = self.prompt_tok(dummy_prompt, return_tensors="pt", padding=True)
            p = {k: v.to(self.device) for k, v in p.items()}
            d = self.desc_tok(dummy_desc, return_tensors="pt", padding=True)
            d = {k: v.to(self.device) for k, v in d.items()}
            
            # Dummy inference
            _ = self.model.generate(
                input_ids=d["input_ids"],
                attention_mask=d["attention_mask"],
                prompt_input_ids=p["input_ids"],
                prompt_attention_mask=p["attention_mask"],
                max_length=100  # Short for warmup
            )
        torch.cuda.synchronize()
        print("Warmup complete!")
    
    def generate_speech(self, prompt, description, output_file="output.wav"):
        """Fast inference after model is loaded"""
        start_total = time.time()
        
        # Tokenize inputs
        t1 = time.time()
        p = self.prompt_tok(prompt, return_tensors="pt", padding=True)
        p = {k: v.to(self.device) for k, v in p.items()}
        d = self.desc_tok(description, return_tensors="pt", padding=True)
        d = {k: v.to(self.device) for k, v in d.items()}
        tokenization_time = time.time() - t1
        
        # GPU inference with timing
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        with torch.inference_mode():
            wav = self.model.generate(
                input_ids=d["input_ids"],
                attention_mask=d["attention_mask"],
                prompt_input_ids=p["input_ids"],
                prompt_attention_mask=p["attention_mask"],
                do_sample=False,  # Deterministic, faster
                num_beams=1,      # No beam search
                pad_token_id=self.desc_tok.pad_token_id
            )
        end_evt.record()
        torch.cuda.synchronize()
        inference_time_ms = start_evt.elapsed_time(end_evt)
        
        # Save audio
        t2 = time.time()
        audio = wav.cpu().float().numpy().squeeze().astype(np.float32)
        sf.write(output_file, audio, self.model.config.sampling_rate)
        save_time = time.time() - t2
        
        total_time = time.time() - start_total
        
        # Report
        print(f"\n===== Fast Inference Report =====")
        print(f"Tokenization:     {tokenization_time*1000:.1f} ms")
        print(f"GPU Inference:    {inference_time_ms:.1f} ms")
        print(f"Audio save:       {save_time*1000:.1f} ms")
        print(f"TOTAL TIME:       {total_time*1000:.1f} ms")
        print(f"Audio duration:   {len(audio)/self.model.config.sampling_rate:.3f} s")
        print(f"Saved to:         {os.path.abspath(output_file)}")
        
        return audio, total_time

# ──── Usage ────────────────────────────────────────────────────────────────
def run_server():
    """Keep model loaded and accept multiple prompts"""
    print("Initializing TTS server...")
    tts_engine = OptimizedParlerTTS(".")
    
    print("\n🚀 TTS Server ready! Model is loaded and waiting...")
    print("Enter prompts (or 'quit' to exit):")
    
    counter = 1
    while True:
        prompt = input(f"\nPrompt {counter}: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
            
        if prompt:
            description = "A female Hindi speaker, expressive tone, moderate speed, clear quality."
            output_file = f"output_{counter}.wav"
            
            audio, inference_time = tts_engine.generate_speech(prompt, description, output_file)
            counter += 1

if __name__ == "__main__":
    run_server()
