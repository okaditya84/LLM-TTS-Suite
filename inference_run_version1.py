import time
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import gc

# ========== GPU & Model Optimizations ========== #
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

# Try to use Flash Attention if available (PyTorch 2.1+)
flash_kwargs = {}
try:
    from transformers.utils import is_flash_attn_2_available
    if is_flash_attn_2_available():
        flash_kwargs = {"attn_implementation": "flash_attention_2"}
except Exception:
    pass

# Try quantization (8-bit) if available
quantize_kwargs = {}
quantization_enabled = False
try:
    from transformers import BitsAndBytesConfig
    quantize_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    quantization_enabled = True
except Exception:
    pass

# ========== Load Model & Tokenizers Once ========== #
print("🧠 Loading model and tokenizers (one-time)...")
start_time = time.time()
model = ParlerTTSForConditionalGeneration.from_pretrained(
    ".",
    torch_dtype=torch.float16 if use_amp else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    **flash_kwargs,
    **quantize_kwargs
)
# Only call .to(device) if not quantized
if not quantize_kwargs:
    model = model.to(device)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
model.eval()
prompt_tokenizer = AutoTokenizer.from_pretrained(".", use_fast=True)
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, use_fast=True)
print(f"Model+tokenizers loaded in {time.time() - start_time:.2f}s")

# ========== Inference Function ========== #
def tts_infer(prompt, description, out_wav="output.wav"):
    timing = {}
    # Pre-tokenize and cache on GPU
    start_time = time.time()
    desc = desc_tokenizer(description, return_tensors="pt", padding=True)
    desc_ids = desc["input_ids"].to(device, non_blocking=True)
    desc_mask = desc["attention_mask"].to(device, non_blocking=True)
    prompt_tok = prompt_tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_ids = prompt_tok["input_ids"].to(device, non_blocking=True)
    prompt_mask = prompt_tok["attention_mask"].to(device, non_blocking=True)
    timing['text_preprocessing'] = time.time() - start_time

    # Inference
    start_time = time.time()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                generation = model.generate(
                    input_ids=desc_ids,
                    attention_mask=desc_mask,
                    prompt_input_ids=prompt_ids,
                    prompt_attention_mask=prompt_mask,
                    do_sample=False,
                    num_beams=1,
                    max_length=512,  # Reduce for speed
                    min_length=64,   # Reduce for speed
                    use_cache=True,
                    pad_token_id=prompt_tokenizer.eos_token_id,
                    early_stopping=True,
                    length_penalty=1.0,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False
                )
        else:
            generation = model.generate(
                input_ids=desc_ids,
                attention_mask=desc_mask,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                use_cache=True,
                pad_token_id=prompt_tokenizer.eos_token_id,
                early_stopping=True,
                length_penalty=1.0,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False
            )
    torch.cuda.synchronize()
    timing['inference'] = time.time() - start_time

    # Audio processing
    start_time = time.time()
    audio = generation.cpu().numpy().squeeze()
    audio = audio.astype('float32')
    sf.write(out_wav, audio, model.config.sampling_rate)
    timing['audio_saving'] = time.time() - start_time
    return timing

# ========== Warmup (Optional) ========== #
def warmup():
    prompt = "नमस्ते, आप कैसे हैं?"
    description = "A female Hindi speaker, expressive tone, moderate speed, clear quality."
    tts_infer(prompt, description, out_wav="warmup.wav")
    torch.cuda.empty_cache()
    gc.collect()

# ========== Real-Time CLI Loop ========== #
if __name__ == "__main__":
    warmup()  # One-time warmup for CUDA
    print("\nReady for real-time TTS! Type your prompt (or 'exit' to quit):\n")
    description = "A female Hindi speaker, expressive tone, moderate speed, clear quality."
    counter = 1
    while True:
        prompt = input(f"Prompt {counter}: ").strip()
        if prompt.lower() in ["exit", "quit", "q"]:
            break
        if not prompt:
            continue
        print(f"Generating...", end=" ")
        timing = tts_infer(prompt, description, out_wav=f"output_{counter}.wav")
        print(f"Done. [Pre:{timing['text_preprocessing']:.2f}s Inf:{timing['inference']:.2f}s Save:{timing['audio_saving']:.2f}s]")
        counter += 1
    print("\nSession ended.")



# some long hindi sentence examples
"मैं एक लंबा हिंदी वाक्य बोल रहा हूँ।"
"यह एक और उदाहरण है जिसमें हम एक विस्तृत हिंदी वाक्य का उपयोग कर रहे हैं।"
"क्या आप इस लंबे हिंदी वाक्य को सुन सकते हैं जो मैं उत्पन्न कर रहा हूँ?"