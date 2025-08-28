# finetune_qlora.py
import os, math, torch, warnings
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== CONFIG =====
BASE_MODEL_PATH = "/nlsasfs/home/ledgerptf/ashsa/models/sarvam1"  # <== your local model
DATA_DIR = "hf_dataset"
OUTPUT_DIR = "fine_tuned_sarvam_books"
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 5  # Increase from 3 to 5 epochs
BATCH_SIZE = 2
GRAD_ACC = 8
LEARNING_RATE = 1e-4  # Reduce from 2e-4 to 1e-4 for more stable training
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
FP16 = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| cuda_available:", torch.cuda.is_available())

# ensure tokenizers doesn't try to use parallelism after forking (silences warning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ===== Load dataset =====
train_ds = load_from_disk(os.path.join(DATA_DIR, "train"))
valid_ds = load_from_disk(os.path.join(DATA_DIR, "valid"))

# ===== Tokenizer =====
try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )
    print("Loaded tokenizer via AutoTokenizer.")
except Exception as e:
    print("AutoTokenizer failed, trying PreTrainedTokenizerFast fallback:", e)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(BASE_MODEL_PATH, "tokenizer.json"),
        use_fast=True,
    )
    print("Loaded tokenizer via PreTrainedTokenizerFast.")

# Debug tokenizer properties
print(f"Tokenizer vocab_size: {getattr(tokenizer, 'vocab_size', 'Not available')}")
print(f"Tokenizer len: {len(tokenizer) if hasattr(tokenizer, '__len__') else 'Not available'}")
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token_id: {getattr(tokenizer, 'eos_token_id', 'Not available')}")

# Set pad token to eos token if available, but don't add new tokens
if tokenizer.pad_token_id is None:
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("PAD token set to EOS token ID:", tokenizer.pad_token_id)
    else:
        # Use the last token in vocab as pad token to avoid expanding vocab
        vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        tokenizer.pad_token_id = vocab_size - 1
        print("PAD token set to last vocab token:", tokenizer.pad_token_id)

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

def clamp_token_ids(example):
    """Clamp any out-of-bounds token IDs to prevent CUDA assertion errors"""
    # Get actual vocab size from tokenizer or model
    vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
    input_ids = example["input_ids"]
    
    # Check for out-of-bounds tokens and clamp them
    max_id = max(input_ids) if input_ids else 0
    min_id = min(input_ids) if input_ids else 0
    
    if max_id >= vocab_size or min_id < 0:
        print(f"WARNING: Found token IDs out of range [0, {vocab_size-1}]: min={min_id}, max={max_id}")
        # Clamp all token IDs to valid range
        input_ids = [max(0, min(token_id, vocab_size-1)) for token_id in input_ids]
        example["input_ids"] = input_ids
        print(f"Clamped to range [0, {vocab_size-1}]")
    
    return example

if "input_ids" not in train_ds.column_names:
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds = train_ds.map(clamp_token_ids, batched=False)
if "input_ids" not in valid_ds.column_names:
    valid_ds = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    valid_ds = valid_ds.map(clamp_token_ids, batched=False)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ===== Load model with robust k-bit fallback =====
bnb_ok = True
try:
    from bitsandbytes.cuda_setup.main import get_compute_capability
    cc = get_compute_capability()
    print("bitsandbytes CC:", cc)
except Exception as e:
    print("bitsandbytes check failed:", e)
    bnb_ok = False

def load_model():
    if bnb_ok:
        try:
            print("Trying 4-bit load...")
            return AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                local_files_only=True,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                trust_remote_code=True,
            )
        except Exception as e:
            print("4-bit failed:", str(e))
            print("Trying 8-bit load...")
            try:
                return AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_PATH,
                    local_files_only=True,
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            except Exception as e2:
                print("8-bit failed:", str(e2))
    print("Falling back to fp16 full-weight load on GPU...")
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

print("Loading base model...")
model = load_model()

# Expand embeddings if we added PAD
if tokenizer.vocab_size != model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(len(tokenizer))

# Make sure pad token exists; if not, add a dedicated '[PAD]' token to avoid collisions
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("PAD token added, id:", tokenizer.pad_token_id)

# Get the actual vocab size from model embeddings (most reliable)
model_vocab_size = model.get_input_embeddings().weight.shape[0]
tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))

print(f"Model embedding vocab size: {model_vocab_size}")
print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

# Use the smaller of the two to be safe
actual_vocab_size = min(model_vocab_size, tokenizer_vocab_size)
print(f"Using vocab size: {actual_vocab_size}")

# Override tokenizer vocab_size if needed
if not hasattr(tokenizer, 'vocab_size') or tokenizer.vocab_size != actual_vocab_size:
    tokenizer.vocab_size = actual_vocab_size
    print(f"Set tokenizer.vocab_size to {actual_vocab_size}")

# Determine a robust desired vocab size for resizing embeddings. Prefer tokenizer.vocab_size
if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size is not None:
    desired_vocab = tokenizer.vocab_size
else:
    try:
        desired_vocab = len(tokenizer)
    except Exception:
        desired_vocab = None

if desired_vocab is not None:
    cur_emb = model.get_input_embeddings().weight.shape[0]
    if cur_emb != desired_vocab:
        print(f"Resizing model embeddings: {cur_emb} -> {desired_vocab}")
        model.resize_token_embeddings(desired_vocab)
    else:
        print("Model embeddings size matches tokenizer vocab:", cur_emb)
else:
    print("Could not determine tokenizer vocab size; skipping resize (may cause runtime errors)")

# ===== LoRA target auto-detect =====
CANDIDATES = [
    "q_proj","k_proj","v_proj","o_proj",
    "query_key_value","dense","fc","fc_in","fc_out",
    "down_proj","up_proj","gate_proj","wi","wo","w2","w1"
]
present = set()
for n, m in model.named_modules():
    for c in CANDIDATES:
        if c in n:
            present.add(c)
target_modules = sorted(present) if present else ["q_proj","v_proj","k_proj","o_proj"]
print("LoRA target modules:", target_modules)

# Prepare for k-bit training (handles 4/8/fp16 cases)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,  # Increase r from 8 to 16, dropout from 0.05 to 0.1
    bias="none", task_type="CAUSAL_LM",
    target_modules=target_modules
)
model = get_peft_model(model, lora)

# ===== Data collator =====
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== Training args =====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    fp16=FP16,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,  # Increase warmup from 0.03 to 0.05
    weight_decay=0.01,  # Add weight decay for regularization
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,
    # Older Transformers builds require eval/save strategies to match when this is True.
    # Disable automatic best-model loading to keep compatibility with offline/server setup.
    load_best_model_at_end=False,
    report_to="none",
    dataloader_num_workers=4,
    optim="paged_adamw_8bit" if bnb_ok else "adamw_torch",
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collator,
    tokenizer=tokenizer,
)

print("Starting training…")
trainer.train()

print("Saving adapter + tokenizer to:", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Final evaluation…")
res = trainer.evaluate()
eval_loss = res.get("eval_loss")
if eval_loss is not None:
    ppl = math.exp(eval_loss)
    print(f"Eval loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")
else:
    print("No eval_loss in evaluation dict:", res)

print("Done.")
