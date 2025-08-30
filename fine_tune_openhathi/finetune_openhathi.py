# Add CLI args and fix bitsandbytes import
import argparse
import gc
import os
import math
import torch
import warnings
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to base model")
    parser.add_argument("--dataset_dir", required=True, help="Path to HF dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def clear_gpu_memory():
    """Clear GPU memory before loading model"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def load_model_with_quantization(model_path, use_4bit=True, use_8bit=False):
    """Load model with proper quantization config"""
    clear_gpu_memory()
    
    # Try modern BitsAndBytesConfig first
    try:
        from transformers import BitsAndBytesConfig
        
        if use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:  # 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        print(f"Loading model with {'8-bit' if use_8bit else '4-bit'} quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return model
        
    except Exception as e:
        print(f"Modern quantization failed: {e}")
        
        # Fallback to legacy kwargs
        try:
            print("Trying legacy bitsandbytes kwargs...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=use_8bit,
                load_in_4bit=use_4bit and not use_8bit,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            return model
            
        except Exception as e2:
            print(f"Legacy quantization also failed: {e2}")
            print("Loading in full precision - this will use more memory!")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            return model

def validate_and_fix_dataset(dataset, tokenizer, max_length):
    """Validate and fix dataset format issues"""
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    
    # Check first few examples
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"Example {i}: {list(example.keys())}")
        for key, value in example.items():
            print(f"  {key}: {type(value)} -> {str(value)[:100]}...")
    
    # If we have text column, tokenize it
    if "text" in dataset.column_names:
        print("Found 'text' column, tokenizing...")
        def tokenize_fn(examples):
            # Handle both single and batched examples
            texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
            result = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=True,
            )
            return result
        
        dataset = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=["text"],
            desc="Tokenizing"
        )
    
    # Validate input_ids format
    if "input_ids" in dataset.column_names:
        def validate_and_fix_tokens(example):
            input_ids = example["input_ids"]
            
            # If it's a string somehow, re-tokenize
            if isinstance(input_ids, str):
                print(f"Warning: Found string in input_ids, re-tokenizing...")
                tokens = tokenizer(input_ids, truncation=True, max_length=max_length)
                example["input_ids"] = tokens["input_ids"]
                example["attention_mask"] = tokens["attention_mask"]
            
            # Ensure it's a list of integers
            elif isinstance(input_ids, list):
                # Check if all elements are integers
                if not all(isinstance(x, int) for x in input_ids):
                    print(f"Warning: Non-integer tokens found, filtering...")
                    example["input_ids"] = [int(x) for x in input_ids if isinstance(x, (int, float))]
                
                # Ensure length constraints
                if len(example["input_ids"]) > max_length:
                    example["input_ids"] = example["input_ids"][:max_length]
                
                # Ensure attention mask matches
                if "attention_mask" not in example or len(example["attention_mask"]) != len(example["input_ids"]):
                    example["attention_mask"] = [1] * len(example["input_ids"])
            
            return example
        
        dataset = dataset.map(validate_and_fix_tokens, desc="Validating tokens")
    
    # Filter out empty examples
    def filter_empty(example):
        return len(example.get("input_ids", [])) > 10  # At least 10 tokens
    
    dataset = dataset.filter(filter_empty, desc="Filtering empty examples")
    
    print(f"Final dataset size: {len(dataset)}")
    return dataset

def main():
    args = parse_args()
    
    # Set memory optimization env vars
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer first
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load and validate dataset
    print("Loading dataset...")
    train_ds = load_from_disk(os.path.join(args.dataset_dir, "train"))
    valid_ds = load_from_disk(os.path.join(args.dataset_dir, "valid"))
    
    print("Validating training dataset...")
    train_ds = validate_and_fix_dataset(train_ds, tokenizer, args.max_seq_length)
    
    print("Validating validation dataset...")
    valid_ds = validate_and_fix_dataset(valid_ds, tokenizer, args.max_seq_length)
    
    # Set format to torch tensors
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Load model with quantization
    model = load_model_with_quantization(
        args.model_dir, 
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Auto-detect LoRA targets
    target_modules = []
    for name, module in model.named_modules():
        if any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            target_name = name.split('.')[-1]
            if target_name not in target_modules:
                target_modules.append(target_name)
    
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]
    
    print(f"LoRA target modules: {target_modules}")
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=False,  # Use bf16 instead
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=False,
    )
    
    # Data collator - handle padding properly
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # For efficiency
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()