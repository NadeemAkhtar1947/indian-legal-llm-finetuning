#!/usr/bin/env python3
"""
Llama 3.1-8B Fine-tuning 
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
TRAINING_DATA = "/Data2/ds_24901720/Nadeem/indian_legal_extraction/data/processed/training/train_llama_format.json"
OUTPUT_DIR = "/Data2/ds_24901720/Nadeem/llama3_training/models/llama3_improved"
CHECKPOINT_DIR = "/Data2/ds_24901720/Nadeem/llama3_training/checkpoints_improved"
LOG_DIR = "/Data2/ds_24901720/Nadeem/llama3_training/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# GPU info
print("GPU INFORMATION:")
if torch.cuda.is_available():
    print(f"  CUDA Available: Yes")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ERROR: No GPU detected!")
    exit(1)
print()

# Check data
if not os.path.exists(TRAINING_DATA):
    print(f"ERROR: {TRAINING_DATA} not found!")
    exit(1)
print("Data found")
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded")
print()

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    use_cache=False
)
print("Model loaded")
print()

# LoRA config
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"    LoRA configured")
print(f"  Total params: {total:,}")
print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
print()

# Load dataset
print("Loading dataset...")
dataset = load_dataset('json', data_files=TRAINING_DATA, split='train')
print(f" Loaded {len(dataset):,} examples")
print()

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False
    )

print("Tokenizing...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "original_instruction", "original_input", "original_output", "task_type"],
    desc="Tokenizing"
)
print(" Tokenized")
print()

# Training arguments - SAFE for 47GB
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=2,
    
    # SAFE batch size for 47GB MIG
    per_device_train_batch_size=3,   
    gradient_accumulation_steps=11,  
    
    # Learning rate
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Optimization
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Logging
    logging_dir=LOG_DIR,
    logging_steps=50,
    
    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    
    # Performance
    bf16=True,
    tf32=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,  # Saves memory
    
    # Other
    report_to="none",
    disable_tqdm=False
)

total_steps = len(dataset) * 2 // 33
print("="*70)
print("TRAINING CONFIGURATION")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("STARTING TRAINING!")

# Train
trainer.train()

print("TRAINING COMPLETE!")

# Save
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Model saved: {OUTPUT_DIR}")
