#!/usr/bin/env python3
"""
Evaluate Fine-tuned Llama 3.1-8B on Indian Legal Test Data
Evaluates by task categories and saves detailed results
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np

print("FINE-TUNED LLAMA 3.1-8B EVALUATION")

# Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LORA_MODEL = "/Data2/ds_24901720/Nadeem/llama3_training/models/llama3_improved"
TEST_DATA = "/Data2/ds_24901720/Nadeem/indian_legal_extraction/data/processed/training/test_llama_format.json"
OUTPUT_FILE = "/Data2/ds_24901720/Nadeem/llama3_training/results/finetuned_llama_results.json"

# GPU info
print("GPU INFORMATION:")
if torch.cuda.is_available():
    print(f"  CUDA Available: Yes")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ERROR: No GPU!")
    exit(1)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print(" Tokenizer loaded")
print()

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print(" Base model loaded")
print()

# Load LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print(" Fine-tuned model ready")
print()

# Load test data
print("Loading test data...")
test_dataset = load_dataset('json', data_files=TEST_DATA, split='train')
print(f" Loaded {len(test_dataset)} test examples")
print()

# Get task distribution
task_counts = defaultdict(int)
for example in test_dataset:
    task_counts[example['task_type']] += 1

print("Test data by task:")
for task, count in sorted(task_counts.items()):
    print(f"  {task}: {count}")
print()

# Evaluation function
def generate_response(prompt, max_new_tokens=512):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (after the prompt)
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    
    return response.strip()

# Run evaluation
print()

results = []
task_results = defaultdict(list)

for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
    # Extract prompt from formatted text
    prompt = example['text']
    
    # Generate response
    prediction = generate_response(prompt)
    
    # Store result
    result = {
        'id': i,
        'task_type': example['task_type'],
        'instruction': example['original_instruction'],
        'input': example['original_input'],
        'ground_truth': example['original_output'],
        'prediction': prediction
    }
    
    results.append(result)
    task_results[example['task_type']].append(result)
    
    # Save periodically
    if (i + 1) % 100 == 0:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved checkpoint at {i+1} examples")

# Final save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print()

# Summary
print(f"Total examples evaluated: {len(results)}")
print(f"Results saved to: {OUTPUT_FILE}")
print()

print("Examples by task:")
for task, examples in sorted(task_results.items()):
    print(f"  {task}: {len(examples)}")
print()
print("DONE!")
