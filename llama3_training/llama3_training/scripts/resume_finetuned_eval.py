#!/usr/bin/env python3
"""
Resume Fine-tuned Evaluation from Checkpoint
Only processes remaining examples
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm

print("RESUMING FINE-TUNED EVALUATION")

# Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LORA_MODEL = "/Data2/ds_24901720/Nadeem/llama3_training/models/llama3_improved"
TEST_DATA = "/Data2/ds_24901720/Nadeem/indian_legal_extraction/data/processed/training/test_llama_format.json"
OUTPUT_FILE = "/Data2/ds_24901720/Nadeem/llama3_training/results/finetuned_llama_results.json"
BATCH_SIZE = 4

# Load existing results
print("Loading existing results...")
with open(OUTPUT_FILE, 'r') as f:
    results = json.load(f)
print(f"Found {len(results)} existing results")
print()

# Load test data
print("Loading test data...")
test_dataset = load_dataset('json', data_files=TEST_DATA, split='train')
print(f"Total test examples: {len(test_dataset)}")
print(f"Remaining: {len(test_dataset) - len(results)}")
print()

if len(results) >= len(test_dataset):
    print("ALREADY COMPLETE!")
    exit(0)

# GPU info
print("GPU INFORMATION:")
print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Tokenizer loaded")
print()

# Load model
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print("Model ready")
print()

# Batch generation
def generate_batch(prompts, max_new_tokens=512):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
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
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    cleaned_responses = []
    for response in responses:
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        cleaned_responses.append(response.strip())
    
    return cleaned_responses

# Process remaining examples
print(f"PROCESSING REMAINING {len(test_dataset) - len(results)} EXAMPLES")
print(f"Batch size: {BATCH_SIZE}")

start_idx = len(results)
remaining_dataset = test_dataset.select(range(start_idx, len(test_dataset)))

batch_prompts = []
batch_examples = []

for i, example in enumerate(tqdm(remaining_dataset, desc="Completing evaluation")):
    batch_prompts.append(example['text'])
    batch_examples.append(example)
    
    if len(batch_prompts) == BATCH_SIZE or i == len(remaining_dataset) - 1:
        predictions = generate_batch(batch_prompts)
        
        for ex, pred in zip(batch_examples, predictions):
            result = {
                'id': len(results),
                'task_type': ex['task_type'],
                'instruction': ex['original_instruction'],
                'input': ex['original_input'],
                'ground_truth': ex['original_output'],
                'prediction': pred
            }
            results.append(result)
        
        batch_prompts = []
        batch_examples = []
        
        # Save every 50 examples
        if len(results) % 50 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

# Final save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print("EVALUATION COMPLETE!")
print(f"Total examples: {len(results)}/1900")
print(f"Results saved: {OUTPUT_FILE}")
print()
print("DONE!")
