#!/usr/bin/env python3
"""
Fine-tuned Llama Evaluation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
from collections import defaultdict

print("FAST FINE-TUNED LLAMA 3.1-8B")


# Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LORA_MODEL = "/Data2/ds_24901720/Nadeem/llama3_training/models/llama3_improved"
TEST_DATA = "/Data2/ds_24901720/Nadeem/indian_legal_extraction/data/processed/training/test_llama_format.json"
OUTPUT_FILE = "/Data2/ds_24901720/Nadeem/llama3_training/results/finetuned_llama_results.json"
BATCH_SIZE = 2  

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
tokenizer.padding_side = "left"  # For batch generation
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

# Batch generation function
def generate_batch(prompts, max_new_tokens=512):
    """Generate responses for a batch of prompts"""
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
    
    # Extract assistant responses
    cleaned_responses = []
    for response in responses:
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        cleaned_responses.append(response.strip())
    
    return cleaned_responses

# Run evaluation with batching
print("STARTING EVALUATION")
print(f"Batch size: {BATCH_SIZE}")

results = []
batch_prompts = []
batch_examples = []

for i, example in enumerate(tqdm(test_dataset, desc="Evaluating Fine-tuned")):
    batch_prompts.append(example['text'])
    batch_examples.append(example)
    
    # Process batch when full or at end
    if len(batch_prompts) == BATCH_SIZE or i == len(test_dataset) - 1:
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
        
        # Clear batch
        batch_prompts = []
        batch_examples = []
        
        # Save checkpoint every 100 examples
        if len(results) % 100 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

# Final save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print("EVALUATION COMPLETE!")

print(f"Total examples evaluated: {len(results)}")
print(f"Results saved to: {OUTPUT_FILE}")
print()

# Summary by task
task_results = defaultdict(list)
for result in results:
    task_results[result['task_type']].append(result)

print("Results by task:")
for task, examples in sorted(task_results.items()):
    print(f"  {task}: {len(examples)}")
print()

print("DONE!")
