#!/usr/bin/env python3
"""
Evaluate Fine-tuned Llama 3.1-8B on Test Dataset
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
import re

print("Evaluating Fine-tuned Llama 3.1-8B")

# Paths
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
ADAPTER_PATH = "/Data2/ds_24901720/Nadeem/llama3_training/models/llama3_legal_final"
TEST_DATA = "/Data2/ds_24901720/Nadeem/indian_legal_extraction/data/processed/training/test.json"
OUTPUT_FILE = "/Data2/ds_24901720/Nadeem/llama3_training/evaluation_results.json"

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True
)

# Load LoRA adapter
print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("Model loaded successfully")
print(f"Model device: {model.device}")

# Load test data
print(f"\nLoading test data from {TEST_DATA}")
test_dataset = load_dataset('json', data_files=TEST_DATA, split='train')
print(f"Test samples: {len(test_dataset)}")

# Sample 100 examples for quick evaluation (change to full dataset later)
test_dataset = test_dataset.select(range(min(100, len(test_dataset))))
print(f"Evaluating on {len(test_dataset)} samples")

# Evaluation function
def generate_response(instruction, input_text=""):
    """Generate model response"""
    if input_text and input_text.strip():
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Run evaluation
print("STARTING EVALUATION")

results = []
for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
    try:
        instruction = example['instruction']
        input_text = example['input']
        expected_output = example['output']
        
        # Generate prediction
        prediction = generate_response(instruction, input_text)
        
        results.append({
            'id': i,
            'instruction': instruction,
            'input': input_text,
            'expected': expected_output,
            'predicted': prediction
        })
        
        # Show first 3 examples
        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Instruction: {instruction[:100]}...")
            print(f"Expected: {expected_output[:100]}...")
            print(f"Predicted: {prediction[:100]}...")
    
    except Exception as e:
        print(f"Error on example {i}: {e}")
        continue

# Save results
print(f"\nSaving results to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print("EVALUATION COMPLETE!")
print(f"Total examples evaluated: {len(results)}")
print(f"Results saved to: {OUTPUT_FILE}")
