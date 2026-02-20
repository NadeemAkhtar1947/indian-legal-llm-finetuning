#!/usr/bin/env python3
"""
Correct extraction - get FIRST assistant response only
"""

import json
import re

def extract_first_assistant_response(text):
    """Extract only the FIRST assistant response, stopping at next role"""
    
    # Find "assistant" (with or without newline/space after)
    match = re.search(r'assistant[\s\n]', text)
    if not match:
        return text.strip()
    
    # Start from after "assistant"
    start_pos = match.end()
    response = text[start_pos:]
    
    # Find the EARLIEST occurrence of ANY next role tag
    # Important: "assistant" can appear WITH or WITHOUT preceding newline
    next_role = float('inf')
    
    patterns = [
        r'\nuser',
        r'\nassistant',
        r'\nsystem',
        r'user[A-Z]',      # userCLICK
        r'(?<=[.!?])\s*assistant',  # assistant after punctuation
        r'(?<=[a-z])\s*assistant',  # assistant after lowercase letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match and match.start() < next_role:
            next_role = match.start()
    
    # Cut at earliest role marker
    if next_role != float('inf'):
        response = response[:next_role]
    
    # Clean up
    response = response.strip()
    response = re.sub(r'<\|.*?\|>', '', response)
    
    return response

print("EXTRACTING PREDICTIONS CORRECTLY")
print()

print("Processing baseline...")
with open('results/baseline_llama_results.json', 'r') as f:
    baseline = json.load(f)

for result in baseline:
    result['prediction'] = extract_first_assistant_response(result['prediction'])

with open('results/baseline_llama_results_fixed.json', 'w') as f:
    json.dump(baseline, f, indent=2)
print(f"✓ Fixed {len(baseline)} baseline predictions")

print("\nProcessing fine-tuned...")
with open('results/finetuned_llama_results.json', 'r') as f:
    finetuned = json.load(f)

for result in finetuned:
    result['prediction'] = extract_first_assistant_response(result['prediction'])

with open('results/finetuned_llama_results_fixed.json', 'w') as f:
    json.dump(finetuned, f, indent=2)
print(f"✓ Fixed {len(finetuned)} fine-tuned predictions")

# Show examples

print("BASELINE PREDICTION:")
print(baseline[0]['prediction'])
print()

print("FINE-TUNED PREDICTION:")
print(finetuned[0]['prediction'])
print()

print("GROUND TRUTH:")
print(baseline[0]['ground_truth'])
print()

# Check lengths
print(f"Baseline avg length: {sum(len(r['prediction']) for r in baseline) / len(baseline):.0f} chars")
print(f"Fine-tuned avg length: {sum(len(r['prediction']) for r in finetuned) / len(finetuned):.0f} chars")
print(f"Ground truth avg length: {sum(len(r['ground_truth']) for r in baseline) / len(baseline):.0f} chars")
