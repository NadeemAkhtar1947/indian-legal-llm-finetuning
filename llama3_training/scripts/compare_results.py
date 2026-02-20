#!/usr/bin/env python3
"""
Compare Baseline vs Fine-tuned Llama Results
Calculates F1, ROUGE, BLEU scores by task category
"""

import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
import re

print("BASELINE vs FINE-TUNED LLAMA COMPARISON")


# Load results
print("Loading results...")
with open('results/baseline_llama_results_fixed.json', 'r') as f:
    baseline_results = json.load(f)
print(f"Baseline: {len(baseline_results)} examples")

with open('results/finetuned_llama_results_fixed.json', 'r') as f:
    finetuned_results = json.load(f)

# Group by task
baseline_by_task = defaultdict(list)
finetuned_by_task = defaultdict(list)

for result in baseline_results:
    baseline_by_task[result['task_type']].append(result)

for result in finetuned_results:
    finetuned_by_task[result['task_type']].append(result)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Simple token-based F1 for text
def calculate_text_f1(predictions, ground_truths):
    """Calculate token-level F1 score"""
    scores = []
    for pred, truth in zip(predictions, ground_truths):
        pred_tokens = set(str(pred).lower().split())
        truth_tokens = set(str(truth).lower().split())
        
        if len(truth_tokens) == 0:
            scores.append(0.0)
            continue
            
        common = pred_tokens & truth_tokens
        if len(common) == 0:
            scores.append(0.0)
        else:
            precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = len(common) / len(truth_tokens)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)
    
    return sum(scores) / len(scores) if scores else 0.0

# Calculate ROUGE-L for each task
def calculate_rouge_l(predictions, ground_truths):
    """Calculate average ROUGE-L score"""
    scores = []
    for pred, truth in zip(predictions, ground_truths):
        score = scorer.score(str(truth), str(pred))
        scores.append(score['rougeL'].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0

# Evaluate each task
print("CALCULATING METRICS BY TASK")

comparison_data = []

for task in sorted(baseline_by_task.keys()):
    print(f"\n{task.upper()}")
    print("-" * 50)
    
    baseline_examples = baseline_by_task[task]
    finetuned_examples = finetuned_by_task[task]
    
    # Extract predictions and ground truths
    baseline_preds = [ex['prediction'] for ex in baseline_examples]
    baseline_truths = [ex['ground_truth'] for ex in baseline_examples]
    
    finetuned_preds = [ex['prediction'] for ex in finetuned_examples]
    finetuned_truths = [ex['ground_truth'] for ex in finetuned_examples]
    
    # Calculate F1
    baseline_f1 = calculate_text_f1(baseline_preds, baseline_truths)
    finetuned_f1 = calculate_text_f1(finetuned_preds, finetuned_truths)
    
    # Calculate ROUGE-L
    baseline_rouge = calculate_rouge_l(baseline_preds, baseline_truths)
    finetuned_rouge = calculate_rouge_l(finetuned_preds, finetuned_truths)
    
    # Calculate improvements
    f1_improvement = ((finetuned_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    rouge_improvement = ((finetuned_rouge - baseline_rouge) / baseline_rouge * 100) if baseline_rouge > 0 else 0
    
    print(f"  Examples: {len(baseline_examples)}")
    print(f"  F1 Score:")
    print(f"    Baseline:    {baseline_f1:.4f}")
    print(f"    Fine-tuned:  {finetuned_f1:.4f}")
    print(f"    Improvement: {f1_improvement:+.1f}%")
    print(f"  ROUGE-L:")
    print(f"    Baseline:    {baseline_rouge:.4f}")
    print(f"    Fine-tuned:  {finetuned_rouge:.4f}")
    print(f"    Improvement: {rouge_improvement:+.1f}%")
    
    comparison_data.append({
        'Task': task,
        'Count': len(baseline_examples),
        'Baseline_F1': baseline_f1,
        'Finetuned_F1': finetuned_f1,
        'F1_Improvement_%': f1_improvement,
        'Baseline_ROUGE-L': baseline_rouge,
        'Finetuned_ROUGE-L': finetuned_rouge,
        'ROUGE_Improvement_%': rouge_improvement
    })

# Create summary DataFrame
df = pd.DataFrame(comparison_data)

# Calculate overall averages (weighted by count)
total_examples = df['Count'].sum()
df['Weight'] = df['Count'] / total_examples

overall_baseline_f1 = (df['Baseline_F1'] * df['Weight']).sum()
overall_finetuned_f1 = (df['Finetuned_F1'] * df['Weight']).sum()
overall_f1_improvement = ((overall_finetuned_f1 - overall_baseline_f1) / overall_baseline_f1 * 100) if overall_baseline_f1 > 0 else 0

overall_baseline_rouge = (df['Baseline_ROUGE-L'] * df['Weight']).sum()
overall_finetuned_rouge = (df['Finetuned_ROUGE-L'] * df['Weight']).sum()
overall_rouge_improvement = ((overall_finetuned_rouge - overall_baseline_rouge) / overall_baseline_rouge * 100) if overall_baseline_rouge > 0 else 0

print()
print("OVERALL SUMMARY")
print(f"Total examples: {total_examples}")
print()
print(f"F1 Score:")
print(f"  Baseline:    {overall_baseline_f1:.4f}")
print(f"  Fine-tuned:  {overall_finetuned_f1:.4f}")
print(f"  Improvement: {overall_f1_improvement:+.1f}%")
print()
print(f"ROUGE-L:")
print(f"  Baseline:    {overall_baseline_rouge:.4f}")
print(f"  Fine-tuned:  {overall_finetuned_rouge:.4f}")
print(f"  Improvement: {overall_rouge_improvement:+.1f}%")
print()

# Save detailed comparison
df_save = df.drop('Weight', axis=1)
df_save.to_csv('results/comparison_summary.csv', index=False)
print(f"Detailed comparison saved: results/comparison_summary.csv")
print()

# Show table
print("DETAILED RESULTS TABLE")
print(df_save.to_string(index=False))
print()

# Count improvements
tasks_improved_f1 = len(df[df['F1_Improvement_%'] > 0])
tasks_improved_rouge = len(df[df['ROUGE_Improvement_%'] > 0])

print("VERDICT")
print(f"Tasks with F1 improvement:     {tasks_improved_f1}/{len(df)}")
print(f"Tasks with ROUGE-L improvement: {tasks_improved_rouge}/{len(df)}")
print()

if overall_f1_improvement > 0 and overall_rouge_improvement > 0:
    print("FINE-TUNING WAS SUCCESSFUL!")
    print(f"   Overall improvement: F1 {overall_f1_improvement:+.1f}%, ROUGE-L {overall_rouge_improvement:+.1f}%")
elif overall_f1_improvement > 0:
    print("FINE-TUNING HELPED (F1 improved)")
    print(f"   F1 improvement: {overall_f1_improvement:+.1f}%")
else:
    print("Mixed results - check individual tasks")
print()

print("DONE!")
