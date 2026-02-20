# Phi-3-mini Training

Phi-3-mini (3.8B parameters) with F1 score of **0.362**

## 📊 Performance

- **Overall F1**: 0.362
- **Training Time**: 17.4 hours
- **GPU Memory**: 8GB (consumer-grade compatible)
- **Improvement**: +1.8% (minimal learning)

## 📁 Contents

### scripts/
Training and evaluation notebooks for Phi-3-mini

### visualizations/
- Baseline vs fine-tuned comparison
- Task performance analysis
- Learning curves

### results/
- Evaluation metrics
- Task breakdown
- Limited improvement analysis

### data/
**Dataset**: [indian-legal-phi3-dataset](https://huggingface.co/datasets/nadeem172/indian-legal-phi3-dataset)
- Train: 94,495 examples
- Test: 1,900 examples

### model/
**Fine-tuned Model**: [phi3-indian-legal](https://huggingface.co/nadeem172/phi3-indian-legal)

## 🔧 Training Configuration

```python
# LoRA Config (conservative)
rank = 8
alpha = 16
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
learning_rate = 5e-5
batch_size = 16  # Smaller for efficiency
epochs = 1
precision = BF16
```

## 📈 Task-wise Results

| Task | F1 Score |
|------|----------|
| Case Analysis | 0.584 |
| Entity Extraction | 0.369 |
| Keyword Extraction | 0.045 |
| Legal Reasoning | 0.648 |
| Question Answering | 0.287 |
| Similarity | 0.103 |
| Summarization | 0.500 |

## 🔍 Why Limited Improvement?

Phi-3's minimal improvement (+1.8%) despite extensive fine-tuning suggests:
1. **Capacity constraints**: 3.8B parameters may be insufficient for complex legal patterns
2. **Aggressive pre-training curation**: Optimized for specific domains that don't transfer to legal tasks
3. **Catastrophic forgetting**: Struggles to retain general capabilities while learning legal specifics

Good baseline performance (F1: 0.884 on legal reasoning) but limited learning from 94,495 examples indicates architectural limitations for domain adaptation.
