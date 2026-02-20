# Mistral 7B Training

Mistral 7B Instruct with F1 score of **0.199**

## 📊 Performance

- **Overall F1**: 0.199
- **Training Time**: 5.3 hours
- **GPU Memory**: 47GB (H100)
- **Challenge**: Conversational training conflicts with extraction tasks

## 📁 Contents

### scripts/
Training and evaluation notebooks for Mistral 7B

### visualizations/
- Performance analysis
- Task-wise comparisons
- Error analysis

### results/
- Evaluation metrics
- Task breakdown
- Failure mode analysis

### data/
**Dataset**: [indian-legal-mistral-dataset](https://huggingface.co/datasets/nadeem172/indian-legal-mistral-dataset)
- Train: 94,495 examples
- Test: 1,900 examples

### model/
**Fine-tuned Model**: [mistral-7b-indian-legal](https://huggingface.co/nadeem172/mistral-7b-indian-legal)

## 🔧 Training Configuration

```python
# LoRA Config
rank = 32
alpha = 64
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
learning_rate = 5e-5
batch_size = 33
epochs = 2
precision = BF16
```

## 📈 Task-wise Results

| Task | F1 Score |
|------|----------|
| Case Analysis | 0.354 |
| Entity Extraction | 0.014 |
| Keyword Extraction | 0.022 |
| Legal Reasoning | 0.362 |
| Question Answering | 0.227 |
| Similarity | 0.097 |
| Summarization | 0.341 |

## 🔍 Why Lower Performance?

Mistral-7B-Instruct was trained for conversational AI, optimizing for natural responses. This causes:
- **Paraphrasing** instead of verbatim extraction
- **Context addition** that reduces precision
- **2.3× text expansion** hurting F1 scores

Example: Asked for section text, it responds "Section 5 establishes that..." (435 chars) instead of verbatim text (188 chars).
