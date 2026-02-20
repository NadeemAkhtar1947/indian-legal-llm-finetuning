
# Llama 3.1-8B Training

**Best performing model** with F1 score of **0.781** (+26.4% over baseline)

## 📊 Performance

- **Overall F1**: 0.781
- **Training Time**: 4.1 hours
- **GPU Memory**: 47GB (H100)
- **Improvement**: +115.7% over Phi-3, +292% over Mistral

## 📁 Contents

### scripts/
Training and evaluation notebooks for Llama 3.1-8B

### visualizations/
- Performance comparison plots
- Task-wise analysis charts
- Training curves

### results/
- Evaluation metrics (F1, ROUGE, BLEU)
- Task-wise performance breakdown
- Comparison tables

### data/
**Dataset**: [indian-legal-llama3-dataset](https://huggingface.co/datasets/nadeem172/indian-legal-llama3-dataset)
- Train: 94,495 examples
- Test: 1,900 examples
- Format: JSON (instruction, input, output)

### model/
**Fine-tuned Model**: [llama-3.1-indian-legal](https://huggingface.co/nadeem172/llama-3.1-indian-legal)

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

| Task | F1 Score | Improvement |
|------|----------|-------------|
| Case Analysis | 0.872 | +9.0% |
| Entity Extraction | 0.278 | +46.6% |
| Keyword Extraction | 0.578 | +81.6% |
| Legal Reasoning | 0.946 | +8.7% |
| Question Answering | 0.952 | +20.4% |
| Similarity | 0.986 | +72.0% |
| Summarization | 0.957 | +17.2% |

## 🚀 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nadeem172/llama-3.1-indian-legal")
tokenizer = AutoTokenizer.from_pretrained("nadeem172/llama-3.1-indian-legal")

prompt = "Analyze Section 5 of the IT Act 2000"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```
