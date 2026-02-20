# Fine-Tuning Large Language Models for Indian Legal NLP

[![Models](https://img.shields.io/badge/🤗%20Models-HuggingFace-orange)](https://huggingface.co/nadeem172/models)
[![Datasets](https://img.shields.io/badge/🤗%20Datasets-HuggingFace-yellow)](https://huggingface.co/nadeem172/datasets)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A Multi-Task Comparative Study of Llama 3.1-8B, Mistral 7B, and Phi-3 on Indian Legal Documents**

## 📊 Key Results

| Model | F1 Score | vs Baseline | vs Phi-3 | vs Mistral |
|-------|----------|-------------|----------|------------|
| **Llama 3.1-8B** | **0.781** | **+26.4%** | **+115.7%** | **+292%** |
| Phi-3-mini | 0.362 | +1.8% | - | +82.0% |
| Mistral 7B | 0.199 | -26.5% | -45.0% | - |

## 📁 Repository Structure

```
indian-legal-llm-finetuning/
├── llama3_training/          # Llama 3.1-8B experiments
│   ├── scripts/              # Training & evaluation notebooks
│   ├── visualizations/       # Performance plots
│   ├── results/              # Evaluation results
│   ├── data/                 # Link to HuggingFace dataset
│   └── model/                # Link to HuggingFace model
├── mistral_training/         # Mistral 7B experiments
│   ├── scripts/
│   ├── visualizations/
│   ├── results/
│   ├── data/
│   └── model/
├── phi3_training/            # Phi-3-mini experiments
│   ├── scripts/
│   ├── visualizations/
│   ├── results/
│   ├── data/
│   └── model/
├── paper/                    # Research paper
│   ├── paper_springer_cvr2026.pdf
│   └── paper_springer_cvr2026.tex
├── README.md                 # This file
├── requirements.txt
└── LICENSE
```

## 🗂️ Dataset

**94,495 examples** from **7,429 Indian legal documents** covering **7 NLP tasks**:

1. Case Analysis
2. Entity Extraction
3. Keyword Extraction
4. Legal Reasoning
5. Question Answering
6. Similarity Detection
7. Summarization

### Available Datasets:
- [Llama 3.1 Dataset](https://huggingface.co/datasets/nadeem172/indian-legal-llama3-dataset)
- [Mistral 7B Dataset](https://huggingface.co/datasets/nadeem172/indian-legal-mistral-dataset)
- [Phi-3 Dataset](https://huggingface.co/datasets/nadeem172/indian-legal-phi3-dataset)
- [Raw PDFs (7,631 documents)](https://huggingface.co/datasets/nadeem172/indian-legal-raw-pdfs)

## 🤖 Models

All fine-tuned models available on HuggingFace:

| Model | HuggingFace Link | F1 Score |
|-------|------------------|----------|
| Llama 3.1-8B | [nadeem172/llama-3.1-indian-legal](https://huggingface.co/nadeem172/llama-3.1-indian-legal) | 0.781 |
| Mistral 7B | [nadeem172/mistral-7b-indian-legal](https://huggingface.co/nadeem172/mistral-7b-indian-legal) | 0.199 |
| Phi-3-mini | [nadeem172/phi3-indian-legal](https://huggingface.co/nadeem172/phi3-indian-legal) | 0.362 |

## 🚀 Quick Start

### Load Dataset
```python
from datasets import load_dataset

# Load specific model's dataset
dataset = load_dataset("nadeem172/indian-legal-llama3-dataset")
print(dataset)
```

### Load Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nadeem172/llama-3.1-indian-legal")
tokenizer = AutoTokenizer.from_pretrained("nadeem172/llama-3.1-indian-legal")
```

## 📈 Results by Task

| Task | Llama 3.1 | Mistral 7B | Phi-3 | Best Δ |
|------|-----------|------------|-------|--------|
| Case Analysis | 0.872 | 0.354 | 0.584 | +9.0% |
| Entity Extraction | 0.278 | 0.014 | 0.369 | +46.6% |
| Keyword Extraction | 0.578 | 0.022 | 0.045 | +81.6% |
| Legal Reasoning | 0.946 | 0.362 | 0.648 | +8.7% |
| Question Answering | 0.952 | 0.227 | 0.287 | +20.4% |
| Similarity | 0.986 | 0.097 | 0.103 | +72.0% |
| Summarization | 0.957 | 0.341 | 0.500 | +17.2% |

## 📝 Citation

```bibtex
@inproceedings{akhtar2026indian,
  title={Fine-Tuning Large Language Models for Indian Legal NLP: A Multi-Task Comparative Study},
  author={Akhtar, Md Nadeem and Singh, Amritpal},
  booktitle={Proceedings of CVR 2026},
  year={2026},
  organization={Springer}
}
```

## 👥 Authors

**Md Nadeem Akhtar**  
M.Tech Data Science and Engineering  
Dr. B R Ambedkar National Institute of Technology, Jalandhar  
Email: mdna.ds.24@nitj.ac.in

**Dr Amritpal Singh**  
Deptt of Computer Science and Engineering  
Dr. BR Ambedkar National Institute of Technology, Jalandhar  
Email: apsingh@nitj.ac.in

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🔗 Links

- **Paper**: [Springer CVR 2026]
- **Models**: [HuggingFace](https://huggingface.co/nadeem172/models)
- **Datasets**: [HuggingFace](https://huggingface.co/nadeem172/datasets)

---

**Note**: This work is for research and educational purposes. Not intended for production legal advice.
