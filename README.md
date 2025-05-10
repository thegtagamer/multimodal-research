
# 🌍 AI-Powered Disaster Detection from Social Media

This project implements supervised, vanilla multimodal, and generative models to classify social media posts (text + images) into disaster categories:
- flood
- non_damage
- damaged_infrastructure
- human_damage
- fires
- damaged_nature

It supports BERT, ModernBERT, ConvNeXtV2, EfficientNet, FLAN-T5, GPT-4o, and LLaVA models.

---

## 📂 Project Structure

```
train/             → supervised + vanilla training scripts
eval/              → supervised + vanilla evaluation scripts
generative/        → GPT-4o, LLaVA multimodal scripts
configs/           → hyperparameter configs (optional)
data/              → dataset folders (see below)
results/           → output logs, metrics, predictions
models/            → store model checkpoints here
playground/        → test scripts, sandbox
run_experiments.py → wrapper script to call models
requirements.txt   → Python dependencies
```

---

## 📁 Dataset Structure (expected layout)

Download the dataset from this link: 

https://drive.google.com/file/d/1PyhAfxy3Af7ldXfPbxcHYeBN1rm6FBQC/view?usp=drive_link

place it in data directory and untar using the command

tar -zxvf dataset.tar.gz

```
data/
├── train/
│   ├── flood/
│   │   ├── texts/
│   │   └── images/
│   ├── non_damage/
│   ├── damaged_infrastructure/
│   ├── human_damage/
│   ├── fires/
│   └── damaged_nature/
└── test/
    └── same as above...
```

---

## 💻 Installation

```bash
pip install -r requirements.txt
```


## 🚀 Running Models with Wrapper

### ✅ Supervised models

```bash
python run_experiments.py --mode train --model bert
python run_experiments.py --mode eval --model bert
```

Supported supervised models:
- bert
- convnextv2
- efficientnet
- modernbert
- t5
- vanillamultimodal_mbert_efnet
- vanillamultimodal_bert_efnet

---

### ✅ Generative models

```bash
python run_experiments.py --mode generative --model gpt4-text
python run_experiments.py --mode generative --model gpt4-image
python run_experiments.py --mode generative --model gpt4-multimodal
python run_experiments.py --mode generative --model llava-image
python run_experiments.py --mode generative --model llava-multimodal
```

---

## ⚙️ Models and Checkpoints

Save all checkpoints to:
```
models/
```

The scripts will automatically load from and save to this directory.

---

## 📊 Results

All generative output logs, predictions, and metrics are saved to and other results and printed when eval script is executed:
```
results/
```

Examples:
- bert_results.txt
- gpt_text_evaluation_results.txt
- llava_evaluation_results.txt

---

## ⚡ Reproducibility

- Random seeds set for consistent runs
- Few-shot examples selected per class for generative models
- Same prompt templates used across generative runs

---

## ✍ Example: Run All Vanilla Models

```bash
python run_experiments.py --mode train --model vanillamultimodal_mbert_efnet
python run_experiments.py --mode eval --model vanillamultimodal_mbert_efnet

python run_experiments.py --mode train --model vanillamultimodal_bert_efnet
python run_experiments.py --mode eval --model vanillamultimodal_bert_efnet
```


