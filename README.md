<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,8,16&height=180&section=header&text=NLP%20Course%20Labs&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=N-Grams%20%7C%20Word2Vec%20%7C%20Transformers%20%7C%20Q-LoRA%20%26%20DPO&descAlignY=58&descSize=16&descColor=cbd5e1"/>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## Overview

Four progressive NLP labs implemented **from scratch** in Python (NumPy + PyTorch), covering classical NLP through modern LLM fine-tuning — built as part of the NLP course at Alexandria University (Computer & System Engineering, 2026).

---

## Labs Summary

| Lab | Topic | Key Models | Dataset | Result |
|-----|-------|-----------|---------|--------|
| **Lab 1** | Classical NLP | N-gram LMs, Naïve Bayes, Logistic Regression | Shakespeare + Emotion corpus | **94% macro-F1** |
| **Lab 2** | Word Embeddings + NER | Word2Vec (Skip-Gram), FFNN, HMM + Viterbi | CoNLL-2003 | 9-class NER |
| **Lab 3** | Neural Machine Translation | Transformer, Bi-LSTM + Bahdanau Attention | FR→EN corpus | BLEU evaluation |
| **Lab 4** | LLM Fine-Tuning | Q-LoRA (Qwen 1.5B), DPO alignment | Code generation | β-sensitivity analysis |

---

## Lab 1 — Classical NLP

**N-Gram Language Models & Text Classification (Oct 2024)**

- Built **unigram → 10-gram** language models; evaluated using **perplexity** on held-out Shakespeare text
- Implemented **Naïve Bayes** classifier from scratch with NumPy for 6-class emotion recognition
- Implemented **Logistic Regression** with bi-gram features from scratch
- Produced confusion matrices and computed precision, recall, and F1 scores
- **Achieved 94% macro-F1** on the emotion recognition dataset

---

## Lab 2 — Word Embeddings + Named Entity Recognition

**[`lab2.ipynb`](lab2.ipynb) — Nov 2024**

### Part 1: Word2Vec Skip-Gram with Negative Sampling
- Implements the original Mikolov et al. (2013) objective entirely in PyTorch
- Custom vocabulary, negative sampling distribution ∝ freq^(3/4)
- Word analogy evaluation: `king − man + woman ≈ queen`
- t-SNE visualization of embedding space

### Part 2: Named Entity Recognition (9 classes, CoNLL-2003)
Two models compared:

| Model | Approach |
|-------|---------|
| **FFNN** | Token-level MLP using pre-trained Word2Vec embeddings |
| **HMM + Viterbi** | Generative model with Viterbi decoding for sequence labeling |

---

## Lab 3 — Neural Machine Translation (French → English)

**[`lab3.ipynb`](lab3.ipynb) — Dec 2024**

Two seq2seq architectures built and compared:

| Architecture | Details |
|-------------|---------|
| **Transformer** | Encoder-decoder, vectorized multi-head attention, beam search decoding, BLEU scoring |
| **Bi-LSTM + Attention** | Bidirectional encoder, Bahdanau attention, attention weight visualization |

---

## Lab 4 — LLM Fine-Tuning & Alignment

**Dec 2024**

| Part | Method | Model |
|------|--------|-------|
| **I** | Supervised Fine-Tuning (SFT) | Qwen 1.5B |
| **II** | Q-LoRA PEFT | Qwen 1.5B — code generation |
| **III** | Direct Preference Optimization (DPO) | Qwen 1.5B — behavioral alignment |

DPO β-sensitivity analysis: tested β ∈ {0.1, 0.5, 0.8, 1.0} to balance capability vs. guardrails; tracked with **wandb**.

---

## Getting Started

```bash
git clone https://github.com/Ibrahimtareq952001/Natural-Language-Processing-Labs.git
cd Natural-Language-Processing-Labs
pip install -r requirements.txt
jupyter notebook
```

---

<div align="center">

*NLP & Machine Learning — Alexandria University 2024*

[![Resume](https://img.shields.io/badge/View_Resume-PDF-008080?style=flat-square&logo=latex&logoColor=white)](https://github.com/Ibrahimtareq952001/Resume/blob/main/resume.pdf)
[![Portfolio](https://img.shields.io/badge/GitHub-Ibrahimtareq952001-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ibrahimtareq952001)

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,8,16&height=100&section=footer"/>
