# NLP Course Labs — CSED

Three labs covering core Natural Language Processing topics, implemented from scratch in PyTorch and evaluated on standard benchmarks.

---

## Labs Overview

| Lab | Topic | Models | Dataset |
|-----|-------|--------|---------|
| [Lab 2](lab2.ipynb) | Word Embeddings + NER | Skip-Gram Word2Vec, FFN, HMM | CoNLL-2003 |
| [Lab 3](lab3.ipynb) | Neural Machine Translation | Transformer, BiLSTM+Attention | FR→EN parallel corpus |
| [Lab 4](a4_neural_machine_translation.ipynb) | NMT (Assignment) | Encoder-Decoder Transformer | FR→EN (Princeton COS484) |

---

## Lab 2 — Word Embeddings & Named Entity Recognition

**[`lab2.ipynb`](lab2.ipynb)**

### Part 1: Word2Vec Skip-Gram with Negative Sampling
Implements the original Word2Vec Skip-Gram objective (Mikolov et al., 2013) entirely in PyTorch:

- Custom vocabulary with minimum-frequency filtering
- Negative sampling distribution proportional to word frequency^(3/4)
- Separate center and context embedding matrices, averaged at inference
- Word similarity and analogy evaluation (cosine similarity)
- t-SNE visualization of the learned embedding space

### Part 2: Named Entity Recognition (NER)
Two models compared on the CoNLL-2003 benchmark (PER, ORG, LOC, MISC entities):

**Feed-Forward Neural Network**
- Token-level classifier using pre-trained Word2Vec embeddings as input
- Multi-layer MLP with ReLU activations and dropout
- Trained with cross-entropy loss (padding tokens ignored)

**Hidden Markov Model + Viterbi**
- First-order HMM with Laplace smoothing
- Unknown-word handling with boosted entity probabilities
- Exact Viterbi decoding

Both models evaluated with precision, recall, and weighted F1 on the test set.

### Running Lab 2
```bash
pip install torch datasets scikit-learn matplotlib
jupyter notebook lab2.ipynb
```

---

## Lab 3 — Neural Machine Translation

**[`lab3.ipynb`](lab3.ipynb)**

Implements two Seq2Seq architectures for French → English translation:

### Transformer (from scratch)
- Learned positional embeddings
- Multi-head self-attention and cross-attention built without `nn.MultiheadAttention`
- Causal masking via lower-triangular mask
- Weight-tied output projection
- Beam search decoding (beam size 4)

### BiLSTM RNN with Additive Attention
- Bidirectional LSTM encoder
- Bahdanau (additive) attention: `v^T tanh(W_enc * h_enc + W_dec * h_dec)`
- Decoder input = [embedding; context vector] (deep output)
- Beam search decoding

Both models evaluated with corpus-level BLEU. Attention weights visualized as heatmaps.

### Data Setup
```
data/
  train.fr        train.en
  validation.fr   validation.en
  test.fr         test.en
tokenizers/
  fr/             (HuggingFace BPE tokenizer)
  en/
```

```bash
pip install torch transformers datasets matplotlib
jupyter notebook lab3.ipynb
```

---

## Assignment 4 — NMT Transformer (Princeton COS484)

**[`a4_neural_machine_translation.ipynb`](a4_neural_machine_translation.ipynb)**

A complete implementation of the encoder-decoder Transformer for the COS484 NMT assignment.

### Implemented Components

| Class / Function | Description |
|-----------------|-------------|
| `MultiHeadAttention.__init__` | Q/K/V/O projection matrices |
| `MultiHeadAttention.causal_attention_mask` | Upper-triangular causal mask |
| `MultiHeadAttention.forward` | Scaled dot-product attention with causal + padding masking |
| `plot_attention_matrix` | Heatmap of attention weights |
| `TransformerEmbeddings.__init__` | Token + positional embedding tables |
| `TransformerEmbeddings.compute_logits` | Weight-tied vocabulary projection |
| `TransformerEmbeddings.forward` | Sum of token and position embeddings |
| `EncoderDecoderModel.__init__` | Full encoder-decoder with `nn.ModuleList` |
| `EncoderDecoderModel.forward_encoder` | Encoder stack forward pass |
| `EncoderDecoderModel.forward_decoder` | Decoder stack + logit projection |
| `map_example` | Tokenise source/target text pairs |

### Setup
```bash
pip install transformers==4.27.0 datasets==2.10.0 nltk tqdm torch
# Download assignment resources
wget https://princeton-nlp.github.io/cos484/assignments/a4/resources.zip
unzip resources.zip
jupyter notebook a4_neural_machine_translation.ipynb
```

---

## Repository Structure

```
.
├── lab2.ipynb                           # Word2Vec + NER
├── lab3.ipynb                           # NMT (Transformer + RNN)
├── a4_neural_machine_translation.ipynb  # NMT assignment implementation
├── requirements.txt
└── README.md
```

---

## Requirements

```
torch>=1.13
datasets>=2.10
transformers>=4.27
scikit-learn>=1.0
matplotlib>=3.5
numpy>=1.21
tqdm
nltk
```

Install everything:
```bash
pip install -r requirements.txt
```

---

## Key References

- Mikolov et al. (2013). [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)
- Bahdanau et al. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Sang & De Meulder (2003). [Introduction to the CoNLL-2003 Shared Task](https://aclanthology.org/W03-0419/)
