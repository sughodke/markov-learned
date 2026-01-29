# Scan Markov Chain Model

A neural network that uses a linear recurrence with input-dependent gating (Mamba-style) for Markov chain next-character prediction.

## Overview

The model composes a sequence of character embeddings into a single latent vector using a selective gating recurrence:

```
e_t = embed(token_t) + pos_embed(t)
[A_t, B_t] = sigmoid(gate_proj(e_t))     # input-dependent gates
h_0 = B_0 * e_0
h_t = A_t * h_{t-1} + B_t * e_t          # linear recurrence
output = decode(layer_norm(h_final))
```

This is the associative scan operator `(a1, b1) ⊕ (a2, b2) = (a2·a1, a2·b1 + b2)`, run sequentially for short n-gram contexts (n=2-5).

## Vocabulary & Tokenization

Character-level tokenization via `CharVocab`:

- **Vocab**: Every unique character in the corpus becomes a token
- **Tiny Shakespeare vocab size**: 65 characters (a-z, A-Z, punctuation, space, newline)
- **No subword tokenization**: No BPE, WordPiece, or SentencePiece

## Architecture

```
Embedding:      token_id → 128-dim
PosEmbedding:   position → 128-dim
GateProjection: 128 → 512 (GELU) → 256 → [A_t; B_t] (sigmoid)
GateNorm:       LayerNorm(128)
DecoderHidden:  128 → 512 (GELU) → 128
DecoderOut:     128 → vocab_size (weight-tied to embedding)
```

~340K parameters.

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sughodke/markov-learned/blob/main/chainable_markov.ipynb)

Or run locally:

```bash
pip install torch
python train.py
```

## Training

- **Data**: Tiny Shakespeare (~1MB, character-level)
- **N-grams**: Mixed 2-5 gram samples (~4M training samples)
- **Optimizer**: AdamW with cosine annealing LR
- **Batch size**: 256
- **Epochs**: 10 (default)
- **Label smoothing**: 0.1

## Generation

From seed text like "Follow those":
1. Embed and compose all seed characters via the gated recurrence
2. Decode latent → sample next token (temperature + nucleus sampling)
3. Compose new token with current latent
4. Repeat

## Files

- `model.py` - Core model (`ScanMarkovModel`, `CharVocab`, `train`, `generate`)
- `train.py` - Standalone training script
- `chainable_markov.ipynb` - Colab notebook (recommended)
- `data/shakespeare.txt` - Training corpus (Tiny Shakespeare, ~1MB)
