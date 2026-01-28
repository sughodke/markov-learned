# Chainable Markov Chain Model

A neural network that learns a chainable composition operation in latent space for Markov chain prediction.

## Overview

This model composes tokens into a latent representation that can:
1. Predict the next token
2. Be further composed with additional tokens (chainability)

The key insight is that the composition operation outputs vectors in the **same 128-dim space** as the input embeddings, enabling arbitrary-length chaining:

```
2-gram: decode(compose(embed(t1), embed(t2))) → t3
3-gram: decode(compose(compose(embed(t1), embed(t2)), embed(t3))) → t4
4-gram: decode(compose(compose(compose(t1, t2), t3), t4)) → t5
```

## Architecture

```
TokenEmbedding: token_id → 128-dim latent vector
CompositionMLP: (128, 128) → 512 → 512 → 128 (with LayerNorm, GELU, Dropout)
DecoderHead: 128 → 512 → vocab_size (with weight tying)
```

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sughodke/markov-learned/blob/main/chainable_markov.ipynb)

Or run locally:

```bash
pip install torch
python train.py
```

## Training

- **Data**: Tiny Shakespeare (~1MB, character-level)
- **N-grams**: Mixed 2-5 gram samples
- **Optimizer**: AdamW with cosine annealing
- **Expected loss**: < 1.0 after 50 epochs

## Generation

From seed text like "Follow those":
1. Embed and compose all seed characters
2. Decode latent → sample next token (temperature + nucleus sampling)
3. Compose new token with current latent
4. Repeat

## Files

- `train.py` - Standalone training script
- `chainable_markov.ipynb` - Colab notebook
- `data/shakespeare.txt` - Training corpus (auto-downloaded)
