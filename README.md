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

## Vocabulary & Tokenization

This implementation uses **character-level tokenization**:

- **Vocab**: Every unique character in the corpus becomes a token
- **Tiny Shakespeare vocab size**: 65 characters (a-z, A-Z, punctuation, space, newline)
- **No subword tokenization**: No BPE, WordPiece, or SentencePiece

The `CharVocab` class handles tokenization:

```python
class CharVocab:
    def __init__(self, text: str):
        chars = sorted(set(text))  # Find all unique characters
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join(self.idx_to_char[i] for i in indices)
```

**Example**:
```
"Hello" → [20, 33, 44, 44, 47]  # Each char maps to an integer
[20, 33, 44, 44, 47] → "Hello"  # And back
```

**Why character-level?**
- Simpler implementation
- No OOV (out-of-vocabulary) tokens
- Works well for small corpora
- Demonstrates the chainable composition concept clearly

## Architecture

```
TokenEmbedding: token_id → 128-dim latent vector
CompositionMLP: (128, 128) → 512 → 512 → 128 (with LayerNorm, GELU, Dropout, Residual)
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
- **N-grams**: Mixed 2-5 gram samples (~4M training samples)
- **Optimizer**: AdamW with cosine annealing LR
- **Batch size**: 256
- **Epochs**: 10 (default)
- **Improvements**: Residual connections, LayerNorm, label smoothing
- **Expected loss**: ~1.7-1.8 (perplexity ~5-6)

## Generation

From seed text like "Follow those":
1. Embed and compose all seed characters
2. Decode latent → sample next token (temperature + nucleus sampling)
3. Compose new token with current latent
4. Repeat

## Files

- `model.py` - Core model classes (`CharVocab`, `ChainableMarkovModel`, `train`, `generate`)
- `train.py` - Standalone training script
- `chainable_markov.ipynb` - Colab notebook (recommended)
- `data/shakespeare.txt` - Training corpus (Tiny Shakespeare, ~1MB)
