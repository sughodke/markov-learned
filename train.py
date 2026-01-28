#!/usr/bin/env python3
"""
Chainable Markov Chain Model

Trains a model that learns a chainable composition operation in latent space
for Markov chain prediction. Given tokens, the model composes them into a
latent representation that can predict the next token and be further composed.
"""

import os
import urllib.request
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Character Vocabulary
# -----------------------------------------------------------------------------

class CharVocab:
    """Character-level tokenization."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join(self.idx_to_char[i] for i in indices)


# -----------------------------------------------------------------------------
# N-gram Dataset
# -----------------------------------------------------------------------------

class NgramDataset(Dataset):
    """Creates 2-5 gram samples from text."""

    def __init__(self, text: str, vocab: CharVocab, min_n: int = 2, max_n: int = 5):
        self.vocab = vocab
        self.min_n = min_n
        self.max_n = max_n

        # Encode full text
        self.encoded = vocab.encode(text)

        # Create all n-gram samples
        self.samples = []
        for n in range(min_n, max_n + 1):
            for i in range(len(self.encoded) - n):
                # Input: first n tokens, Target: (n+1)th token
                context = self.encoded[i:i+n]
                target = self.encoded[i+n]
                self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


def collate_ngrams(batch: list[tuple[list[int], int]]) -> tuple[list[list[int]], torch.Tensor]:
    """Collate variable-length n-grams."""
    contexts = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return contexts, targets


# -----------------------------------------------------------------------------
# Chainable Markov Model
# -----------------------------------------------------------------------------

class ChainableMarkovModel(nn.Module):
    """
    Model that learns a chainable composition operation in latent space.

    Key property: composition outputs live in the SAME space as embeddings,
    enabling arbitrary-length chaining.
    """

    def __init__(
        self,
        vocab_size: int,
        d_latent: int = 128,
        d_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.vocab_size = vocab_size

        # Token embedding: token_id → latent_vector
        self.embedding = nn.Embedding(vocab_size, d_latent)

        # Composition MLP: (latent, latent) → latent (CHAINABLE)
        self.compose_mlp = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )

        # Decoder head: latent → logits
        # Structure: latent → hidden → latent → vocab (for weight tying)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent),
        )
        # Output projection with weight tying to embedding
        self.decoder_out = nn.Linear(d_latent, vocab_size, bias=False)
        self.decoder_out.weight = self.embedding.weight  # Weight tying

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens into latent space."""
        return self.embedding(token_ids)

    def compose_latents(self, latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        """
        Compose two latent vectors into one.

        CHAINABLE: Output lives in same 128-dim space as inputs.
        """
        combined = torch.cat([latent1, latent2], dim=-1)
        return self.compose_mlp(combined)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to vocabulary logits."""
        hidden = self.decoder_hidden(latent)
        return self.decoder_out(hidden)

    def forward_chain(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """
        Process variable-length token sequences through left-to-right composition.

        For tokens [t1, t2, t3, t4]:
            result = compose(compose(compose(embed(t1), embed(t2)), embed(t3)), embed(t4))
        """
        batch_latents = []

        for seq in token_sequences:
            # Convert to tensor and embed all tokens
            tokens = torch.tensor(seq, dtype=torch.long, device=device)
            embeddings = self.embed(tokens)  # (seq_len, d_latent)

            # Left-to-right composition
            latent = embeddings[0]
            for i in range(1, len(seq)):
                latent = self.compose_latents(latent, embeddings[i])

            batch_latents.append(latent)

        return torch.stack(batch_latents)  # (batch_size, d_latent)

    def forward(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """Full forward pass: compose tokens and decode to logits."""
        latent = self.forward_chain(token_sequences, device)
        return self.decode(latent)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    model: ChainableMarkovModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
) -> ChainableMarkovModel:
    """Training loop with cosine annealing LR schedule."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        for contexts, targets in train_loader:
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts, device)
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for contexts, targets in val_loader:
                targets = targets.to(device)
                logits = model(contexts, device)
                loss = F.cross_entropy(logits, targets)
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'markov_model_best.pt')

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Load best model
    model.load_state_dict(torch.load('markov_model_best.pt', weights_only=True))
    return model


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: ChainableMarkovModel,
    vocab: CharVocab,
    seed: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device('cpu'),
) -> str:
    """
    Generate text from a seed string using temperature and nucleus sampling.

    1. Embed and compose all seed characters
    2. Decode latent → sample next token
    3. Compose new token with current latent
    4. Repeat for desired length
    """
    model.eval()

    # Encode seed and get initial latent
    tokens = vocab.encode(seed)

    # Initial composition of seed
    embeddings = model.embed(torch.tensor(tokens, dtype=torch.long, device=device))
    latent = embeddings[0]
    for i in range(1, len(tokens)):
        latent = model.compose_latents(latent, embeddings[i])

    generated = list(seed)

    for _ in range(max_length):
        # Decode to logits
        logits = model.decode(latent)

        # Temperature scaling
        logits = logits / temperature

        # Nucleus (top-p) sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens above threshold
        sorted_indices_to_remove = cumsum > top_p
        sorted_indices_to_remove[0] = False  # Keep at least one token
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample
        idx = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[idx].item()

        generated.append(vocab.idx_to_char[next_token])

        # Compose new token with current latent for next iteration
        next_embedding = model.embed(torch.tensor([next_token], dtype=torch.long, device=device))[0]
        latent = model.compose_latents(latent, next_embedding)

    return ''.join(generated)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def download_shakespeare(data_dir: str = 'data') -> str:
    """Download Tiny Shakespeare corpus."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(filepath):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Tiny Shakespeare from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Using existing {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    # Configuration
    d_latent = 128
    d_hidden = 512
    dropout = 0.1
    batch_size = 128
    epochs = 50
    lr = 3e-4
    weight_decay = 0.01
    gradient_clip = 1.0

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    text = download_shakespeare()
    print(f"Corpus size: {len(text):,} characters")

    # Create vocabulary
    vocab = CharVocab(text)
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Train/validation split
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create datasets
    train_dataset = NgramDataset(train_text, vocab)
    val_dataset = NgramDataset(val_text, vocab)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ngrams,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ngrams,
        num_workers=0,
    )

    # Create model
    model = ChainableMarkovModel(
        vocab_size=vocab.vocab_size,
        d_latent=d_latent,
        d_hidden=d_hidden,
        dropout=dropout,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    model = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )

    # Test chainability with different sequence lengths
    print("\n" + "="*60)
    print("Chainability Test")
    print("="*60)

    model.eval()
    with torch.no_grad():
        test_seqs = [
            vocab.encode("ab"),      # 2-gram
            vocab.encode("abc"),     # 3-gram
            vocab.encode("abcd"),    # 4-gram
            vocab.encode("abcde"),   # 5-gram
        ]
        for seq in test_seqs:
            latent = model.forward_chain([seq], device)
            print(f"  {len(seq)}-gram: latent shape = {latent.shape}")

    # Generate text
    print("\n" + "="*60)
    print("Text Generation")
    print("="*60)

    seed = "Follow those"
    print(f"\nSeed: '{seed}'")
    print("-" * 40)

    generated = generate(
        model,
        vocab,
        seed,
        max_length=200,
        temperature=0.8,
        top_p=0.9,
        device=device,
    )
    print(generated)
    print("-" * 40)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_char_to_idx': vocab.char_to_idx,
        'vocab_idx_to_char': vocab.idx_to_char,
        'd_latent': d_latent,
        'd_hidden': d_hidden,
    }, 'markov_model_final.pt')
    print("\nModel saved to markov_model_final.pt")


if __name__ == '__main__':
    main()
