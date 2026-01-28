"""
Chainable Markov Chain Model

Core classes for training a chainable composition model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


class NgramDataset(Dataset):
    """Creates 2-5 gram samples from text."""

    def __init__(self, text: str, vocab: CharVocab, min_n: int = 2, max_n: int = 5):
        self.vocab = vocab
        self.min_n = min_n
        self.max_n = max_n
        self.encoded = vocab.encode(text)

        self.samples = []
        for n in range(min_n, max_n + 1):
            for i in range(len(self.encoded) - n):
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

        self.embedding = nn.Embedding(vocab_size, d_latent)

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

        self.decoder_hidden = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent),
        )
        self.decoder_out = nn.Linear(d_latent, vocab_size, bias=False)
        self.decoder_out.weight = self.embedding.weight

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
        return self.embedding(token_ids)

    def compose_latents(self, latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([latent1, latent2], dim=-1)
        return self.compose_mlp(combined)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_hidden(latent)
        return self.decoder_out(hidden)

    def forward_chain(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """
        Vectorized forward chain - processes entire batch in parallel.

        For sequences of varying lengths (2-5), pads to max length and uses
        masking to handle the variable composition steps.
        """
        batch_size = len(token_sequences)
        lengths = [len(seq) for seq in token_sequences]
        max_len = max(lengths)

        # Pad sequences to max length
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        # Embed all tokens at once: (batch, max_len, d_latent)
        embeddings = self.embed(padded)

        # Start with first token embedding: (batch, d_latent)
        latent = embeddings[:, 0, :]

        # Create length tensor for masking
        lengths_t = torch.tensor(lengths, device=device)

        # Compose step by step across entire batch
        for pos in range(1, max_len):
            # Mask: which sequences have a token at this position
            mask = (pos < lengths_t).unsqueeze(-1)  # (batch, 1)

            # Get next embeddings: (batch, d_latent)
            next_emb = embeddings[:, pos, :]

            # Compose all pairs in batch
            new_latent = self.compose_latents(latent, next_emb)

            # Update only sequences that have this position
            latent = torch.where(mask, new_latent, latent)

        return latent

    def forward(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        latent = self.forward_chain(token_sequences, device)
        return self.decode(latent)


def train(
    model: ChainableMarkovModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    use_wandb: bool = False,
    vocab: 'CharVocab' = None,
    sample_seed: str = "The ",
) -> tuple[ChainableMarkovModel, dict]:
    """Training loop with cosine annealing LR schedule. Returns model and history.

    If vocab is provided, generates sample text at each epoch end.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Import wandb if requested
    wandb = None
    if use_wandb:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            print("wandb not installed, skipping logging")

    global_step = 0
    log_every = 500  # Log to wandb every N batches

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        running_loss = 0.0

        for batch_idx, (contexts, targets) in enumerate(train_loader):
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(contexts, device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()
            train_steps += 1
            global_step += 1

            # Log batch loss frequently
            if wandb and global_step % log_every == 0:
                wandb.log({
                    'batch_loss': running_loss / log_every,
                    'global_step': global_step,
                    'epoch': epoch + (batch_idx / len(train_loader)),
                })
                running_loss = 0.0

        scheduler.step()
        avg_train_loss = train_loss / train_steps

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
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        if wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'markov_model_best.pt')

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        # Generate sample text at end of each epoch
        if vocab is not None:
            sample = _generate_sample(model, vocab, sample_seed, max_length=100, device=device)
            print(f"  Sample: {sample[:80]}...")
            if wandb:
                wandb.log({'sample_text': wandb.Html(f'<pre>{sample}</pre>'), 'epoch': epoch + 1})

    model.load_state_dict(torch.load('markov_model_best.pt', weights_only=True))
    return model, history


def _generate_sample(model, vocab, seed, max_length=100, temperature=0.8, device=torch.device('cpu')):
    """Quick generation for training samples."""
    model.eval()
    tokens = vocab.encode(seed)
    embeddings = model.embed(torch.tensor(tokens, dtype=torch.long, device=device))
    latent = embeddings[0]
    for i in range(1, len(tokens)):
        latent = model.compose_latents(latent, embeddings[i])

    generated = list(seed)
    for _ in range(max_length):
        logits = model.decode(latent) / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(vocab.idx_to_char[next_token])
        next_embedding = model.embed(torch.tensor([next_token], dtype=torch.long, device=device))[0]
        latent = model.compose_latents(latent, next_embedding)

    model.train()
    return ''.join(generated)


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
    """Generate text using temperature and nucleus sampling."""
    model.eval()
    tokens = vocab.encode(seed)

    embeddings = model.embed(torch.tensor(tokens, dtype=torch.long, device=device))
    latent = embeddings[0]
    for i in range(1, len(tokens)):
        latent = model.compose_latents(latent, embeddings[i])

    generated = list(seed)

    for _ in range(max_length):
        logits = model.decode(latent) / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumsum > top_p
        sorted_indices_to_remove[0] = False
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[idx].item()

        generated.append(vocab.idx_to_char[next_token])

        next_embedding = model.embed(torch.tensor([next_token], dtype=torch.long, device=device))[0]
        latent = model.compose_latents(latent, next_embedding)

    return ''.join(generated)
