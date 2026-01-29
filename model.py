"""
Scan Markov Chain Model

Linear recurrence model with input-dependent gating for Markov chain prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


def log_embedding_visualizations(model, vocab, wandb, epoch, device):
    """Log embedding similarity heatmap and t-SNE scatter to wandb."""
    if wandb is None:
        return

    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(vocab.vocab_size, device=device)
        embeddings = model.embed(all_ids).cpu().numpy()

        labels = []
        for i in range(vocab.vocab_size):
            ch = vocab.idx_to_char[i]
            if ch == '\n':
                labels.append('\\n')
            elif ch == ' ':
                labels.append('␣')
            elif ch == '\t':
                labels.append('\\t')
            else:
                labels.append(ch)

        _log_similarity_heatmap(embeddings, labels, wandb, epoch)
        _log_tsne_scatter(embeddings, labels, wandb, epoch)
        _log_3d_embeddings(embeddings, labels, wandb, epoch)
        _log_composition_heatmaps(model, vocab, labels, wandb, epoch, device)

    model.train()


def _log_similarity_heatmap(embeddings, labels, wandb, epoch):
    """Log cosine similarity heatmap of character embeddings."""
    try:
        import matplotlib.pyplot as plt

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity = normalized @ normalized.T

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_title(f'Character Embedding Cosine Similarity (Epoch {epoch})')
        fig.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()

        wandb.log({f'embedding_similarity': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    except ImportError:
        pass


def _log_tsne_scatter(embeddings, labels, wandb, epoch):
    """Log t-SNE visualization of character embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
        coords = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = []
        for label in labels:
            if label in 'aeiouAEIOU':
                colors.append('red')
            elif label.isalpha():
                colors.append('blue')
            elif label.isdigit():
                colors.append('green')
            elif label in '.,!?;:\'"':
                colors.append('orange')
            else:
                colors.append('gray')

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7, s=100)

        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='center')

        ax.set_title(f't-SNE of Character Embeddings (Epoch {epoch})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Vowels'),
            Patch(facecolor='blue', label='Consonants'),
            Patch(facecolor='orange', label='Punctuation'),
            Patch(facecolor='gray', label='Whitespace/Other'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        wandb.log({f'embedding_tsne': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    except ImportError:
        pass


def _log_3d_embeddings(embeddings, labels, wandb, epoch):
    """Log 3D point cloud visualization of embeddings using PCA."""
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(embeddings)

        colors = []
        for label in labels:
            if label in 'aeiouAEIOU':
                colors.append([255, 80, 80])
            elif label.isalpha():
                colors.append([80, 80, 255])
            elif label.isdigit():
                colors.append([80, 255, 80])
            elif label in '.,!?;:\'"':
                colors.append([255, 165, 0])
            else:
                colors.append([128, 128, 128])

        colors = np.array(colors, dtype=np.uint8)
        point_cloud = np.hstack([coords_3d, colors]).astype(np.float32)

        wandb.log({
            'embedding_3d': wandb.Object3D(point_cloud),
            'pca_variance_explained': sum(pca.explained_variance_ratio_),
        }, step=epoch)

    except ImportError:
        pass


def _log_composition_heatmaps(model, vocab, labels, wandb, epoch, device):
    """Log heatmaps showing what the compose() operation learned."""
    try:
        import matplotlib.pyplot as plt
        import torch

        vocab_size = vocab.vocab_size
        all_ids = torch.arange(vocab_size, device=device)

        token_emb = model.embed(all_ids)
        pos_0 = model.pos_embedding(torch.tensor([0], device=device))
        pos_1 = model.pos_embedding(torch.tensor([1], device=device))
        emb_pos0 = token_emb + pos_0
        emb_pos1 = token_emb + pos_1

        predictions = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        confidences = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for i in range(vocab_size):
            emb_i = emb_pos0[i].unsqueeze(0).expand(vocab_size, -1)
            composed = model.compose_latents(emb_i, emb_pos1)
            logits = model.decode(composed)
            probs = torch.softmax(logits, dim=-1)

            predictions[i] = logits.argmax(dim=-1).cpu().numpy()
            confidences[i] = probs.max(dim=-1).values.cpu().numpy()

        # Bigram Prediction Heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(predictions, cmap='tab20', aspect='auto')

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        plt.setp(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel('Second character (j)')
        ax.set_ylabel('First character (i)')
        ax.set_title(f'Bigram Predictions: compose(i,j) → predicted char (Epoch {epoch})')

        plt.tight_layout()
        wandb.log({'composition_predictions': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

        # Confidence Heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(confidences, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label='Max Probability')

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        plt.setp(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel('Second character (j)')
        ax.set_ylabel('First character (i)')
        ax.set_title(f'Bigram Prediction Confidence (Epoch {epoch})')

        plt.tight_layout()
        wandb.log({'composition_confidence': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

        # Interesting bigram predictions table
        interesting_bigrams = [
            ('t', 'h'), ('h', 'e'), ('a', 'n'), ('i', 'n'), ('e', 'r'),
            ('o', 'u'), ('t', 'o'), ('i', 't'), (' ', 't'), ('\\n', ' '),
        ]
        table_data = []
        for c1, c2 in interesting_bigrams:
            if c1 in vocab.char_to_idx or c1 == '␣':
                c1_lookup = ' ' if c1 == '␣' else ('\n' if c1 == '\\n' else c1)
                c2_lookup = ' ' if c2 == '␣' else ('\n' if c2 == '\\n' else c2)
                if c1_lookup in vocab.char_to_idx and c2_lookup in vocab.char_to_idx:
                    i, j = vocab.char_to_idx[c1_lookup], vocab.char_to_idx[c2_lookup]
                    pred_idx = predictions[i, j]
                    pred_char = labels[pred_idx]
                    conf = confidences[i, j]
                    table_data.append([c1, c2, pred_char, f"{conf:.3f}"])

        if table_data:
            table = wandb.Table(columns=["Char1", "Char2", "Predicted", "Confidence"], data=table_data)
            wandb.log({'bigram_predictions_table': table}, step=epoch)

    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: composition visualization failed: {e}")


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


class ScanMarkovModel(nn.Module):
    """
    Linear recurrence model with input-dependent gating (Mamba-style).

    Recurrence: h_t = A_t * h_{t-1} + B_t * e_t
    where A_t, B_t = sigmoid(gate_proj(e_t)) are input-dependent gates.

    The associative scan operator: (a1, b1) ⊕ (a2, b2) = (a2·a1, a2·b1 + b2)
    Sequential implementation used for short sequences (n=2-5).
    """

    def __init__(
        self,
        vocab_size: int,
        d_latent: int = 128,
        d_hidden: int = 512,
        dropout: float = 0.1,
        max_positions: int = 16,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_latent)
        self.pos_embedding = nn.Embedding(max_positions, d_latent)

        self.gate_proj = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent * 2),
        )
        self.gate_norm = nn.LayerNorm(d_latent)

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
        """Compose two latents using input-dependent gating."""
        gates = self.gate_proj(latent2)
        A, B = gates.chunk(2, dim=-1)
        return self.gate_norm(torch.sigmoid(A) * latent1 + torch.sigmoid(B) * latent2)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_hidden(latent)
        return self.decoder_out(hidden)

    def forward_chain(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """
        Forward pass using linear recurrence with input-dependent gating.

        h_0 = B_0 * e_0
        h_t = A_t * h_{t-1} + B_t * e_t
        """
        batch_size = len(token_sequences)
        lengths = [len(seq) for seq in token_sequences]
        max_len = max(lengths)

        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        positions = torch.arange(max_len, device=device)
        token_emb = self.embed(padded)
        pos_emb = self.pos_embedding(positions)
        embeddings = token_emb + pos_emb.unsqueeze(0)

        gates = self.gate_proj(embeddings)
        A_all = torch.sigmoid(gates[..., :self.d_latent])
        B_all = torch.sigmoid(gates[..., self.d_latent:])

        latent = B_all[:, 0, :] * embeddings[:, 0, :]

        lengths_t = torch.tensor(lengths, device=device)
        for pos in range(1, max_len):
            mask = (pos < lengths_t).unsqueeze(-1)
            new_latent = A_all[:, pos, :] * latent + B_all[:, pos, :] * embeddings[:, pos, :]
            latent = torch.where(mask, new_latent, latent)

        return self.gate_norm(latent)

    def forward(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        latent = self.forward_chain(token_sequences, device)
        return self.decode(latent)


def train(
    model: nn.Module,
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
) -> tuple[nn.Module, dict]:
    """Training loop with cosine annealing LR schedule. Returns model and history."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    wandb = None
    if use_wandb:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            print("wandb not installed, skipping logging")

    global_step = 0
    log_every = 500

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        running_loss = 0.0

        for batch_idx, (contexts, targets) in enumerate(train_loader):
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(contexts, device)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()
            train_steps += 1
            global_step += 1

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

        if vocab is not None:
            sample = _generate_sample(model, vocab, sample_seed, max_length=100, device=device)
            print(f"  Sample: {sample[:80]}...")

            log_embedding_visualizations(model, vocab, wandb, epoch + 1, device)
            if wandb:
                wandb.log({'sample_text': wandb.Html(f'<pre>{sample}</pre>'), 'epoch': epoch + 1})

    model.load_state_dict(torch.load('markov_model_best.pt', weights_only=True))
    return model, history


def _generate_sample(model, vocab, seed, max_length=100, temperature=0.8, device=torch.device('cpu')):
    """Quick generation for training samples."""
    model.eval()
    tokens = vocab.encode(seed)

    token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(tokens), device=device)
    token_emb = model.embed(token_ids)
    pos_emb = model.pos_embedding(positions)
    embeddings = token_emb + pos_emb
    latent = embeddings[0]
    for i in range(1, len(tokens)):
        latent = model.compose_latents(latent, embeddings[i])

    generated = list(seed)
    current_pos = len(tokens)

    for _ in range(max_length):
        logits = model.decode(latent) / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(vocab.idx_to_char[next_token])

        next_tok_t = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_idx = min(current_pos, model.pos_embedding.num_embeddings - 1)
        next_token_emb = model.embed(next_tok_t)[0]
        next_pos_emb = model.pos_embedding(torch.tensor([pos_idx], device=device))[0]
        next_embedding = next_token_emb + next_pos_emb
        latent = model.compose_latents(latent, next_embedding)

        current_pos += 1

    model.train()
    return ''.join(generated)


@torch.no_grad()
def generate(
    model: nn.Module,
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

    token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(tokens), device=device)
    token_emb = model.embed(token_ids)
    pos_emb = model.pos_embedding(positions)
    embeddings = token_emb + pos_emb
    latent = embeddings[0]
    for i in range(1, len(tokens)):
        latent = model.compose_latents(latent, embeddings[i])

    generated = list(seed)
    current_pos = len(tokens)

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

        next_tok_t = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_idx = min(current_pos, model.pos_embedding.num_embeddings - 1)
        next_token_emb = model.embed(next_tok_t)[0]
        next_pos_emb = model.pos_embedding(torch.tensor([pos_idx], device=device))[0]
        next_embedding = next_token_emb + next_pos_emb
        latent = model.compose_latents(latent, next_embedding)

        current_pos += 1

    return ''.join(generated)
