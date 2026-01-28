"""
Chainable Markov Chain Model

Core classes for training a chainable composition model.
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
        # Get all embeddings
        all_ids = torch.arange(vocab.vocab_size, device=device)
        embeddings = model.embed(all_ids).cpu().numpy()  # (vocab_size, d_latent)

        # Get character labels (escape special chars for display)
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

        # 1. Cosine similarity heatmap
        _log_similarity_heatmap(embeddings, labels, wandb, epoch)

        # 2. t-SNE scatter plot (2D)
        _log_tsne_scatter(embeddings, labels, wandb, epoch)

        # 3. 3D point cloud visualization
        _log_3d_embeddings(embeddings, labels, wandb, epoch)

        # 4. Composition operation visualization
        _log_composition_heatmaps(model, vocab, labels, wandb, epoch, device)

    model.train()


def _log_similarity_heatmap(embeddings, labels, wandb, epoch):
    """Log cosine similarity heatmap of character embeddings."""
    try:
        import matplotlib.pyplot as plt

        # Compute cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity = normalized @ normalized.T  # (vocab_size, vocab_size)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)

        # Add labels
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
        pass  # matplotlib not available


def _log_tsne_scatter(embeddings, labels, wandb, epoch):
    """Log t-SNE visualization of character embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
        coords = tsne.fit_transform(embeddings)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Color by character type
        colors = []
        for label in labels:
            if label in 'aeiouAEIOU':
                colors.append('red')  # vowels
            elif label.isalpha():
                colors.append('blue')  # consonants
            elif label.isdigit():
                colors.append('green')  # digits
            elif label in '.,!?;:\'"':
                colors.append('orange')  # punctuation
            else:
                colors.append('gray')  # whitespace/other

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7, s=100)

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='center')

        ax.set_title(f't-SNE of Character Embeddings (Epoch {epoch})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        # Add legend
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
        pass  # sklearn or matplotlib not available


def _log_3d_embeddings(embeddings, labels, wandb, epoch):
    """Log 3D point cloud visualization of embeddings using PCA."""
    try:
        from sklearn.decomposition import PCA

        # Project to 3D using PCA (faster than t-SNE, good for 3D)
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(embeddings)

        # Create colors as RGB values (0-255)
        colors = []
        for label in labels:
            if label in 'aeiouAEIOU':
                colors.append([255, 80, 80])  # red - vowels
            elif label.isalpha():
                colors.append([80, 80, 255])  # blue - consonants
            elif label.isdigit():
                colors.append([80, 255, 80])  # green - digits
            elif label in '.,!?;:\'"':
                colors.append([255, 165, 0])  # orange - punctuation
            else:
                colors.append([128, 128, 128])  # gray - whitespace/other

        colors = np.array(colors, dtype=np.uint8)

        # Create point cloud array: (N, 6) with [x, y, z, r, g, b]
        point_cloud = np.hstack([coords_3d, colors]).astype(np.float32)

        # Log as 3D object
        wandb.log({
            'embedding_3d': wandb.Object3D(point_cloud),
            'pca_variance_explained': sum(pca.explained_variance_ratio_),
        }, step=epoch)

    except ImportError:
        pass  # sklearn not available


def _log_composition_heatmaps(model, vocab, labels, wandb, epoch, device):
    """Log heatmaps showing what the compose() operation learned."""
    try:
        import matplotlib.pyplot as plt
        import torch

        vocab_size = vocab.vocab_size
        all_ids = torch.arange(vocab_size, device=device)
        is_additive = isinstance(model, AdditiveMarkovModel)

        if is_additive:
            # AdditiveMarkovModel: use position-specific embeddings
            emb_pos0 = model.pos_embeddings[0](all_ids)  # All chars at position 0
            emb_pos1 = model.pos_embeddings[1](all_ids)  # All chars at position 1
        else:
            # ChainableMarkovModel: token embedding + position embedding
            token_emb = model.embed(all_ids)
            pos_0 = model.pos_embedding(torch.tensor([0], device=device))
            pos_1 = model.pos_embedding(torch.tensor([1], device=device))
            emb_pos0 = token_emb + pos_0
            emb_pos1 = token_emb + pos_1

        # Compute compose(i, j) for all pairs and get predictions
        predictions = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        confidences = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for i in range(vocab_size):
            emb_i = emb_pos0[i].unsqueeze(0).expand(vocab_size, -1)  # (vocab_size, d_latent)

            if is_additive:
                # Additive: just sum and normalize
                composed = model.latent_norm(emb_i + emb_pos1)
            else:
                # Chainable: use compose_latents
                composed = model.compose_latents(emb_i, emb_pos1)

            logits = model.decode(composed)
            probs = torch.softmax(logits, dim=-1)

            predictions[i] = logits.argmax(dim=-1).cpu().numpy()
            confidences[i] = probs.max(dim=-1).values.cpu().numpy()

        # 1. Bigram Prediction Heatmap - what char is predicted after bigram (i,j)?
        fig, ax = plt.subplots(figsize=(14, 12))

        # Use a categorical colormap - we'll show the predicted char as color
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

        # 2. Confidence Heatmap - how confident is the model?
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

        # 3. Log some interesting bigram predictions as a table
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


class ChainableMarkovModel(nn.Module):
    """
    Model that learns a chainable composition operation in latent space.

    Key property: composition outputs live in the SAME space as embeddings,
    enabling arbitrary-length chaining.

    Improvements:
    - Positional embeddings: tokens know their position in the sequence
    - Deeper composition: 3-layer MLP with wider hidden (768) for more capacity
    """

    def __init__(
        self,
        vocab_size: int,
        d_latent: int = 128,
        d_hidden: int = 512,
        dropout: float = 0.1,
        max_positions: int = 16,  # Support up to 16-gram (plenty of headroom)
    ):
        super().__init__()
        self.d_latent = d_latent
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_latent)

        # Positional embedding - learnable position representations
        self.pos_embedding = nn.Embedding(max_positions, d_latent)

        # Deeper composition MLP with wider hidden layer for more capacity
        # Input: concatenated latents (d_latent * 2)
        # Hidden: 768 (wider than before) with 3 layers
        # Output: d_latent (same space for chainability)
        d_compose = 768  # Wider hidden for more composition capacity
        self.compose_mlp = nn.Sequential(
            nn.Linear(d_latent * 2, d_compose),
            nn.LayerNorm(d_compose),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_compose, d_compose),
            nn.LayerNorm(d_compose),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_compose, d_hidden),  # Gradually reduce
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )
        self.compose_norm = nn.LayerNorm(d_latent)  # Normalize after residual

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
        """Compose with residual connection and layer norm for better gradient flow."""
        combined = torch.cat([latent1, latent2], dim=-1)
        return self.compose_norm(latent1 + self.compose_mlp(combined))  # Residual + LayerNorm

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_hidden(latent)
        return self.decoder_out(hidden)

    def forward_chain(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """
        Vectorized forward chain - processes entire batch in parallel.

        For sequences of varying lengths (2-5), pads to max length and uses
        masking to handle the variable composition steps.

        Includes positional embeddings so tokens know their position in the sequence.
        """
        batch_size = len(token_sequences)
        lengths = [len(seq) for seq in token_sequences]
        max_len = max(lengths)

        # Pad sequences to max length
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        positions = torch.arange(max_len, device=device)

        # Embed all tokens at once: (batch, max_len, d_latent)
        token_emb = self.embed(padded)
        pos_emb = self.pos_embedding(positions)  # (max_len, d_latent)

        # Add positional embeddings to token embeddings
        embeddings = token_emb + pos_emb.unsqueeze(0)  # Broadcasting: (batch, max_len, d_latent)

        # Start with first token embedding: (batch, d_latent)
        latent = embeddings[:, 0, :]

        # Create length tensor for masking
        lengths_t = torch.tensor(lengths, device=device)

        # Compose step by step across entire batch
        for pos in range(1, max_len):
            # Mask: which sequences have a token at this position
            mask = (pos < lengths_t).unsqueeze(-1)  # (batch, 1)

            # Get next embeddings (already has position info): (batch, d_latent)
            next_emb = embeddings[:, pos, :]

            # Compose all pairs in batch
            new_latent = self.compose_latents(latent, next_emb)

            # Update only sequences that have this position
            latent = torch.where(mask, new_latent, latent)

        return latent

    def forward(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        latent = self.forward_chain(token_sequences, device)
        return self.decode(latent)


class AdditiveMarkovModel(nn.Module):
    """
    Position-specific embeddings with interaction MLP.

    Key insight: Pure addition can't capture interactions between positions.
    We need a non-linear layer to learn "t at pos 0 AND h at pos 1 → e".

    Architecture:
        1. Sum position-specific embeddings (preserves all information)
        2. Apply interaction MLP (learns non-linear relationships)
        3. Decode to vocabulary

    For tokens [t1, t2, t3, t4]:
        summed = embed_0(t1) + embed_1(t2) + embed_2(t3) + embed_3(t4)
        latent = interaction_mlp(summed)
        output = decode(latent)

    Benefits over chained composition:
    - No repeated compression (sum preserves info)
    - Single MLP application (not n-1 compositions)
    - Position-aware by design
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
        self.max_positions = max_positions

        # Position-specific embeddings: each position has its own embedding matrix
        self.pos_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_latent) for _ in range(max_positions)
        ])

        # Shared embedding for positions beyond max (fallback for generation)
        self.fallback_embedding = nn.Embedding(vocab_size, d_latent)

        # Interaction MLP: learns non-linear relationships between summed embeddings
        # This is the key addition - without this, pure addition can't learn AND relationships
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )
        self.latent_norm = nn.LayerNorm(d_latent)

        # Decoder: latent → vocab logits (no weight tying - doesn't make sense here)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, vocab_size),
        )

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
        """Embed using position 0 embedding (for compatibility)."""
        return self.pos_embeddings[0](token_ids)

    def embed_at_position(self, token_ids: torch.Tensor, position: int) -> torch.Tensor:
        """Embed tokens at a specific position."""
        if position < self.max_positions:
            return self.pos_embeddings[position](token_ids)
        else:
            return self.fallback_embedding(token_ids)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to vocabulary logits."""
        return self.decoder(latent)

    def forward_chain(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """
        Additive forward pass with interaction MLP.

        1. Sum position-specific embeddings (preserves all information)
        2. Apply interaction MLP (learns non-linear relationships)
        """
        batch_size = len(token_sequences)
        lengths = [len(seq) for seq in token_sequences]
        max_len = max(lengths)

        # Initialize sum as zeros
        summed = torch.zeros(batch_size, self.d_latent, device=device)

        # Pad sequences
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        # Create mask for valid positions
        lengths_t = torch.tensor(lengths, device=device)

        # Sum position-specific embeddings
        for pos in range(max_len):
            # Get embedding for this position
            if pos < self.max_positions:
                pos_emb = self.pos_embeddings[pos](padded[:, pos])  # (batch, d_latent)
            else:
                pos_emb = self.fallback_embedding(padded[:, pos])

            # Mask: only add for sequences that have this position
            mask = (pos < lengths_t).unsqueeze(-1).float()  # (batch, 1)
            summed = summed + pos_emb * mask

        # Apply interaction MLP to learn non-linear relationships
        latent = self.interaction_mlp(summed)
        latent = self.latent_norm(latent)

        return latent

    def forward(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """Full forward pass: sum embeddings and decode."""
        latent = self.forward_chain(token_sequences, device)
        return self.decode(latent)

    # For compatibility with visualization code
    @property
    def pos_embedding(self):
        """Dummy property for visualization compatibility."""
        return self.pos_embeddings[0]

    def compose_latents(self, latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        """For compatibility - adds and applies interaction MLP."""
        summed = latent1 + latent2
        return self.latent_norm(self.interaction_mlp(summed))

    def get_raw_sum(self, token_sequences: list[list[int]], device: torch.device) -> torch.Tensor:
        """Get raw sum of embeddings (before interaction MLP) for incremental generation."""
        batch_size = len(token_sequences)
        lengths = [len(seq) for seq in token_sequences]
        max_len = max(lengths)

        summed = torch.zeros(batch_size, self.d_latent, device=device)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        lengths_t = torch.tensor(lengths, device=device)

        for pos in range(max_len):
            if pos < self.max_positions:
                pos_emb = self.pos_embeddings[pos](padded[:, pos])
            else:
                pos_emb = self.fallback_embedding(padded[:, pos])
            mask = (pos < lengths_t).unsqueeze(-1).float()
            summed = summed + pos_emb * mask

        return summed

    def apply_interaction(self, raw_sum: torch.Tensor) -> torch.Tensor:
        """Apply interaction MLP to raw sum to get latent."""
        return self.latent_norm(self.interaction_mlp(raw_sum))


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
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)  # Helps generalization
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

            # Log embedding visualizations every epoch
            log_embedding_visualizations(model, vocab, wandb, epoch + 1, device)
            if wandb:
                wandb.log({'sample_text': wandb.Html(f'<pre>{sample}</pre>'), 'epoch': epoch + 1})

    model.load_state_dict(torch.load('markov_model_best.pt', weights_only=True))
    return model, history


def _generate_sample(model, vocab, seed, max_length=100, temperature=0.8, device=torch.device('cpu')):
    """Quick generation for training samples. Works with both model types."""
    model.eval()
    tokens = vocab.encode(seed)
    is_additive = isinstance(model, AdditiveMarkovModel)

    if is_additive:
        # AdditiveMarkovModel: track raw sum, apply interaction MLP for decoding
        raw_sum = torch.zeros(model.d_latent, device=device)
        for i, tok in enumerate(tokens):
            tok_t = torch.tensor([tok], dtype=torch.long, device=device)
            raw_sum = raw_sum + model.embed_at_position(tok_t, i)[0]
        latent = model.apply_interaction(raw_sum)
    else:
        # ChainableMarkovModel: use positional embeddings added to token embeddings
        token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(len(tokens), device=device)
        token_emb = model.embed(token_ids)
        pos_emb = model.pos_embedding(positions)
        embeddings = token_emb + pos_emb
        latent = embeddings[0]
        for i in range(1, len(tokens)):
            latent = model.compose_latents(latent, embeddings[i])
        raw_sum = None  # Not used for chainable

    generated = list(seed)
    current_pos = len(tokens)

    for _ in range(max_length):
        logits = model.decode(latent) / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(vocab.idx_to_char[next_token])

        # Get embedding for next token at current position
        next_tok_t = torch.tensor([next_token], dtype=torch.long, device=device)
        if is_additive:
            # Add to raw sum, then apply interaction MLP
            next_embedding = model.embed_at_position(next_tok_t, current_pos)[0]
            raw_sum = raw_sum + next_embedding
            latent = model.apply_interaction(raw_sum)
        else:
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
    model,  # ChainableMarkovModel or AdditiveMarkovModel
    vocab: CharVocab,
    seed: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device('cpu'),
) -> str:
    """Generate text using temperature and nucleus sampling. Works with both model types."""
    model.eval()
    tokens = vocab.encode(seed)
    is_additive = isinstance(model, AdditiveMarkovModel)

    if is_additive:
        # AdditiveMarkovModel: track raw sum, apply interaction MLP for decoding
        raw_sum = torch.zeros(model.d_latent, device=device)
        for i, tok in enumerate(tokens):
            tok_t = torch.tensor([tok], dtype=torch.long, device=device)
            raw_sum = raw_sum + model.embed_at_position(tok_t, i)[0]
        latent = model.apply_interaction(raw_sum)
    else:
        # ChainableMarkovModel: compose with positional embeddings
        token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(len(tokens), device=device)
        token_emb = model.embed(token_ids)
        pos_emb = model.pos_embedding(positions)
        embeddings = token_emb + pos_emb
        latent = embeddings[0]
        for i in range(1, len(tokens)):
            latent = model.compose_latents(latent, embeddings[i])
        raw_sum = None  # Not used for chainable

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

        # Get embedding for next token at current position
        next_tok_t = torch.tensor([next_token], dtype=torch.long, device=device)
        if is_additive:
            # Add to raw sum, then apply interaction MLP
            next_embedding = model.embed_at_position(next_tok_t, current_pos)[0]
            raw_sum = raw_sum + next_embedding
            latent = model.apply_interaction(raw_sum)
        else:
            pos_idx = min(current_pos, model.pos_embedding.num_embeddings - 1)
            next_token_emb = model.embed(next_tok_t)[0]
            next_pos_emb = model.pos_embedding(torch.tensor([pos_idx], device=device))[0]
            next_embedding = next_token_emb + next_pos_emb
            latent = model.compose_latents(latent, next_embedding)

        current_pos += 1

    return ''.join(generated)
