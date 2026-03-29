import string
import torch
import torch.nn as nn


class CharTokenizer:
    """Very small character tokenizer for caption conditioning."""

    def __init__(self, max_length: int = 48):
        chars = string.ascii_lowercase + string.digits + " .,!?-_/"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.max_length = max_length
        self.vocab = [self.pad_token, self.unk_token] + list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower().strip()
        ids = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text][: self.max_length]
        if len(ids) == 0:
            ids = [self.stoi[self.unk_token]]
        if len(ids) < self.max_length:
            ids = ids + [self.stoi[self.pad_token]] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class TinyTextEncoder(nn.Module):
    """Token embedding + BiGRU + attention pooling for stronger prompt representations."""

    def __init__(self, vocab_size: int, emb_dim: int = 192, hidden_dim: int = 384):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(token_ids)
        h_seq, _ = self.gru(x)

        scores = self.attn_score(h_seq).squeeze(-1)
        mask = token_ids.eq(0)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        pooled = (h_seq * weights).sum(dim=1)
        return self.proj(pooled)
