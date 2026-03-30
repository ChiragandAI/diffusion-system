import string
from typing import List, Tuple

import torch
import torch.nn as nn


class CharTokenizer:
    """Very small character tokenizer for caption conditioning."""

    def __init__(self, max_length: int = 64):
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
    """Token embedding + BiGRU + attention pooling with token-level outputs."""

    def __init__(self, vocab_size: int, emb_dim: int = 192, hidden_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor, return_sequence: bool = False):
        x = self.token_emb(token_ids)
        h_seq, _ = self.gru(x)

        mask = token_ids.eq(0)
        scores = self.attn_score(h_seq).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, neg_inf)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        pooled = (h_seq * weights).sum(dim=1)
        pooled = self.proj(pooled)

        if return_sequence:
            return pooled, h_seq, mask
        return pooled


class HFTextEncoder(nn.Module):
    """Optional pretrained transformer text encoder with projection to cond dim."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 64,
        hidden_dim: int = 384,
        trainable: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError("transformers is required for HFTextEncoder. Install with: pip install transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        src_dim = self.model.config.hidden_size
        self.proj = nn.Linear(src_dim, hidden_dim) if src_dim != hidden_dim else nn.Identity()

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def encode_texts(self, texts: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = self.model(**enc).last_hidden_state
        seq = self.proj(out)

        attn_mask = enc["attention_mask"].bool()
        pad_mask = ~attn_mask

        denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (seq * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled, seq, pad_mask
