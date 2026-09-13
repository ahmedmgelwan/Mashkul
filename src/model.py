import torch
import torch.nn as nn


class DiacritizationModelBiGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        labels_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Linear(hidden_dim * 2, labels_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)

        lengths = (x != self.pad_idx).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.output(output)
        return logits.permute(0, 2, 1)  # [batch, num_classes, seq_len] عشان يطابق CrossEntropyLoss

    @classmethod
    def from_config(cls, config, vocab_size: int, labels_size: int):
        return cls(
            vocab_size=vocab_size,
            labels_size=labels_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)