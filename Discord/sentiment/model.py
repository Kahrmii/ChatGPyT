import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        mask = (x != 0).unsqueeze(-1).float()          # (batch, seq_len, 1)
        embedded = self.dropout(self.embedding(x))     # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(embedded)               # (batch, seq_len, hidden_dim)
        # Mean pool over non-padding positions only
        lengths = mask.sum(dim=1).clamp(min=1)         # (batch, 1)
        pooled = (outputs * mask).sum(dim=1) / lengths # (batch, hidden_dim)
        return self.fc(self.dropout(pooled))
