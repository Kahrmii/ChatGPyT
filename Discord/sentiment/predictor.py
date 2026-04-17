import json
import re
from pathlib import Path
from typing import List

import torch

from .model import SentimentLSTM

_SENTIMENT_DIR = Path(__file__).parent
MODEL_PATH = _SENTIMENT_DIR / "sentiment_model.pt"
VOCAB_PATH = _SENTIMENT_DIR / "vocab.json"

# Label mapping: 0=negativ, 1=neutral, 2=positiv
LABELS = {0: "negativ", 1: "neutral", 2: "positiv"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


class SentimentPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
            raise FileNotFoundError(
                "Sentiment model not found. Run `python -m sentiment.train --data your_data.csv` first "
                "to train the model. See sentiment/train.py for details."
            )

        with open(VOCAB_PATH, encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.max_len: int = checkpoint.get("max_len", 64)

        model = SentimentLSTM(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint.get("embed_dim", 64),
            hidden_dim=checkpoint.get("hidden_dim", 128),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model

    def _encode(self, text: str) -> torch.Tensor:
        tokens = _tokenize(text)[: self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]  # 1 = <unk>
        # pad to max_len
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def predict(self, text: str) -> int:
        with torch.no_grad():
            x = self._encode(text).unsqueeze(0)
            logits = self.model(x)
            return int(logits.argmax(dim=-1).item())

    def predict_batch(self, texts: List[str]) -> List[int]:
        if not texts:
            return []
        with torch.no_grad():
            batch = torch.stack([self._encode(t) for t in texts])
            logits = self.model(batch)
            return logits.argmax(dim=-1).tolist()
