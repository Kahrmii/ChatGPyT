import json, random, pickle
import numpy as np
import torch
import torch.nn as nn

# Device
try:
    import torch_directml as dml
    device = dml.device()
    print("Using DirectML:", device)
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

# Model
class FFN(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# Load artifacts
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

meta = torch.load("tfidf_meta.pth", map_location="cpu")
id2tag = meta["id2tag"]

with open("intents.json", "r", encoding="utf-8") as f:
    intents = {it["tag"]: it for it in json.load(f)["intents"]}

input_dim = len(vectorizer.get_feature_names_out())
model = FFN(input_dim, hidden=128, num_classes=len(id2tag), p=0.3).to(device)
state = torch.load("tfidf_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

def respond(text: str, threshold: float = 0.30) -> str:
    X = vectorizer.transform([text])
    if X.nnz == 0:
        return random.choice(intents.get("fallback", {"responses": ["Tell me more…"]})["responses"])
    x = torch.tensor(X.toarray(), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

    rawTag = id2tag[pred_idx]
    print(f"[debug] raw_pred={rawTag}, conf={conf:.2f}")
    tag = rawTag if conf >= threshold else ("fallback" if "fallback" in intents else rawTag)
    return random.choice(intents[tag]["responses"])

print("Chat Bot (quit with 'exit')")
while True:
    try:
        msg = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if msg.lower() in {"exit","quit"}:
        break
    print("Bot:", respond(msg))