import json, re, random
import torch
import torch.nn as nn

try:
    import torch_directml as dml
    device = dml.device()
    print("Using DirectML:", device)
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-zäöüß0-9\s]", " ", s)
    return s.split()

def bow_vector(tokens, stoi):
    v = torch.zeros(len(stoi), dtype=torch.float32)
    for tok in tokens:
        idx = stoi.get(tok)
        if idx is not None:
            v[idx] += 1.0
    n = v.norm(p=2)
    if n > 0: v = v / n
    return v
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

meta = torch.load("metadata.pth", map_location="cpu")
stoi, id2tag = meta["stoi"], meta["id2tag"]

with open("intents.json", "r", encoding="utf-8") as f:
    intents = {it["tag"]: it for it in json.load(f)["intents"]}

model = FFN(len(stoi), hidden=128, num_classes=len(id2tag), p=0.2).to(device)
state = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

def respond(text: str, threshold: float = 0.30) -> str:
    tokens = tokenize(text)
    # Immediate fallback if no known tokens appear
    if not any(t in stoi for t in tokens):
        return random.choice(intents.get("fallback", {"responses": ["Tell me more…"]})["responses"])
    x = bow_vector(tokens, stoi).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        conf, pred_idx = torch.max(probs, dim=0)
    raw_tag = id2tag[int(pred_idx)]
    print(f"[debug] raw_pred={raw_tag}, conf={conf.item():.2f}")
    tag = raw_tag if conf.item() >= threshold else ("fallback" if "fallback" in intents else raw_tag)
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