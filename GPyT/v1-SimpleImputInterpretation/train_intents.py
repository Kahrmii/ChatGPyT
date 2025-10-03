import json, re, math, random
from collections import Counter

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Device
try:
    import torch_directml as dml # sometimes VSC says that it cant be resolved, but it works anyways ¯\_(ツ)_/¯ just dont touch it..
    device = dml.device()
    print("Using DirectML:", device)
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Text-Utils
def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-zäöüß0-9\s]", " ", s)
    return s.split()

def build_vocab(texts, min_freq=1, specials=["<unk>"]):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    itos = list(specials) + [w for w, c in counter.items() if c >= min_freq and w not in specials]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def bow_vector(tokens, stoi):
    v = torch.zeros(len(stoi), dtype=torch.float32)
    for tok in tokens:
        idx = stoi.get(tok)
        if idx is not None:
            v[idx] += 1.0
    n = v.norm(p=2)
    if n > 0:
        v = v / n
    return v

# load data
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

samples, labels = [], []
for it in intents:
    tag = it["tag"]
    if tag.lower() == "fallback":
        continue 
    for p in it["patterns"]:
        samples.append(p)
        labels.append(tag)

assert len(samples) > 0, "No training samples found."

# Vocab + Vectors
stoi, itos = build_vocab(samples, min_freq=1)
X = torch.stack([bow_vector(tokenize(s), stoi) for s in samples])
tags = sorted(set(labels))
tag2id = {t: i for i, t in enumerate(tags)}
y = torch.tensor([tag2id[l] for l in labels], dtype=torch.long)

# Train diag / Val Split
counts = Counter(labels)
print("Class counts:", counts)

min_per_class = min(counts.values())
use_stratify = y if min_per_class >= 2 else None
if use_stratify is None:
    print("[Warning] At least one intent has <2 examples - split without Stratify.")

n_classes = len(counts)
N = len(labels)
test_size_abs = n_classes

Xtr, Xva, ytr, yva = train_test_split(
    X, y,
    test_size=test_size_abs,
    random_state=42,
    stratify=use_stratify
)

# Model
class FFN(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

num_classes = len(tag2id)
model = FFN(X.shape[1], hidden=64, num_classes=num_classes, p=0.2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def batchify(X, y, bs=16):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i+bs]
        yield X[j], y[j]

best_va = 0.0
best_state = None
no_improve = 0
patience = 8             # stops after so many epochs without improvement
max_epochs = 200         # can be large - Early-Stopping ends earlier

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=3, min_lr=1e-5
)

for epoch in range(max_epochs):
    model.train()
    for xb, yb in batchify(Xtr, ytr, bs=16):
        xb = xb.to(device).float()
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()

    # validation
    model.eval()
    with torch.no_grad():
        logits = model(Xva.to(device).float())
        preds = logits.argmax(1).cpu()
        acc = (preds == yva).float().mean().item()

    scheduler.step(acc)

    improved = acc > best_va + 1e-4   # small tolerance
    if improved:
        best_va = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0 or improved:
        print(f"Epoch {epoch+1:03d} | val acc: {acc:.3f} | best: {best_va:.3f} | no_improve: {no_improve}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1} (best val acc {best_va:.3f})")
        break

# breplay best state
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), "model.pth")
torch.save(
    {"stoi": stoi, "itos": itos, "tag2id": tag2id, "id2tag": {v: k for k, v in tag2id.items()}},
    "metadata.pth",
)
print("Saved model.pth and metadata.pth")
