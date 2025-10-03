import json, re, math, random
from collections import Counter

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    import torch_directml as dml
    device = dml.device()
    print("Using DirectML:", device)
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# ----- Tokenizer & BoW (OOV ignored + L2 norm)
def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-zäöüß0-9\s]", " ", s)
    return s.split()

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    itos = [w for w, c in counter.items() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def bow_vector(tokens, stoi):
    v = torch.zeros(len(stoi), dtype=torch.float32)
    for tok in tokens:
        idx = stoi.get(tok)  # None means OOV -> ignore
        if idx is not None:
            v[idx] += 1.0
    n = v.norm(p=2)
    if n > 0:
        v = v / n  # L2 normalization stabilizes training
    return v

# ----- Load data (do NOT train on fallback)
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

assert len(samples) > 0, "No training samples found. Check intents.json."

# ----- Build vocab & features
stoi, itos = build_vocab(samples, min_freq=1)
X = torch.stack([bow_vector(tokenize(s), stoi) for s in samples])
tags = sorted(set(labels))
tag2id = {t: i for i, t in enumerate(tags)}
y = torch.tensor([tag2id[l] for l in labels], dtype=torch.long)

print(f"Vocab size: {len(stoi)} | #classes: {len(tags)} | #samples: {len(samples)}")
print("Sanity check:", "'hi' in vocab" if "hi" in stoi else "'hi' NOT in vocab")

# ----- Split (make val size = #classes; stratify if possible)
counts = Counter(labels)
print("Class counts:", counts)
n_classes = len(counts)
test_size_abs = n_classes
use_stratify = y if min(counts.values()) >= 2 else None

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=test_size_abs, random_state=42, stratify=use_stratify
)

# ----- Model
class FFN(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=10, p=0.0):
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
model = FFN(X.shape[1], hidden=128, num_classes=num_classes, p=0.2).to(device)

# Conservative optimizer
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=50, min_lr=1e-5
)

def batchify(X, y, bs=32):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i+bs]
        yield X[j], y[j]

def accuracy(logits, y_true):
    preds = logits.argmax(1)
    return (preds == y_true).float().mean().item(), preds

best_va = 0.0
best_state = None
no_improve = 0
patience = 30
max_epochs = 300

for epoch in range(1, max_epochs + 1):
    # Train
    model.train()
    for xb, yb in batchify(Xtr, ytr, bs=32):
        xb = xb.to(device).float()
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Evaluate on train/val (full sets are small)
    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr.to(device).float())
        tr_acc, _ = accuracy(tr_logits, ytr.to(device))

        va_logits = model(Xva.to(device).float())
        va_acc, va_preds = accuracy(va_logits, yva.to(device))
        va_preds = va_preds.cpu()

    scheduler.step(va_acc)

    improved = va_acc > best_va + 1e-4
    if improved:
        best_va = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 5 == 0 or improved:
        print(f"Epoch {epoch:03d} | train acc: {tr_acc:.3f} | val acc: {va_acc:.3f} | best: {best_va:.3f} | no_improve: {no_improve}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch} (best val acc {best_va:.3f})")
        break

# Load best checkpoint
if best_state is not None:
    model.load_state_dict(best_state)

# Validation diagnostics
print("\nValidation report:")
print(classification_report(yva.numpy(), va_preds.numpy(), target_names=tags, zero_division=0))

# Save artifacts
torch.save(model.state_dict(), "model.pth")
torch.save({"stoi": stoi, "itos": itos, "tag2id": tag2id, "id2tag": {v: k for k, v in tag2id.items()}}, "metadata.pth")
print("Saved model.pth and metadata.pth")