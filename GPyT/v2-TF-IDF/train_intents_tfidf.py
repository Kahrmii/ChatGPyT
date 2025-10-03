import json, random
from collections import Counter

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# device
try:
    import torch_directml as dml
    device = dml.device()
    print("Using DirectML:", device)
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

random.seed(42)
torch.manual_seed(42)

# Load data
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

assert len(samples) > 0, "No training samples found. Check intents.json"

# TF-IDF 
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    norm="l2",
    sublinear_tf=True
)
X_sparse = vectorizer.fit_transform(samples)
X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)

# Labels
tags = sorted(set(labels))
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for t, i in tag2id.items()}
y = torch.tensor([tag2id[l] for l in labels], dtype=torch.long)

print(f"TF-IDF vocab size: {X.shape[1]} | #classes: {len(tags)} | #samples: {len(samples)}")
print("Class counts:", Counter(labels))

counts = Counter(labels)
nClasses = len(counts)
useStratify = y if min(counts.values()) >= 2 else None

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=nClasses, random_state=42, stratify=useStratify
)

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

model = FFN(X.shape[1], hidden=128, num_classes=len(tags), p=0.3).to(device)

# Optimizer / loss / scheduler
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
crit = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=3, min_lr=1e-5
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
bestState = None
no_improve = 0
patience = 30
max_epochs = 300

for epoch in range(1, max_epochs + 1):
    model.train()
    for xb, yb in batchify(Xtr, ytr, bs=32):
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # eval
    model.eval()
    with torch.no_grad():
        trLogits = model(Xtr.to(device))
        trAcc, _ = accuracy(trLogits, ytr.to(device))

        vaLogits = model(Xva.to(device))
        vaAcc, vaPreds = accuracy(vaLogits, yva.to(device))
        vaPreds = vaPreds.cpu()

    scheduler.step(vaAcc)
    improved = vaAcc > best_va + 1e-4
    if improved:
        best_va = vaAcc
        bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 5 == 0 or improved:
        print(f"Epoch {epoch:03d} | train acc: {trAcc:.3f} | val acc: {vaAcc:.3f} | best: {best_va:.3f} | no_improve: {no_improve}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch} (best val acc {best_va:.3f})")
        break

if bestState is not None:
    model.load_state_dict(bestState)
all_labels = list(range(len(tags)))
print(classification_report(yva.numpy(), vaPreds.numpy(), labels=all_labels, target_names=tags, zero_division=0))

# save model + artifacts
torch.save(model.state_dict(), "tfidf_model.pth")
torch.save({"tag2id": tag2id, "id2tag": id2tag}, "tfidf_meta.pth")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved tfidf_model.pth, tfidf_meta.pth, and vectorizer.pkl")