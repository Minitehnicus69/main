import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from model.cnn import WaferCNN
from model.dataset import load_data, WaferDataset

PKL = sys.argv[1] if len(sys.argv) > 1 else "LSWMD.pkl"
EPOCHS, BATCH, LR = 25, 128, 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

X, y, classes = load_data(PKL)
classes = [str(c) for c in classes]
print("clase:", classes, "| total:", len(y))

idx = np.arange(len(y))
tr, rest = train_test_split(idx, test_size=0.3, stratify=y.numpy(), random_state=42)
va, te = train_test_split(rest, test_size=0.5, stratify=y[rest].numpy(), random_state=42)

train_dl = DataLoader(WaferDataset(X[tr], y[tr]), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(WaferDataset(X[va], y[va]), batch_size=BATCH)
test_dl = DataLoader(WaferDataset(X[te], y[te]), batch_size=BATCH)

# ponderi mai blânde (rădăcină pătrată), ca să nu forțeze prea tare clasele rare
counts = np.bincount(y[tr].numpy(), minlength=len(classes))
w = (len(tr) / (len(classes) * np.maximum(counts, 1))) ** 0.5
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))

model = WaferCNN(len(classes)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def augment(xb):
    # un wafer rotit sau oglindit e tot același tip de defect
    if np.random.rand() < 0.5: xb = torch.flip(xb, dims=[3])
    if np.random.rand() < 0.5: xb = torch.flip(xb, dims=[2])
    return torch.rot90(xb, int(np.random.randint(4)), dims=[2, 3])

def evaluate(dl):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for xb, yb in dl:
            pred += model(xb.to(device)).argmax(1).cpu().tolist()
            true += yb.tolist()
    return true, pred

best_f1, best_state = -1, None
for ep in range(EPOCHS):
    model.train()
    total = 0
    for xb, yb in train_dl:
        xb, yb = augment(xb).to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
    sched.step()
    t, p = evaluate(val_dl)
    f1 = f1_score(t, p, average="macro", zero_division=0)
    print(f"epoca {ep+1}/{EPOCHS} | loss {total/len(tr):.4f} | val acc {np.mean(np.array(t)==np.array(p)):.3f} | val F1 {f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)   # cea mai bună epocă pe validare, nu ultima
t, p = evaluate(test_dl)
print(f"\nTEST final (cel mai bun model, val F1 {best_f1:.3f})")
print(classification_report(t, p, target_names=classes, zero_division=0))
torch.save({"state": best_state, "classes": classes}, "weights.pt")
print("salvat: weights.pt")