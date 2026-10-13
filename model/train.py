import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model.cnn import WaferCNN
from model.dataset import load_data, WaferDataset

PKL = sys.argv[1] if len(sys.argv) > 1 else "LSWMD.pkl"
EPOCHS, BATCH, LR = 15, 128, 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

X, y, classes = load_data(PKL)
print("clase:", classes, "| total:", len(y))

idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=42)
train_dl = DataLoader(WaferDataset(X[tr], y[tr]), batch_size=BATCH, shuffle=True)
test_dl = DataLoader(WaferDataset(X[te], y[te]), batch_size=BATCH)

# ponderi pe clase: clasele rare contează mai mult în loss
counts = np.bincount(y[tr].numpy(), minlength=len(classes))
w = torch.tensor(len(tr) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=w)

model = WaferCNN(len(classes)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate():
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            pred += model(xb.to(device)).argmax(1).cpu().tolist()
            true += yb.tolist()
    return true, pred

for ep in range(EPOCHS):
    model.train()
    total = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
    true, pred = evaluate()
    acc = np.mean(np.array(true) == np.array(pred))
    print(f"epoca {ep+1}/{EPOCHS} | loss {total/len(tr):.4f} | acc test {acc:.3f}")

true, pred = evaluate()
print(classification_report(true, pred, target_names=classes, zero_division=0))
torch.save({"state": model.state_dict(), "classes": classes}, "weights.pt")
print("salvat: weights.pt")