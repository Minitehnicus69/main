import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.cnn import WaferCNN
from model.dataset import load_data, WaferDataset

PKL = sys.argv[1] if len(sys.argv) > 1 else "LSWMD.pkl"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else "weights.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

X, y, classes = load_data(PKL)          # aceeași ordine ca la antrenare (seed fix)
ck = torch.load(WEIGHTS, map_location=device, weights_only=False)
model = WaferCNN(len(ck["classes"])).to(device)
model.load_state_dict(ck["state"])
model.eval()

embs = []
with torch.no_grad():
    for xb, _ in DataLoader(WaferDataset(X, y), batch_size=256):
        embs.append(model.embed(xb.to(device)).cpu())
E = torch.cat(embs).numpy()
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)   # normalizat: cosine = produs scalar

np.savez_compressed(
    "history.npz",
    emb=E.astype(np.float32),
    labels=y.numpy(),
    maps=X.numpy(),
    classes=np.array([str(c) for c in ck["classes"]]),
)
print("salvat history.npz |", E.shape)