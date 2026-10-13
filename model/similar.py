import numpy as np
from model.predict import embed

_H = None

def _load(path="history.npz"):
    global _H
    if _H is None:
        _H = dict(np.load(path))     # dict, ca să nu se decomprime la fiecare acces
    return _H

def similar(wafer, k=5):
    H = _load()
    q = np.array(embed(wafer), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    scores = H["emb"] @ q
    top = np.argsort(-scores)[:k]
    return [
        {"id": int(i), "clasa": str(H["classes"][H["labels"][i]]), "scor": round(float(scores[i]), 3)}
        for i in top
    ]