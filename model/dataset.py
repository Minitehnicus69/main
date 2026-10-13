import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# LSWMD.pkl e făcut cu un pandas vechi; fără linia asta poate da "No module named pandas.indexes"
import pandas.core.indexes
sys.modules.setdefault("pandas.indexes", pandas.core.indexes)

def resize_wafer(w, size=64):
    """waferMap (valori 0/1/2, dimensiuni diferite) -> tensor uint8 size x size"""
    t = torch.from_numpy(np.asarray(w, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t[0, 0].to(torch.uint8)

def load_data(path, max_none=5000, size=64, seed=42):
    df = pd.read_pickle(path)
    df = df[df["failureType"].apply(lambda x: len(x) > 0)].copy()
    df["label"] = df["failureType"].apply(lambda x: x[0][0])

    # limităm clasa "none", altfel domină tot setul
    none = df[df["label"] == "none"].sample(n=max_none, random_state=seed)
    df = pd.concat([df[df["label"] != "none"], none])

    classes = sorted(df["label"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    X = torch.stack([resize_wafer(w, size) for w in df["waferMap"]])  # N x 64 x 64, uint8
    y = torch.tensor(df["label"].map(idx).values, dtype=torch.long)
    return X, y, classes

class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].float().div(2.0).unsqueeze(0), self.y[i]  # 0/1/2 -> 0/0.5/1, 1 canal