import numpy as np
from model.predict import embed

_C = None

def _dist(Q, mu, P):
    d = np.stack([(((Q - m) @ P) * (Q - m)).sum(1) for m in mu])
    return d.min(0), d.argmin(0)

def _load(path="history.npz", pct=99):
    global _C
    if _C is None:
        H = np.load(path)
        E, y = H["emb"].astype(np.float64), H["labels"]
        n = int(y.max()) + 1
        mu = np.stack([E[y == c].mean(0) for c in range(n)])
        X = E - mu[y]
        P = np.linalg.inv(X.T @ X / len(E) + 1e-6 * np.eye(E.shape[1]))
        d, _ = _dist(E, mu, P)
        _C = {"mu": mu, "P": P, "thr": float(np.percentile(d, pct)),
              "classes": [str(c) for c in H["classes"]]}

def is_unknown(wafer):
    _load()
    q = np.array(embed(wafer), dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-9
    d, b = _dist(q[None], _C["mu"], _C["P"])
    return {
        "necunoscut": bool(d[0] > _C["thr"]),
        "scor": round(float(d[0]), 1),
        "prag": round(_C["thr"], 1),
        "cel_mai_apropiat": _C["classes"][int(b[0])],
    }