import json
import numpy as np

H = np.load("history.npz")
labels = H["labels"]
classes = [str(c) for c in H["classes"]]
rng = np.random.default_rng(7)

def pick(cls, n):
    pool = np.where(labels == classes.index(cls))[0]
    return rng.choice(pool, n, replace=False).tolist()

def lot(name, echipament, secventa):
    """secventa = lista de clase, una pe wafer, în ordinea testării"""
    ids = {c: pick(c, secventa.count(c)) for c in set(secventa)}
    wafere = []
    for k, c in enumerate(secventa):
        wafere.append({
            "wafer_id": f"{name}_w{k+1:02d}",
            "history_id": int(ids[c].pop()),
            "echipament": echipament,   # INVENTAT, nu vine din WM-811K
            "clasa_reala": c,
        })
    return {"lot": name, "wafere": wafere}

lots = [
    lot("A", "ETCH-01", ["none"] * 25),                                   # lot normal
    lot("B", "ETCH-03", ["none"] * 7 + ["Edge-Ring"] * 18),               # drift Edge-Ring din wafer-ul 8
    lot("C", "CMP-02", ["none"] * 12 + ["Center"] * 13),                  # drift Center din wafer-ul 13
    lot("D", "LITHO-01", ["none"] * 10 + ["Scratch"] + ["none"] * 9 + ["Scratch"] + ["none"] * 4),  # defecte izolate, NU drift
]
json.dump(lots, open("data/lots.json", "w"), indent=1)
print("salvat data/lots.json |", sum(len(l["wafere"]) for l in lots), "wafere")