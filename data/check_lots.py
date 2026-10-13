import json
import numpy as np
from model.predict import predict

H = np.load("history.npz")
maps = H["maps"]
for lot in json.load(open("data/lots.json")):
    sec = []
    for w in lot["wafere"]:
        p = predict(maps[w["history_id"]])
        sec.append(p["clasa"])
    ok = sum(a == w["clasa_reala"] for a, w in zip(sec, lot["wafere"]))
    print(lot["lot"], f"corect {ok}/25")
    print("  ", " ".join(s[:4] for s in sec))