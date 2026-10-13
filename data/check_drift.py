import json
import numpy as np
from collections import Counter
from model.predict import predict

H = np.load("history.npz")
maps = H["maps"]
for lot in json.load(open("data/lots.json")):
    preds = [predict(maps[w["history_id"]])["clasa"] for w in lot["wafere"]]
    alarma = None
    for k in range(len(preds)):
        fer = [p for p in preds[max(0, k - 4):k + 1] if p != "none"]
        if fer:
            c, n = Counter(fer).most_common(1)[0]
            if n >= 3:
                alarma = (f"wafer {k+1}", c)
                break
    greseli = [(w["wafer_id"], w["clasa_reala"], p)
               for w, p in zip(lot["wafere"], preds) if w["clasa_reala"] != p]
    print(lot["lot"], "alarmă:", alarma or "nu")
    print("   greșeli:", greseli)