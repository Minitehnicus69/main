import numpy as np
import torch
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

from model.cnn import WaferCNN

H = np.load("history.npz")
X, y = H["maps"], H["labels"]
classes = [str(c) for c in H["classes"]]

# același split ca în train.py
idx = np.arange(len(y))
tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=42)

yy, xx = np.indices((64, 64))
cy = cx = 31.5
R = np.hypot(yy - cy, xx - cx) / 32
A = np.arctan2(yy - cy, xx - cx)
ring_id = np.minimum((R * 5).astype(int), 4)
sec_id = ((A + np.pi) / (2 * np.pi) * 8).astype(int) % 8

def feats(w):
    die, bad = w > 0, w == 2
    n_die = max(int(die.sum()), 1)
    f = [bad.sum() / n_die]                                    # densitate totală
    for r in range(5):                                         # densitate pe 5 inele
        m = die & (ring_id == r)
        f.append(bad[m].sum() / max(int(m.sum()), 1))
    for s in range(8):                                         # densitate pe 8 sectoare
        m = die & (sec_id == s)
        f.append(bad[m].sum() / max(int(m.sum()), 1))
    if bad.any():                                              # unde e centrul defectelor și cât sunt de împrăștiate
        f += [(yy[bad].mean() - cy) / 32, (xx[bad].mean() - cx) / 32,
              yy[bad].std() / 32, xx[bad].std() / 32]
    else:
        f += [0, 0, 0, 0]
    lab, n = ndimage.label(bad)                                # componente conexe
    f += [n, (np.bincount(lab.ravel())[1:].max() / n_die) if n else 0]
    return f

print("calculez features...")
F = np.array([feats(w) for w in X])

print("antrenez Random Forest...")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(F[tr], y[tr])
p_rf = rf.predict(F[te])

print("evaluez CNN...")
ck = torch.load("weights.pt", map_location="cpu", weights_only=False)
cnn = WaferCNN(len(ck["classes"]))
cnn.load_state_dict(ck["state"])
cnn.eval()
xt = torch.from_numpy(X[te]).float().div(2.0).unsqueeze(1)
with torch.no_grad():
    p_cnn = torch.cat([cnn(xt[i:i + 512]).argmax(1) for i in range(0, len(xt), 512)]).numpy()

rep_rf = classification_report(y[te], p_rf, target_names=classes, output_dict=True, zero_division=0)
rep_cnn = classification_report(y[te], p_cnn, target_names=classes, output_dict=True, zero_division=0)

lines = [
    "# Rezultate pe setul de test (4578 wafere, neîntâlnite la antrenare)\n",
    "| Model | Acuratețe | Macro F1 |", "|---|---|---|",
    f"| Random Forest (features simple) | {accuracy_score(y[te], p_rf):.3f} | {f1_score(y[te], p_rf, average='macro'):.3f} |",
    f"| CNN | {accuracy_score(y[te], p_cnn):.3f} | {f1_score(y[te], p_cnn, average='macro'):.3f} |",
    "\n## F1 pe clasă\n", "| Clasă | RF | CNN | Nr. test |", "|---|---|---|---|",
]
for c in classes:
    lines.append(f"| {c} | {rep_rf[c]['f1-score']:.2f} | {rep_cnn[c]['f1-score']:.2f} | {int(rep_cnn[c]['support'])} |")
open("data/metrics.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))