import numpy as np
from model.unknown import is_unknown
from model.predict import predict

H = np.load("history.npz")
maps, labels = H["maps"], H["labels"]
base = maps[np.where(labels == 8)[0][0]].copy()      # 8 = none
yy, xx = np.indices(base.shape)

def make(mask):
    w = base.copy()
    w[(base == 1) & mask] = 2                        # die bun -> picat, doar în zona mask
    return w

tests = {
    "jumătate picată": make(xx < 32),
    "bandă diagonală lată": make(abs(xx - yy) < 8),
    "tablă de șah": make((xx // 4 + yy // 4) % 2 == 0),
    "inel la jumătatea razei": make(abs(np.hypot(yy - 32, xx - 32) - 16) < 3),
}
for nume, w in tests.items():
    print(f"{nume:26}", predict(w), is_unknown(w))

rng = np.random.default_rng(0)
idx = rng.choice(len(labels), 2000, replace=False)
fals = np.mean([is_unknown(maps[i])["necunoscut"] for i in idx])
print("\nwafere reale marcate necunoscute:", round(float(fals), 3))