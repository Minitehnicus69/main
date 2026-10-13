import numpy as np
from model.predict import predict
from model.gradcam import gradcam_png

H = np.load("history.npz")
maps, labels, classes = H["maps"], H["labels"], H["classes"]
rng = np.random.default_rng(1)

for c in [0, 3, 1, 7]:
    for i in rng.permutation(np.where(labels == c)[0])[:50]:
        p = predict(maps[i])
        if p["clasa"] == classes[c] and p["incredere"] > 0.95:
            print(classes[c], "| id", int(i), "|", p)
            gradcam_png(maps[i], f"gc_{classes[c]}.png")
            break