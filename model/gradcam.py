import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model.predict as P


def gradcam(wafer, class_idx=None):
    """Întoarce (hartă 64x64 cu valori 0..1, numele clasei explicate)."""
    P._load()
    m = P._model
    x = P._prep(wafer)
    layer = m.features[3][2]          # ReLU-ul ultimului bloc (hartă 8x8)
    feats, grads = {}, {}

    def fwd(mod, inp, out):
        feats["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    m.zero_grad()
    logits = m(x)
    i = int(logits.argmax(1)) if class_idx is None else class_idx
    logits[0, i].backward()
    h.remove()

    w = grads["g"].mean(dim=(2, 3), keepdim=True)            # importanța fiecărui canal
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))      # combinație ponderată
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam.detach().numpy(), P._classes[i]


def gradcam_png(wafer, path="gradcam.png"):
    """Salvează hartă + suprapunerea Grad-CAM și întoarce calea."""
    cam, clasa = gradcam(wafer)
    img = P._prep(wafer)[0, 0].numpy() * 2      # înapoi la 0/1/2
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    ax[0].imshow(img, cmap="viridis"); ax[0].set_title("Wafer map")
    ax[1].imshow(img, cmap="gray"); ax[1].imshow(cam, cmap="jet", alpha=0.5)
    ax[1].set_title(f"De ce: {clasa}")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path