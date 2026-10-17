# llm/classifier.py
import os
import sys

import numpy as np
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

class WaferCNN(nn.Module):
    """
    CNN architecture compatible with the tensor structure in weights.pt.
    4 convolutional blocks followed by Global Average Pooling and a linear classifier.
    """
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            # Block 0: Conv 1 -> 16
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 1: Conv 16 -> 32
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 2: Conv 32 -> 64
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 3: Conv 64 -> 128
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.features:
            x = block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

_global_model: WaferCNN | None = None

def load_trained_model(weights_path: str | None = None) -> WaferCNN:
    """
    Loads weights from weights.pt into the CNN model.
    """
    global _global_model
    if _global_model is not None:
        return _global_model

    if weights_path is None:
        weights_path = os.environ.get("WEIGHTS_PATH", r"E:\download\weights.pt")

    model = WaferCNN(num_classes=len(CLASSES))
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state" in checkpoint:
                state_dict = checkpoint["state"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()
            model.load_state_dict(state_dict)
            model.eval()
            _global_model = model
            print(f"[WaferCNN] Model successfully loaded from {weights_path}")
            return _global_model
        except Exception as e:
            print(f"[WaferCNN] Warning loading weights: {e}")
    else:
        print(f"[WaferCNN] File {weights_path} not found. Using uninitialized model.")
    
    model.eval()
    _global_model = model
    return _global_model

def predict_wafer(wafer_matrix) -> dict:
    """
    Receives a wafer matrix (e.g. 26x26, 52x52) and returns the prediction and confidence score.
    """
    model = load_trained_model()
    arr = np.array(wafer_matrix, dtype=np.float32)
    if arr.ndim == 2:
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3:
        tensor = torch.tensor(arr).unsqueeze(0)
    else:
        tensor = torch.tensor(arr)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

    return {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
    }

if __name__ == "__main__":
    model = load_trained_model()
    dummy_wafer = np.random.rand(26, 26).tolist()
    res = predict_wafer(dummy_wafer)
    print("CNN model prediction test:", res)
