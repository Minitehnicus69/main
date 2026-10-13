import torch.nn as nn

def bloc(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class WaferCNN(nn.Module):
    def __init__(self, n_clase=9):
        super().__init__()
        self.features = nn.Sequential(bloc(1, 16), bloc(16, 32), bloc(32, 64), bloc(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_clase)

    def embed(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.fc(self.embed(x))