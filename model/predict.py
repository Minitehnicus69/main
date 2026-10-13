import torch
from model.cnn import WaferCNN
from model.dataset import resize_wafer

_model, _classes = None, None

def _load(path="weights.pt"):
    global _model, _classes
    if _model is None:
        ck = torch.load(path, map_location="cpu")
        _classes = ck["classes"]
        _model = WaferCNN(len(_classes))
        _model.load_state_dict(ck["state"])
        _model.eval()

def _prep(wafer):
    return resize_wafer(wafer)[None, None].float().div(2.0)

def predict(wafer):
    _load()
    with torch.no_grad():
        p = torch.softmax(_model(_prep(wafer)), dim=1)[0]
    i = int(p.argmax())
    return {"clasa": _classes[i], "incredere": round(float(p[i]), 3)}

def embed(wafer):
    _load()
    with torch.no_grad():
        return _model.embed(_prep(wafer))[0].tolist()