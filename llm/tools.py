import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.predict as _P
import model.similar as _S
import model.unknown as _U

log = logging.getLogger("tools")

try:
    CAUSES = json.loads((HERE / "causes.json").read_text(encoding="utf-8"))
except Exception:
    CAUSES = {}

_ready = False
_MAPS = None
_LOTS = {}
_WAFERS = {}
_cache = {}
_rag_warned = False


def _init():
    """Încarcă modelul, istoricul și loturile o singură dată (primul apel e lent)."""
    global _ready, _MAPS, _LOTS, _WAFERS
    if _ready:
        return
    _P._load(str(ROOT / "weights.pt"))
    _S._load(str(ROOT / "history.npz"))
    _U._load(str(ROOT / "history.npz"))
    _MAPS = _S._H["maps"]
    lots = json.loads((ROOT / "data" / "lots.json").read_text(encoding="utf-8"))
    _LOTS = {l["lot"]: l["wafere"] for l in lots}
    _WAFERS = {w["wafer_id"]: {**w, "lot": l["lot"]} for l in lots for w in l["wafere"]}
    _ready = True


def _resolve(wafer_id):
    """'B_w10' (din lots.json) sau un număr (history_id) -> history_id."""
    _init()
    if wafer_id in _WAFERS:
        return _WAFERS[wafer_id]["history_id"]
    if re.fullmatch(r"\d+", str(wafer_id)):
        return int(wafer_id)
    raise ValueError(f"wafer necunoscut: {wafer_id}")


def lot_of(wafer_id):
    _init()
    return _WAFERS.get(wafer_id, {}).get("lot")


def tool_clasifica(wafer_id):
    _init()
    if wafer_id in _cache:
        return _cache[wafer_id]
    hid = _resolve(wafer_id)
    w = _MAPS[hid]
    p, u = _P.predict(w), _U.is_unknown(w)
    out = {"class": p["clasa"], "confidence": p["incredere"], "unknown": u["necunoscut"],
           "nearest_class": u["cel_mai_apropiat"], "history_id": hid}
    _cache[wafer_id] = out
    return out


def tool_verifica_drift(lot_id, wafer_id=None):
    """Regulă: în ultimele 5 wafere (până la wafer_id), fără 'none', aceeași clasă de >= 3 ori."""
    _init()
    wafere = _LOTS.get(lot_id)
    if not wafere:
        return {"alarm": False, "repeated_class": None, "affected": 0, "window": [], "error": f"lot necunoscut: {lot_id}"}
    ids = [w["wafer_id"] for w in wafere]
    end = ids.index(wafer_id) + 1 if wafer_id in ids else len(ids)
    window = [tool_clasifica(i)["class"] for i in ids[max(0, end - 5):end]]
    fer = [c for c in window if c != "none"]
    if fer:
        c, n = Counter(fer).most_common(1)[0]
        if n >= 3:
            return {"alarm": True, "repeated_class": c, "affected": n, "window": window}
    return {"alarm": False, "repeated_class": None, "affected": 0, "window": window}


def tool_gaseste_similare(wafer_id, k=5):
    _init()
    hid = _resolve(wafer_id)
    res = [r for r in _S.similar(_MAPS[hid], k + 1) if r["id"] != hid][:k]
    return [{"wafer_id": f"history_{r['id']}", "history_id": r["id"], "clasa": r["clasa"], "scor": r["scor"]} for r in res]


def _first(d, keys, default=None):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


def _normalize(item):
    """Acceptă dict-uri cu chei diferite, șiruri sau tuple, și le aduce la {doc, page, text}."""
    if isinstance(item, str):
        return {"doc": "rag", "page": None, "text": item}
    if isinstance(item, (tuple, list)) and item:
        return {"doc": "rag", "page": None, "text": str(item[0])}
    if isinstance(item, dict):
        return {"doc": _first(item, ("doc", "sursa", "source", "file", "filename", "document"), "rag"),
                "page": _first(item, ("page", "pagina", "page_number", "p")),
                "text": str(_first(item, ("text", "content", "chunk", "snippet"), ""))}
    return {"doc": "rag", "page": None, "text": str(item)}


def _rag_search(query, k):
    from llm import rag
    for name in ("cauta_manual", "search", "query", "retrieve", "search_manual", "cauta"):
        fn = getattr(rag, name, None)
        if callable(fn):
            try:
                return fn(query, k)
            except TypeError:
                return fn(query)
    raise AttributeError("rag.py nu are nicio funcție de căutare cunoscută; funcții găsite: "
                         + ", ".join(n for n in dir(rag) if not n.startswith("_")))


def tool_cauta_manual(query, k=3):
    """Folosește rag.py dacă merge; altfel cade pe causes.json (și scrie motivul în log, o singură dată)."""
    global _rag_warned
    try:
        out = [_normalize(r) for r in _rag_search(query, k)]
        if out:
            return out
    except Exception as e:
        if not _rag_warned:
            log.warning("rag.py nu răspunde, folosesc causes.json: %r", e)
            _rag_warned = True
    info = CAUSES.get(query)
    return [{"doc": "causes.json", "page": None, "text": json.dumps(info, ensure_ascii=False)}] if info else []


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


OPENAI_TOOLS_DEFINITIONS = [
    _fn("clasifica", "Clasifică un wafer map (clasă, încredere, dacă e pattern necunoscut).",
        {"wafer_id": {"type": "string"}}, ["wafer_id"]),
    _fn("verifica_drift", "Verifică dacă lotul are alarmă de drift până la un wafer.",
        {"lot_id": {"type": "string"}, "wafer_id": {"type": "string"}}, ["lot_id"]),
    _fn("gaseste_similare", "Găsește cele mai asemănătoare wafere din istoric.",
        {"wafer_id": {"type": "string"}}, ["wafer_id"]),
    _fn("cauta_manual", "Caută în manuale cauze și proceduri pentru un tip de defect.",
        {"query": {"type": "string"}}, ["query"]),
]
TOOL_ROUTER = {"clasifica": tool_clasifica, "verifica_drift": tool_verifica_drift,
               "gaseste_similare": tool_gaseste_similare, "cauta_manual": tool_cauta_manual}