"""Legătura dintre API și modulele echipei: model/ (#1), detectorul de necunoscut (#2), agentul llm/ (#3).

Pașii din analyze() urmează llm/agent.analyze(), dar:
- acceptă și o matrice trimisă de UI (wafer care nu e în istoric), nu doar un ID;
- caută similare și în wafere confirmate prin /feedback;
- adaugă harta, Grad-CAM și motorul de cost.
Textul (explicație, acțiune) vine din funcțiile agentului, deci cache-ul LLM rămâne valabil.
"""
import base64
import io
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Modelul de embedding pentru RAG (all-MiniLM-L6-v2) se descarcă o singură dată, apoi vine din cache-ul HF.
# Pe un server fără internet: HF_HUB_OFFLINE=1 și cache-ul copiat dinainte.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger("api")

MOCK_FILE = Path(__file__).resolve().parent / "mock_response.json"
DB_FILE = Path(os.environ.get("FEEDBACK_DB", Path(__file__).resolve().parent / "feedback.db"))
LOT_SIZE = 25          # loturile din data/lots.json au 25 de wafere
DRIFT_WINDOW, DRIFT_MIN = 5, 3

_lock = threading.Lock()   # modelul (și hook-urile Grad-CAM) nu sunt sigure la apeluri paralele
_state = {"ready": False, "error": None, "rag": False, "rag_error": None}
T = A = P = S = U = G = None


def _boot():
    """Încarcă modelul, istoricul și agentul o singură dată. Dacă lipsește ceva, API-ul merge pe mock."""
    global T, A, P, S, U, G
    if _state["ready"] or _state["error"]:
        return _state["ready"]
    try:
        missing = [f for f in ("weights.pt", "history.npz") if not (ROOT / f).exists()]
        if missing:
            raise FileNotFoundError("lipsesc " + ", ".join(missing) + " din rădăcina proiectului")
        from llm import tools as T_
        from llm import agent as A_
        import model.predict as P_
        import model.similar as S_
        import model.unknown as U_
        import model.gradcam as G_
        T, A, P, S, U, G = T_, A_, P_, S_, U_, G_
        T._init()
        _state["ready"] = True
        log.info("model încărcat: %d wafere în istoric, loturi %s", len(T._MAPS), list(T._LOTS))
    except Exception as e:
        _state["error"] = repr(e)
        log.warning("model indisponibil, răspund cu mock: %r", e)
        return False
    _boot_rag()
    return _state["ready"]


def _boot_rag():
    """Construiește indexul RAG la pornire, nu la prima cerere. Dacă nu merge, agentul folosește causes.json."""
    try:
        from llm import rag
        if rag.index is None:
            rag.build_index([])
        _state["rag"] = True
        log.info("RAG pornit: %d fragmente în index", rag.index.ntotal)
    except Exception as e:
        _state["rag_error"] = repr(e)
        log.warning("RAG indisponibil, folosesc causes.json: %r", e)


def status():
    ready = _boot()
    return {"mode": "model" if ready else "mock", "error": _state["error"],
            "history": len(T._MAPS) if ready else 0,
            "lots": {k: len(v) for k, v in T._LOTS.items()} if ready else {},
            "llm_model": A.MODEL_NAME if ready else None,
            "rag": _state["rag"], "rag_error": _state["rag_error"]}


def mock_response(wafer_id):
    data = json.loads(MOCK_FILE.read_text(encoding="utf-8"))
    data["wafer_id"] = wafer_id
    return data


def lots():
    _boot()
    return [{"lot_id": k, "n_wafers": len(v), "echipament": v[0].get("echipament") if v else None,
             "wafer_ids": [w["wafer_id"] for w in v]} for k, v in T._LOTS.items()]


# ---------- feedback (SQLite) ----------

def _db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, wafer_id TEXT, history_id INTEGER,
        predicted_class TEXT, correct INTEGER, true_class TEXT, comment TEXT, emb BLOB, map BLOB)""")
    return con


def _label(predicted, correct, true_class):
    """Eticheta confirmată de tehnician; None dacă nu știm clasa corectă."""
    lab = predicted if correct else true_class
    return lab if lab and lab != "necunoscut" else None


def _feedback_state():
    """(corecții pe history_id, wafere noi confirmate cu embedding și hartă). Ultimul feedback câștigă."""
    corr, pool = {}, {}
    with _db() as con:
        rows = con.execute("SELECT id, wafer_id, history_id, predicted_class, correct, true_class, emb, map "
                           "FROM feedback ORDER BY id").fetchall()
    for fid, wid, hid, pred, ok, true_c, emb, mp in rows:
        lab = _label(pred, ok, true_c)
        if hid is not None:
            if lab:
                corr[hid] = lab
        elif emb is not None:
            if lab:
                pool[wid] = {"id": fid, "label": lab, "emb": np.frombuffer(emb, dtype=np.float32),
                             "map": np.frombuffer(mp, dtype=np.uint8).reshape(64, 64)}
            else:
                pool.pop(wid, None)
    return corr, list(pool.values())


def add_feedback(req):
    ready = _boot()
    hid, emb, mp = None, None, None
    if ready:
        if req.wafer is not None:
            mp = _to64(req.wafer)
        else:
            try:
                hid = T._resolve(req.wafer_id)
            except ValueError:
                pass
        if mp is not None:
            with _lock:
                emb = _unit(P.embed(mp))
    with _db() as con:
        cur = con.execute("INSERT INTO feedback (ts, wafer_id, history_id, predicted_class, correct, true_class, "
                          "comment, emb, map) VALUES (?,?,?,?,?,?,?,?,?)",
                          (time.time(), req.wafer_id, hid, req.predicted_class, int(req.correct), req.true_class,
                           req.comment, emb.tobytes() if emb is not None else None,
                           mp.tobytes() if mp is not None else None))
        fid = cur.lastrowid
    used = _label(req.predicted_class, req.correct, req.true_class) is not None and (hid is not None or mp is not None)
    return {"id": fid, "stored": True, "used_for_similar": used}


def list_feedback(limit=50):
    with _db() as con:
        rows = con.execute("SELECT id, ts, wafer_id, history_id, predicted_class, correct, true_class, comment "
                           "FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    keys = ("id", "ts", "wafer_id", "history_id", "predicted_class", "correct", "true_class", "comment")
    return [dict(zip(keys, r), correct=bool(r[5])) for r in rows]


# ---------- model ----------

def _to64(wafer):
    from model.dataset import resize_wafer
    return resize_wafer(np.asarray(wafer, dtype=np.uint8)).numpy()


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _similar(m, k=5, exclude_hid=None):
    """Top-k din istoric + wafere confirmate prin feedback, cu etichetele corectate de tehnicieni."""
    corr, pool = _feedback_state()
    q = _unit(P.embed(m))
    H = S._load()
    scores = H["emb"] @ q
    top = np.argsort(-scores)[:k + 1]
    cand = [{"wafer_id": f"history_{i}", "history_id": int(i),
             "clasa": corr.get(int(i), str(H["classes"][H["labels"][i]])), "scor": round(float(scores[i]), 3)}
            for i in top if int(i) != exclude_hid]
    for f in pool:
        cand.append({"wafer_id": f"feedback_{f['id']}", "history_id": None, "clasa": f["label"],
                     "scor": round(float(f["emb"] @ q), 3), "_map": f["map"]})
    return sorted(cand, key=lambda c: -c["scor"])[:k]


def _drift_from(window):
    """Regula de drift: în ultimele 5 wafere, fără 'none', aceeași clasă de cel puțin 3 ori."""
    fer = [c for c in window if c != "none"]
    if fer:
        c, n = Counter(fer).most_common(1)[0]
        if n >= DRIFT_MIN:
            return {"alarm": True, "repeated_class": c, "affected": n, "window": list(window)}
    return {"alarm": False, "repeated_class": None, "affected": 0, "window": list(window)}


_lot_state = {}   # lot_id -> OrderedDict(wafer_id -> clasă), pentru wafere trimise ca matrice


def _track(lot, wafer_id, cls):
    hist = _lot_state.setdefault(lot, OrderedDict())
    hist[wafer_id] = cls
    while len(hist) > 50:
        hist.popitem(last=False)
    return _drift_from(list(hist.values())[-DRIFT_WINDOW:])


def _window(lot, wafer_id):
    ids = list(_lot_state[lot])
    k = ids.index(wafer_id)
    return _drift_from([_lot_state[lot][i] for i in ids[max(0, k - DRIFT_WINDOW + 1):k + 1]])


def sample_wafers():
    """Wafere reale din istoric, câte unul pe clasă: fișierul de exemplu pentru pagina de upload."""
    from api import upload
    return upload.sample_wafers(T._MAPS, S._H["labels"], S._H["classes"])


def _gradcam_png(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cam, cls = G.gradcam(m)
    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="#0b1020")
    ax.imshow(m, cmap="gray", vmin=0, vmax=2, interpolation="nearest")
    ax.imshow(cam, cmap="jet", alpha=0.5)
    ax.set_title(f"Grad-CAM: {cls}", color="#eef3ff", fontsize=10)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- cost ----------

def cost(p_defect, wafere_ramase, params):
    rem = params.wafere_ramase if params.wafere_ramase is not None else wafere_ramase
    rebut = rem * params.pret_wafer * p_defect
    oprire = params.ore_oprire * params.cost_ora
    return {"cost_rebut": round(rebut, 2), "cost_oprire": round(oprire, 2), "p_defect": round(p_defect, 3),
            "wafere_ramase": int(rem), "recomandare": "opreste" if rebut > oprire else "continua"}


def _p_defect(clf, drift):
    """Probabilitatea ca wafer-ele următoare să iasă rebut = rata defectului dominant în ultimele 5 wafere.
    Un defect izolat (și un pattern necunoscut) dă 1/5, nu oprește singur linia; o alarmă de drift dă cel puțin 3/5."""
    if drift["window"]:
        fer = Counter(c for c in drift["window"] if c != "none")
        return fer.most_common(1)[0][1] / DRIFT_WINDOW if fer else 0.0
    # fără context de lot: doar wafer-ul curent
    return clf["confidence"] / DRIFT_WINDOW if clf["class"] != "none" else 0.0


def _remaining(lot, wafer_id):
    ids = [w["wafer_id"] for w in T._LOTS.get(lot, [])]
    if wafer_id in ids:
        return len(ids) - ids.index(wafer_id) - 1
    if lot in _lot_state:
        return max(LOT_SIZE - len(_lot_state[lot]), 0)
    return LOT_SIZE - 1


# ---------- analiză ----------

def _classify(wafer_id, m, from_history):
    if from_history:
        return T.tool_clasifica(wafer_id)               # același cache ca agentul
    p, u = P.predict(m), U.is_unknown(m)
    return {"class": p["clasa"], "confidence": p["incredere"], "unknown": u["necunoscut"],
            "nearest_class": u["cel_mai_apropiat"], "history_id": None}


def _resolve(wafer_id, wafer):
    """-> (hartă 64x64, history_id sau None)."""
    if wafer is not None:
        return _to64(wafer), None
    hid = T._resolve(wafer_id)                          # ValueError dacă nu există
    if not 0 <= hid < len(T._MAPS):
        raise ValueError(f"history_id în afara istoricului: {hid}")
    return np.asarray(T._MAPS[hid]), hid


def analyze(wafer_id, wafer=None, lot_id=None, use_llm=True, include_gradcam=True,
            cost_params=None, drift=None, remaining=None, lang="ro"):
    m, hid = _resolve(wafer_id, wafer)
    from_history = hid is not None
    lot = lot_id or (T.lot_of(wafer_id) if from_history else None)

    with _lock:
        clf = _classify(wafer_id, m, from_history)
        if drift is None:
            if lot in T._LOTS and wafer_id in {w["wafer_id"] for w in T._LOTS[lot]}:
                drift = T.tool_verifica_drift(lot, wafer_id)
            elif lot in _lot_state and wafer_id in _lot_state[lot]:
                drift = _window(lot, wafer_id)      # wafer dintr-un lot deja rulat: fereastra de la poziția lui
            elif lot:
                drift = _track(lot, wafer_id, clf["class"])
            else:
                drift = _drift_from([])
        sim = [] if clf["unknown"] else _similar(m, 5, hid)
        cam = _gradcam_png(m) if include_gradcam else None

    # De aici, aceeași logică de decizie ca în llm/agent.analyze()
    w = wafer_id
    ai = False   # True doar dacă textul vine efectiv de la LLM (nu din răspunsul determinist)
    manual, target, acord, incert = [], None, 1.0, False
    if not clf["unknown"]:
        acord = (sum(1 for s in sim if s["clasa"] == clf["class"]) / len(sim)) if sim else 0.0
        incert = (not drift["alarm"]) and clf["confidence"] < A.PRAG_INCERT and acord < A.ACORD_MIN
        if not incert:
            target = drift["repeated_class"] if drift["alarm"] else clf["class"]
            if target != "none":
                manual = A._relevante(T.tool_cauta_manual(target), target)

    sim_facts = [{k: v for k, v in s.items() if not k.startswith("_")} for s in sim]
    facts = {"wafer_id": w, "lot_id": lot,
             "clasificare": {"class": clf["class"], "confidence": clf["confidence"]},
             "acord_vecini": round(acord, 2), "drift": drift, "similare": sim_facts,
             "manual": [{"doc": x["doc"], "page": x["page"], "text": x["text"][:400]} for x in manual]}

    if clf["unknown"]:
        text, surse = A._local(clf, drift, sim_facts, w, lot, lang), []
    elif incert:
        text, surse = A._incert(clf, sim_facts, w, lang), []
    else:
        try:
            if not use_llm:
                raise RuntimeError("use_llm=False")
            text = A._llm(facts, lang)
            ai = True
        except Exception as e:
            log.info("fără LLM pentru %s: %s", w, e)
            text = A._local(clf, drift, sim_facts, w, lot, lang)
        surse = A._surse(manual, target)
    if not lot:   # wafer trimis fără lot (ex. din UI): textul determinist al agentului ar scrie "din lotul None"
        text = {k: v.replace(" din lotul None", "") for k, v in text.items()}

    cls ="necunoscut" if clf["unknown"] else clf["class"]
    rem = remaining if remaining is not None else _remaining(lot, w)
    return {
        "wafer_id": w, "lot_id": lot,
        "class_name": cls, "is_defect": cls != "none", "confidence": float(clf["confidence"]),
        "is_unknown": bool(clf["unknown"]), "uncertain": incert,
        "explanation": text["explicatie"], "action": text["actiune"], "sources": surse,
        "similar_images": [{"wafer_id": s["wafer_id"], "class_name": s["clasa"],
                            "probability": round(min(max(s["scor"], 0.0), 1.0) * 100, 1),
                            "wafer": (s["_map"] if "_map" in s else T._MAPS[s["history_id"]]).tolist()}
                           for s in sim],
        "drift": {k: drift.get(k) for k in ("alarm", "repeated_class", "affected", "window")},
        "cost": cost(_p_defect(clf, drift), rem, cost_params),
        "wafer": m.tolist(),
        "gradcam_png": cam,
        "ai_generated": ai,
    }


def analyze_lot(lot_id, wafers=None, use_llm=True, include_maps=True, cost_params=None, lang="ro"):
    if wafers is None:
        if lot_id not in T._LOTS:
            raise ValueError(f"lot necunoscut: {lot_id}; trimite și lista 'wafers'")
        wafers = [{"wafer_id": w["wafer_id"], "wafer": None} for w in T._LOTS[lot_id]]

    items, classes, maps = [], [], []
    first = None
    with _lock:
        for k, wf in enumerate(wafers):
            m, hid = _resolve(wf["wafer_id"], wf.get("wafer"))
            clf = _classify(wf["wafer_id"], m, hid is not None)
            classes.append(clf["class"])
            d = _drift_from(classes[-DRIFT_WINDOW:])
            if d["alarm"] and first is None:
                first = (k, d)
            maps.append(m)
            items.append({"index": k + 1, "wafer_id": wf["wafer_id"],
                          "class_name": "necunoscut" if clf["unknown"] else clf["class"],
                          "confidence": float(clf["confidence"]), "is_unknown": bool(clf["unknown"]),
                          "alarm": d["alarm"], "wafer": m.tolist() if include_maps else None})

    # Lotul trimis ca matrici devine istoricul de drift pentru următoarele /analyze pe același lot
    if lot_id not in T._LOTS:
        _lot_state[lot_id] = OrderedDict((it["wafer_id"], c) for it, c in zip(items, classes))

    out = {"lot_id": lot_id, "n_wafers": len(items), "alarm": first is not None,
           "first_alarm_index": first[0] + 1 if first else None,
           "repeated_class": first[1]["repeated_class"] if first else None,
           "wafers": items, "cost": None, "alarm_analysis": None}
    if first:
        k, d = first
        wf = wafers[k]
        res = analyze(wf["wafer_id"], wf.get("wafer"), lot_id, use_llm=use_llm, include_gradcam=True,
                      cost_params=cost_params, drift=d, remaining=len(items) - k - 1, lang=lang)
        out["alarm_analysis"], out["cost"] = res, res["cost"]
    return out
