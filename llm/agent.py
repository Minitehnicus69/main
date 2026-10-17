import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm import tools as T

load_dotenv(HERE / ".env")
log = logging.getLogger("agent")

API_KEY = os.environ.get("LLM_API_KEY", "rc-096ecf812a39bafd5551d273d4f7cedeb4c90eeea326187e")
MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3.8-Flash-Next")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "25"))
# Flash-Next acceptă "none" (fără gândire); Qwen3.8-27B acceptă doar low/medium/xhigh.
EFFORT = os.environ.get("LLM_EFFORT") or ("none" if "Flash" in MODEL_NAME else "low")
MAX_TOKENS = 600 if EFFORT == "none" else 2000   # gândirea consumă din max_tokens
PRAG_INCERT = float(os.environ.get("PRAG_INCERT", "0.60"))   # încredere sub care verificăm vecinii
ACORD_MIN = float(os.environ.get("ACORD_MIN", "0.60"))       # cât din vecini trebuie să confirme clasa

PROMPT = """Ești un asistent pentru ingineri de test și proces într-o fabrică de semiconductori.
Analizezi wafer map-uri (clase: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none).

Cauze probabile și verificări pe clase (sunt ipoteze, nu certitudini):
__CAUZE__

Reguli:
1. Folosește doar faptele primite și tabelul de mai sus. Nu inventa cauze, cifre sau documente.
2. Nu menționa piese, componente sau proceduri care nu apar în tabelul de cauze sau în fragmentele din manual.
3. Formulează cauzele ca "probabil" sau "posibil".
4. Dacă drift.alarm e true, spune că tot lotul trebuie verificat, nu doar wafer-ul curent. Alarma înseamnă că aceeași clasă apare pe cel puțin 3 din ultimele 5 wafere (nu neapărat consecutive). Nu scrie "consecutive".
5. "acord_vecini" e partea din cele mai asemănătoare wafere din istoric care au aceeași clasă. Dacă e sub 0.6, spune că istoricul nu confirmă clasa.
6. Răspunde STRICT cu un obiect JSON, fără markdown: {"explicatie": "...", "actiune": "..."}
Scrie scurt, în română, pe înțelesul unui tehnician de tură."""

# lang="en" (UI-ul în engleză): alt tabel de cauze și răspuns în engleză. "ro" rămâne implicit, cu același cache.
try:
    CAUSES_EN = json.loads((HERE / "causes_en.json").read_text(encoding="utf-8"))
except Exception:
    CAUSES_EN = {}


def _causes(lang):
    return (CAUSES_EN or T.CAUSES) if lang == "en" else T.CAUSES


def _prompt(lang):
    if lang != "en":
        return PROMPT
    return PROMPT.replace("Scrie scurt, în română, pe înțelesul unui tehnician de tură.",
                          'Write briefly, in English, so a shift technician can understand it. '
                          'Describe the facts in plain language and never quote internal field names '
                          '(such as acord_vecini or drift.alarm). '
                          'Keep the JSON keys "explicatie" and "actiune" but write their values in English.')

# Cache pe disc: fiecare răspuns Qwen se cere o singură dată (cheia include modelul și prompt-ul).
CACHE_FILE = HERE / "llm_cache.json"
try:
    _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
except Exception:
    _cache = {}


def _save_cache():
    try:
        CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("nu am putut salva cache-ul: %s", e)


def _key(facts, prompt=PROMPT):
    raw = MODEL_NAME + "|" + EFFORT + "|" + prompt + "|" + json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _relevante(manual, target):
    """Păstrează doar paragrafele care menționează clasa (potrivire pe cuvânt întreg)."""
    if not target:
        return []
    rx = re.compile(r"(?<![\w-])" + re.escape(target) + r"(?![\w-])", re.IGNORECASE)
    return [m for m in manual if rx.search(m.get("text", ""))]


def _surse(manual, target):
    out = []
    for m in manual[:1]:
        out.append(f"{m['doc']} (p. {m['page']})" if m.get("page") else str(m["doc"]))
    if target in T.CAUSES and "causes.json" not in out:
        out.append("causes.json")
    return out


def _incert(clf, sim, w, lang="ro"):
    """Clasificare nesigură (încredere mică, vecinii nu confirmă): fără acțiune pe echipament, fără LLM."""
    c, conf = clf["class"], clf["confidence"]
    same = sum(1 for s in sim if s["clasa"] == c)
    rest = Counter(s["clasa"] for s in sim if s["clasa"] != c)
    if lang == "en":
        others = ", ".join(f"{k} ({v})" for k, v in rest.most_common(3)) or "none"
        return {"explicatie": (f"The classification of wafer {w} is uncertain: the CNN predicts '{c}' at {conf * 100:.0f}%, "
                               f"but only {same} of the {len(sim)} most similar historical wafers share that class "
                               f"(others: {others}). This is not enough to justify an equipment intervention."),
                "actiune": "Inspect the wafer visually before any intervention. Do not stop or adjust the tool based on this classification."}
    altele = ", ".join(f"{k} ({v})" for k, v in rest.most_common(3)) or "niciuna"
    return {"explicatie": (f"Clasificarea wafer-ului {w} e nesigură: CNN-ul zice '{c}' cu {conf * 100:.0f}%, iar dintre "
                           f"cele {len(sim)} wafere cele mai asemănătoare din istoric doar {same} au aceeași clasă "
                           f"(celelalte: {altele}). Nu e suficient pentru a recomanda o intervenție pe echipament."),
            "actiune": "Verifică wafer-ul vizual înainte de orice intervenție. Nu opri și nu modifica echipamentul pe baza acestei clasificări."}


def _local(clf, drift, sim, w, lot, lang="ro"):
    """Răspuns determinist: pentru necunoscut, pentru wafere banale și ca rezervă dacă LLM-ul nu răspunde."""
    c, conf = clf["class"], clf["confidence"]
    pct = f"{conf * 100:.0f}%"
    if lang == "en":
        return _local_en(clf, drift, sim, w, lot, c, pct)
    if clf["unknown"]:
        return {"explicatie": (f"Wafer-ul {w} nu seamănă suficient cu niciun pattern cunoscut "
                               f"(cel mai apropiat: {clf['nearest_class']}). CNN-ul zice '{c}' cu {pct}, "
                               f"dar pentru acest caz încrederea nu contează."),
                "actiune": "Pattern necunoscut: escaladează la inginerul de proces. Nu presupune o cauză."}
    if drift["alarm"]:
        rc = drift["repeated_class"]
        check = T.CAUSES.get(rc, {}).get("check", "inspecție conform procedurii")
        return {"explicatie": (f"Alarmă de drift în lotul {lot}: '{rc}' apare pe cel puțin 3 din ultimele 5 wafere "
                               f"(wafer curent: {w}, '{c}', {pct}). Indică o problemă de proces, nu un wafer izolat."),
                "actiune": f"Oprește lotul și verifică echipamentul. Verificare sugerată pentru '{rc}': {check}"}
    if c == "none":
        return {"explicatie": f"Wafer-ul {w} din lotul {lot} arată nominal (clasa 'none', {pct}). Fără alarmă de drift.",
                "actiune": "Poate continua spre stația următoare."}
    check = T.CAUSES.get(c, {}).get("check", "inspecție conform procedurii standard")
    same = sum(1 for s in sim if s["clasa"] == c)
    acord = f" Dintre cele {len(sim)} wafere cele mai asemănătoare din istoric, {same} au aceeași clasă." if sim else ""
    return {"explicatie": f"Wafer-ul {w} din lotul {lot} are un defect izolat '{c}' ({pct}). Fără alarmă de drift.{acord}",
            "actiune": f"Verificare sugerată pentru '{c}': {check}"}


def _local_en(clf, drift, sim, w, lot, c, pct):
    causes = _causes("en")
    in_lot = f" in lot {lot}" if lot else ""
    if clf["unknown"]:
        return {"explicatie": (f"Wafer {w} does not sufficiently match any known pattern "
                               f"(closest: {clf['nearest_class']}). The CNN predicts '{c}' at {pct}, "
                               f"but confidence is not meaningful in this case."),
                "actiune": "Unknown pattern: escalate to the process engineer. Do not assume a root cause."}
    if drift["alarm"]:
        rc = drift["repeated_class"]
        check = causes.get(rc, {}).get("check", "inspection according to procedure")
        return {"explicatie": (f"Drift alarm{in_lot}: '{rc}' appears on at least 3 of the last 5 wafers "
                               f"(current wafer: {w}, '{c}', {pct}). This points to a process issue, not an isolated wafer."),
                "actiune": f"Hold the lot and inspect the equipment. Suggested check for '{rc}': {check}"}
    if c == "none":
        return {"explicatie": f"Wafer {w}{in_lot} looks nominal (class 'none', {pct}). No drift alarm.",
                "actiune": "Can proceed to the next station."}
    check = causes.get(c, {}).get("check", "inspection according to standard procedure")
    same = sum(1 for s in sim if s["clasa"] == c)
    agree = f" Of the {len(sim)} most similar historical wafers, {same} share the same class." if sim else ""
    return {"explicatie": f"Wafer {w}{in_lot} shows an isolated '{c}' defect ({pct}). No drift alarm.{agree}",
            "actiune": f"Suggested check for '{c}': {check}"}


def _parse(text):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None
    ok = isinstance(d, dict) and all(isinstance(d.get(k), str) and d[k].strip() for k in ("explicatie", "actiune"))
    return {"explicatie": d["explicatie"], "actiune": d["actiune"]} if ok else None


def _llm(facts, lang="ro"):
    prompt = _prompt(lang)
    k = _key(facts, prompt)
    if k in _cache:
        log.info("răspuns din cache")
        return _cache[k]
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT, max_retries=0)
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt.replace("__CAUZE__", json.dumps(_causes(lang), ensure_ascii=False))},
            {"role": "user", "content": "Fapte verificate de unelte:\n" + json.dumps(facts, ensure_ascii=False, default=str)
                                        + "\nScrie răspunsul final în JSON."},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        # Flash-Next ignoră uneori reasoning_effort="none" și gândește până la max_tokens (răspuns gol, ~45 s);
        # enable_thinking=False oprește gândirea sigur (~7 s).
        extra_body={"reasoning_effort": EFFORT,
                    **({"chat_template_kwargs": {"enable_thinking": False}} if EFFORT == "none" else {})},
    )
    res = _parse(r.choices[0].message.content or "")
    if res is None:
        raise ValueError(f"răspuns LLM gol sau invalid (finish_reason={r.choices[0].finish_reason})")
    _cache[k] = res
    _save_cache()
    return res


def analyze(wafer_json: dict) -> dict:
    """analyze({"wafer_id": "B_w10"}) -> contractul JSON complet al proiectului.
    Opțional: "lot_id", "use_llm": False (sare peste LLM, pentru wafere banale și ca să respecte limita de cereri)."""
    w = wafer_json["wafer_id"]
    lot = wafer_json.get("lot_id") or wafer_json.get("lot") or T.lot_of(w)

    clf = T.tool_clasifica(w)
    drift = T.tool_verifica_drift(lot, w) if lot else {"alarm": False, "repeated_class": None, "affected": 0, "window": []}
    sim, manual, target = [], [], None
    acord, incert = 1.0, False
    if not clf["unknown"]:
        sim = T.tool_gaseste_similare(w)
        acord = (sum(1 for s in sim if s["clasa"] == clf["class"]) / len(sim)) if sim else 0.0
        incert = (not drift["alarm"]) and clf["confidence"] < PRAG_INCERT and acord < ACORD_MIN
        if not incert:
            target = drift["repeated_class"] if drift["alarm"] else clf["class"]
            if target != "none":
                manual = _relevante(T.tool_cauta_manual(target), target)

    facts = {"wafer_id": w, "lot_id": lot,
             "clasificare": {"class": clf["class"], "confidence": clf["confidence"]},
             "acord_vecini": round(acord, 2), "drift": drift, "similare": sim,
             "manual": [{"doc": m["doc"], "page": m["page"], "text": m["text"][:400]} for m in manual]}

    if clf["unknown"]:                       # regulă dură: fără LLM, ca să nu inventeze o cauză
        text, surse = _local(clf, drift, sim, w, lot), []
    elif incert:                             # clasificare nesigură: fără acțiune pe echipament
        text, surse = _incert(clf, sim, w), []
    else:
        try:
            if not wafer_json.get("use_llm", True):
                raise RuntimeError("use_llm=False")
            text = _llm(facts)
            log.info("răspuns generat de LLM (%s)", MODEL_NAME)
        except Exception as e:
            log.warning("fără LLM, folosesc răspunsul determinist: %s", e)
            text = _local(clf, drift, sim, w, lot)
        surse = _surse(manual, target)       # sursele vin din unelte, nu din LLM

    return {"wafer_id": w,
            "clasa": "necunoscut" if clf["unknown"] else clf["class"],
            "incredere": clf["confidence"],
            "necunoscut": clf["unknown"],
            "nesigur": incert,
            "similare": [{"wafer_id": s["wafer_id"], "clasa": s["clasa"], "scor": s["scor"]} for s in sim],
            "drift": {"alarma": drift["alarm"], "wafere_afectate": drift["affected"]},
            "explicatie": text["explicatie"], "actiune": text["actiune"], "surse": surse}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(analyze({"wafer_id": "B_w10"}), indent=2, ensure_ascii=False))