import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("agent").setLevel(logging.INFO)
logging.getLogger("tools").setLevel(logging.INFO)

from llm import tools as T
from llm.agent import analyze, PRAG_INCERT, ACORD_MIN

T._init()
asteptat = {"A": None, "B": 10, "C": 15, "D": None}
for lot, wafere in T._LOTS.items():
    prima = next((i for i, w in enumerate(wafere, 1)
                  if T.tool_verifica_drift(lot, w["wafer_id"])["alarm"]), None)
    print(f"lot {lot}: alarmă la wafer {prima} | așteptat {asteptat[lot]} | {'OK' if prima == asteptat[lot] else 'GREȘIT'}")

# Cât de bun e pragul de încredere pe cele 100 de wafere din loturi
sus, jos, nesigure = [0, 0], [0, 0], []
for wid, info in T._WAFERS.items():
    clf = T.tool_clasifica(wid)
    sim = T.tool_gaseste_similare(wid)
    acord = sum(1 for s in sim if s["clasa"] == clf["class"]) / len(sim) if sim else 0.0
    bun = clf["class"] == info["clasa_reala"]
    grup = sus if clf["confidence"] >= PRAG_INCERT else jos
    grup[0] += 1
    grup[1] += bun
    if not clf["unknown"] and clf["confidence"] < PRAG_INCERT and acord < ACORD_MIN:
        nesigure.append((wid, bun))
print(f"\nîncredere >= {PRAG_INCERT}: {sus[0]} wafere, {sus[1]} corecte")
print(f"încredere <  {PRAG_INCERT}: {jos[0]} wafere, {jos[1]} corecte")
print(f"marcate 'nesigur' (încredere mică + vecinii nu confirmă): {len(nesigure)}, "
      f"din care clasificate de fapt corect: {sum(b for _, b in nesigure)}")

for wid in ["B_w10", "C_w15", "A_w05", "D_w11"]:
    print("\n==", wid)
    print(json.dumps(analyze({"wafer_id": wid}), indent=2, ensure_ascii=False))