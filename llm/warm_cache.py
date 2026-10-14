import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("agent").setLevel(logging.INFO)

from llm import tools as T
from llm.agent import analyze

T._init()
n = 0
for lot, wafere in T._LOTS.items():
    for w in wafere:
        clf = T.tool_clasifica(w["wafer_id"])
        alarm = T.tool_verifica_drift(lot, w["wafer_id"])["alarm"]
        if clf["unknown"] or (clf["class"] == "none" and not alarm):
            continue                      # banalele și necunoscutele nu folosesc LLM
        t0 = time.time()
        r = analyze({"wafer_id": w["wafer_id"]})
        print(f"{w['wafer_id']:7} {r['clasa']:10} alarmă={r['drift']['alarma']!s:5} nesigur={r['nesigur']!s:5} {time.time() - t0:5.1f}s")
        if not r["nesigur"]:
            n += 1
            time.sleep(3)                 # cheia e comună pe echipă: 30 cereri/minut
print("gata,", n, "răspunsuri LLM generate")