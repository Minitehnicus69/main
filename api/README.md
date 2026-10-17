# Backend WaferSight (`api/`)

FastAPI care leagă modelul vizual (`model/`), detectorul de necunoscut, agentul LLM (`llm/`) și UI-ul (`ui/`).
UI-ul vorbește **doar** cu acest API.

## Pornire

Din rădăcina proiectului (unde sunt `weights.pt` și `history.npz`):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -r api/requirements.txt
.\.venv\Scripts\uvicorn api.main:app --reload --port 8000
```

Interfața (din `ui/`, servită de același server): http://localhost:8000
Documentația interactivă: http://localhost:8000/docs

Fără `weights.pt` / `history.npz`, `POST /analyze` întoarce `api/mock_response.json` (vezi `GET /health` → `"mode": "mock"`),
ca UI-ul să aibă oricând ce apela.

## Endpoint-uri

| Metodă | Cale | Ce face |
|---|---|---|
| GET | `/health` | `mode` (`model`/`mock`), mărimea istoricului, loturile disponibile |
| GET | `/lots` | Loturile sintetice din `data/lots.json` (A, B, C, D) cu ID-urile wafer-elor |
| POST | `/analyze` | Un wafer: clasă, încredere, necunoscut, similare, drift, explicație, acțiune, surse, cost, Grad-CAM |
| POST | `/lot` | Un lot întreg: clasa pe fiecare wafer, unde pornește alarma, analiza agentului pe primul wafer cu alarmă |
| POST | `/feedback` | Salvează „corect/greșit” (SQLite, `api/feedback.db`) |
| GET | `/feedback` | Ultimele feedback-uri |
| POST | `/cost` | Recalculează costul pentru slidere, fără să ruleze modelul |
| POST | `/upload` | Fișiere încărcate (JSON, CSV/TXT, NPY/NPZ, PNG/JPG) → hărți 0/1/2 validate; analiza se face apoi cu `/analyze` sau `/lot` |
| GET | `/upload/sample?fmt=json\|csv` | Fișier de exemplu: câte un wafer real din istoric pentru fiecare clasă |

**Upload:** `curl -F files=@wafers.json -F files=@map.png localhost:8000/upload`. Erorile sunt per fișier (`errors`),
nu opresc restul. Fișierele pickle sunt refuzate (deserializarea lor poate rula cod).

## Exemple curl

> În PowerShell folosește `curl.exe` (nu aliasul `curl`). Exemplele de mai jos sunt pentru bash / Git Bash.

**Wafer din loturi (după ID):**
```bash
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"wafer_id": "B_w10"}'
```

**Wafer din istoric după index, fără LLM și fără Grad-CAM (rapid):**
```bash
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"wafer_id": "123", "use_llm": false, "include_gradcam": false}'
```

**Wafer trimis ca matrice (așa trimite UI-ul).** Orice dimensiune; e redimensionat la 64×64. Cu `lot_id`, API-ul ține
ultimele 5 predicții ale lotului pentru drift:
```bash
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"wafer_id": "WF-001", "lot_id": "LOT-X", "wafer": [[0,1,1,0],[1,2,2,1],[1,2,2,1],[0,1,1,0]]}'
```

**Ipoteze de cost personalizate:**
```bash
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"wafer_id": "C_w15", "cost": {"pret_wafer": 2000, "ore_oprire": 1, "cost_ora": 8000}}'
```

**Lot întreg din `data/lots.json`:**
```bash
curl -X POST localhost:8000/lot -H "Content-Type: application/json" \
  -d '{"lot_id": "B", "include_maps": false}'
```

**Lot propriu (wafere în ordinea testării):**
```bash
curl -X POST localhost:8000/lot -H "Content-Type: application/json" \
  -d '{"lot_id": "LOT-Y", "wafers": [{"wafer_id": "1"}, {"wafer_id": "Y2", "wafer": [[0,1],[1,2]]}]}'
```

**Feedback:**
```bash
# predicția a fost corectă
curl -X POST localhost:8000/feedback -H "Content-Type: application/json" \
  -d '{"wafer_id": "B_w10", "predicted_class": "Edge-Ring", "correct": true}'

# greșită: clasa reală e Loc (pentru wafere care nu sunt în istoric trimite și "wafer")
curl -X POST localhost:8000/feedback -H "Content-Type: application/json" \
  -d '{"wafer_id": "D_w11", "predicted_class": "Edge-Loc", "correct": false, "true_class": "Loc"}'
```

**Cost pentru slidere:**
```bash
curl -X POST localhost:8000/cost -H "Content-Type: application/json" \
  -d '{"p_defect": 0.8, "wafere_ramase": 15, "params": {"pret_wafer": 1500, "ore_oprire": 2, "cost_ora": 5000}}'
```

## Răspunsul `/analyze` (contractul)

Cheile pe care le citește `ui/app.py` (`class_name`, `confidence`, `is_defect`, `explanation`, `action`, `sources`,
`similar_images[].wafer/probability`) plus:

```jsonc
{
  "wafer_id": "B_w10", "lot_id": "B",
  "class_name": "Edge-Ring",        // sau "necunoscut"
  "is_defect": true,
  "confidence": 0.97,               // 0..1, softmax CNN
  "is_unknown": false,              // detectorul de necunoscut (pattern în afara claselor)
  "uncertain": false,               // încredere mică ȘI vecinii din istoric nu confirmă clasa
  "explanation": "...", "action": "...", "sources": ["causes.json"],
  "similar_images": [{"wafer_id": "history_812", "class_name": "Edge-Ring", "probability": 96.4, "wafer": [[...]]}],
  "drift": {"alarm": true, "repeated_class": "Edge-Ring", "affected": 3, "window": ["none", "...", "Edge-Ring"]},
  "cost": {"cost_rebut": 13500.0, "cost_oprire": 10000.0, "p_defect": 0.6, "wafere_ramase": 15, "recomandare": "opreste"},
  "wafer": [[...64x64...]],         // harta analizată, pentru imshow
  "gradcam_png": "iVBORw0..."       // PNG base64: st.image(base64.b64decode(r["gradcam_png"]))
}
```

**Limba textului:** `/analyze` și `/lot` acceptă `"lang": "en"` (implicit `"ro"`). Explicația și acțiunea vin atunci
în engleză (prompt LLM + `llm/causes_en.json`). Răspunsul are și `ai_generated`: `true` dacă textul vine de la LLM,
`false` dacă e răspunsul determinist (LLM oprit, timeout, wafer necunoscut sau nesigur). UI-ul cere mereu `en`.

## Reguli

- **Drift:** în ultimele 5 wafere ale lotului, fără `none`, aceeași clasă de cel puțin 3 ori → alarmă.
  Pentru loturile din `lots.json` fereastra se ia după poziția wafer-ului în lot; pentru wafere trimise ca matrice,
  API-ul ține evidența per `lot_id` (în memorie).
- **Similare:** similaritate cosinus între embedding-ul CNN și `history.npz` (`probability` = scor × 100), plus
  wafere confirmate prin `/feedback`. Dacă un tehnician a corectat clasa unui wafer din istoric, la similare apare clasa corectată.
- **Cost:** `cost_rebut = wafere_ramase × pret_wafer × p_defect`, `cost_oprire = ore_oprire × cost_ora`;
  `recomandare = "opreste"` dacă rebutul costă mai mult.
  `p_defect` = câte din ultimele 5 wafere ale lotului au defectul dominant / 5 (un defect izolat → 0.2, alarmă → ≥ 0.6,
  lot curat → 0); fără lot: încrederea CNN / 5 (sau 0 pentru `none`). Un singur wafer nu oprește linia de unul singur.
  `wafere_ramase` implicit = câte wafere mai urmează în lot (lot de 25).
- **Text:** explicația și acțiunea vin din agentul `llm/agent.py` (LLM cu cache; dacă LLM-ul nu răspunde sau
  `use_llm: false`, răspuns determinist). Pentru wafere necunoscute sau nesigure nu se apelează LLM-ul.
