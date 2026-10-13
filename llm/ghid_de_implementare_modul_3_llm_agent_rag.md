# Plan de Implementare Pas cu Pas: Modulul 3 (LLM, Agent, RAG)

Acest document detaliază pașii tehnici pentru construirea modulului de decizie și raționament din directorul `llm/`, folosind un model extern apelat prin API (Claude / OpenAI / Gemini), căutare vectorială FAISS pentru documentația tehnică și apelare de unelte (Tool Calling / Function Calling).

---

## 1. Structura Fișierelor

```text
llm/
├── causes.json            # Bază de cunoștințe: Defect -> Cauze fizice -> Verificare
├── rag.py                 # Procesare PDF, indexare FAISS și căutare semantică
├── tools.py               # Definirea celor 4 unelte (scheme + mock/integrări)
├── agent.py               # Bucla ReAct / Tool Use prin API extern (analyze())
├── demo_scenarios.json    # 6 scenarii de test prestabilite pentru demo
├── test_agent.py          # Script de validare automată a scenariilor
└── requirements.txt       # Dependențele modulului
```

---

## 2. Dependențe (`requirements.txt`)

Instalați pachetele necesare:

```text
anthropic>=0.25.0
pypdf>=4.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0
```

Comandă de instalare:
```bash
pip install -r requirements.txt
```

---

## 3. Pasul 1: Baza de Cunoștințe Structurată (`causes.json`)

Creați fișierul `llm/causes.json` care corelează tiparele specifice setului de date WM-811K cu cauzele de fabricație și acțiunile recomandate:

```json
{
  "Center": {
    "causes": [
      "Depunere neuniformă de strat (CVD/PVD)",
      "Presiune asimetrică la lustruirea mecano-chimică (CMP)",
      "Variații termice pe centrul platanului de încălzire"
    ],
    "check": "Verifică duzele centrale de gaz din camera de proces și planeitatea capului CMP."
  },
  "Edge-Ring": {
    "causes": [
      "Eroziune a inelului de focalizare (focus ring) în gravarea cu plasmă",
      "Răcire deficitară cu heliu pe marginea electrostatic chuck-ului (ESC)"
    ],
    "check": "Inspectează inelul de focalizare din camera de gravare și presiunea de heliu pe margine."
  },
  "Edge-Loc": {
    "causes": [
      "Contact mecanic agresiv pe marginea wafer-ului în timpul transferului",
      "Stres termic localizat la clemele de susținere"
    ],
    "check": "Verifică pinii de aliniere (aligner pins) și forța brațului de prindere mecanică."
  },
  "Scratch": {
    "causes": [
      "Zgâriere provocată de brațul robotic de manipulare",
      "Particule abrazive reziduale pe ventuzele de vid"
    ],
    "check": "Inspectează traseul brațului robotizat și curăță pernele de contact cu vid."
  },
  "Random": {
    "causes": [
      "Contaminare cu particule suspendate în camera curată (cleanroom)",
      "Defectare filtru HEPA/ULPA deasupra stației de lucru"
    ],
    "check": "Măsoară concentrația de particule în incintă și verifică fluxul laminar de aer."
  },
  "none": {
    "causes": [
      "Parametri nominali de operare, fără abateri detectate"
    ],
    "check": "Autorizează transferul lotului către următoarea stație tehnologică."
  }
}
```

---

## 4. Pasul 2: Motorul RAG cu Citare Precisă (`rag.py`)

Acest fișier încarcă manuale de depanare în format PDF, extrage paginile, le indexează în FAISS și oferă o funcție de căutare ce returnează textul alături de document și pagina exactă.

```python
# llm/rag.py
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Model de embedding compact și rapid pe CPU
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents_store = []
index = None

def build_index(pdf_paths: list[str], chunk_size_words: int = 350):
    """
    Citește PDF-urile, extrage fragmente pe pagini și construiește indexul FAISS.
    """
    global index, documents_store
    documents_store = []
    chunks = []

    for path in pdf_paths:
        if not os.path.exists(path):
            continue
        reader = PdfReader(path)
        doc_name = os.path.basename(path)
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words = text.split()
            for i in range(0, len(words), chunk_size_words):
                chunk = " ".join(words[i:i + chunk_size_words])
                if len(chunk.strip()) > 30:
                    chunks.append(chunk)
                    documents_store.append({
                        "doc": doc_name,
                        "page": page_idx,
                        "text": chunk
                    })

    if not chunks:
        # Index mock dacă nu sunt PDF-uri adăugate încă
        documents_store = [
            {
                "doc": "manual_cvd_standard.pdf",
                "page": 14,
                "text": "Defectele centrale (Center) indică neuniformități de debit gaz CVD sau temperatură asimetrică a platanului."
            },
            {
                "doc": "manual_etch_troubleshooting.pdf",
                "page": 29,
                "text": "Pattern-ul Edge-Ring este cauzat în mod frecvent de degradarea fizică a inelului de focalizare (focus ring)."
            }
        ]
        chunks = [d["text"] for d in documents_store]

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

def cauta_manual(query: str, top_k: int = 1) -> list[dict]:
    """
    Returnează fragmentele cele mai relevante împreună cu metadatele de citare.
    """
    global index
    if index is None:
        build_index([])

    q_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    distances, indices = index.search(q_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(documents_store):
            results.append(documents_store[idx])
    return results
```

---

## 5. Pasul 3: Definirea Celor 4 Unelte (`tools.py`)

Fișierul conține definițiile JSON Schema pentru Tool Calling și implementările funcțiilor (mock-uri inițiale, ușor de conectat la modulele #1, #2 și #4).

```python
# llm/tools.py
from rag import cauta_manual

# 1. Implementările funcțiilor (Mock-uri funcționale)
def tool_clasifica(wafer_id: str) -> dict:
    # Va apela direct modelul CNN al echipei #1
    mock_db = {
        "W-101": {"class": "Edge-Ring", "confidence": 0.91},
        "W-102": {"class": "Center", "confidence": 0.96},
        "W-103": {"class": "necunoscut", "confidence": 0.32},
        "W-104": {"class": "Scratch", "confidence": 0.88},
        "W-105": {"class": "none", "confidence": 0.99},
        "W-106": {"class": "Center", "confidence": 0.65}
    }
    return mock_db.get(wafer_id, {"class": "Center", "confidence": 0.85})

def tool_verifica_drift(lot_id: str) -> dict:
    # Verifică dacă 3 din ultimele 5 wafere din lot au aceeași clasă defectă
    if lot_id == "LOT-CRITIC" or lot_id == "L-02":
        return {"lot_id": lot_id, "alarm": True, "repeated_class": "Center", "count": 3}
    return {"lot_id": lot_id, "alarm": False, "repeated_class": "none", "count": 0}

def tool_gaseste_similare(wafer_id: str) -> list[dict]:
    # Va apela căutarea pe vectorii de embeddings.npy
    return [
        {"wafer_id": "W-HIST-441", "class": "Center", "similarity": 0.93},
        {"wafer_id": "W-HIST-812", "class": "Center", "similarity": 0.89}
    ]

def tool_cauta_manual(query: str) -> list[dict]:
    return cauta_manual(query, top_k=1)

# Mapare nume -> funcție executabilă
TOOL_ROUTER = {
    "clasifica": tool_clasifica,
    "verifica_drift": tool_verifica_drift,
    "gaseste_similare": tool_gaseste_similare,
    "cauta_manual": tool_cauta_manual
}

# Scheme transmise către LLM (compatibile Anthropic API)
TOOLS_DEFINITIONS = [
    {
        "name": "clasifica",
        "description": "Obține clasa defectului și scorul de încredere pentru wafer din modelul CNN.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wafer_id": {"type": "string", "description": "ID-ul unic al wafer-ului"}
            },
            "required": ["wafer_id"]
        }
    },
    {
        "name": "verifica_drift",
        "description": "Verifică dacă s-a declanșat o alarmă de drift (același defect repetat în lotul curent).",
        "input_schema": {
            "type": "object",
            "properties": {
                "lot_id": {"type": "string", "description": "ID-ul lotului de producție"}
            },
            "required": ["lot_id"]
        }
    },
    {
        "name": "gaseste_similare",
        "description": "Găsește wafere similare din baza istorică de fabricație pe baza spațiului de embedding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wafer_id": {"type": "string", "description": "ID-ul wafer-ului curent"}
            },
            "required": ["wafer_id"]
        }
    },
    {
        "name": "cauta_manual",
        "description": "Caută proceduri de intervenție și cauze fizice în manualele tehnice de proces.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Cuvinte cheie sau descrierea defectului"}
            },
            "required": ["query"]
        }
    }
]
```

---

## 6. Pasul 4: Bucla Agentului (`agent.py`)

Acest fișier exportă funcția `analyze(wafer_json)` care este consumată direct de modulul de Backend (`api/`).

```python
# llm/agent.py
import json
import os
from anthropic import Anthropic
from tools import TOOL_ROUTER, TOOLS_DEFINITIONS

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "MOCK_KEY"))

SYSTEM_PROMPT = """Ești un agent expert în diagnosticarea procesului de fabricație a wafer-elor de siliciu.
Obiectivul tău:
1. Analizează starea wafer-ului și a lotului apelând uneltele necesare (clasifica, verifica_drift, etc.).
2. Dacă încrederea este scăzută, verifică wafere similare. Dacă apare un defect, caută în manual cauza și acțiunea corectivă.
3. Răspunsul FINAL trebuie să fie STRICT un obiect JSON valid, fără text suplimentar în afara lui:
{
  "explicatie": "Descrierea defectului, nivelul de încredere și starea de drift din lot.",
  "actiune": "Măsura tehnică recomandată pentru operator/inginer.",
  "surse": ["nume_document.pdf (p. X)", "causes.json"]
}"""

def analyze(wafer_json: dict) -> dict:
    """
    Punctul de intrare cerut de contractul de integrare: analyze(wafer_json) -> dict
    """
    wafer_id = wafer_json.get("wafer_id", "W-101")
    lot_id = wafer_json.get("lot_id", "L-01")

    messages = [
        {"role": "user", "content": f"Analizează wafer-ul {wafer_id} din lotul {lot_id}."}
    ]

    # Buclă ReAct limitată la 5 iterații
    for _ in range(5):
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS_DEFINITIONS,
            messages=messages
        )

        # Cazul 1: LLM-ul solicită apelul uneia sau mai multor unelte
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    fn_name = block.name
                    args = block.input
                    
                    fn = TOOL_ROUTER.get(fn_name)
                    res = fn(**args) if fn else {"error": f"Tool {fn_name} necunoscut"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(res)
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Cazul 2: LLM-ul a terminat raționamentul și oferă răspunsul final
        for block in response.content:
            if hasattr(block, "text"):
                text_content = block.text.strip()
                # Curățare markdown code block dacă există
                if text_content.startswith("```"):
                    text_content = text_content.split("```")[1]
                    if text_content.startswith("json"):
                        text_content = text_content[4:]
                try:
                    return json.loads(text_content.strip())
                except json.JSONDecodeError:
                    return {
                        "explicatie": text_content,
                        "actiune": "Inspecție manuală recomandată.",
                        "surse": ["causes.json"]
                    }

    return {"error": "Depășire număr maxim de pași de raționament."}
```

---

## 7. Pasul 5: Cele 6 Scenarii de Demo (`demo_scenarios.json`)

Scenariile demonstrează adaptabilitatea agentului în diferite contexte de producție:

```json
[
  {
    "id": 1,
    "title": "Defect izolat Edge-Ring",
    "input": {"wafer_id": "W-101", "lot_id": "L-01"},
    "expected_type": "Edge-Ring izolat, verificare focus ring la final de tură"
  },
  {
    "id": 2,
    "title": "Alarmă critică de drift (Center repetat)",
    "input": {"wafer_id": "W-102", "lot_id": "LOT-CRITIC"},
    "expected_type": "Alarmă drift activă; oprire imediată a camerei CVD"
  },
  {
    "id": 3,
    "title": "Pattern necunoscut (Out-of-Distribution)",
    "input": {"wafer_id": "W-103", "lot_id": "L-03"},
    "expected_type": "Defect necunoscut; marcare pentru etichetare de către inginer"
  },
  {
    "id": 4,
    "title": "Zgârietură mecanică (Scratch)",
    "input": {"wafer_id": "W-104", "lot_id": "L-04"},
    "expected_type": "Defect Scratch; inspectare braț robotizat și traseu de transfer"
  },
  {
    "id": 5,
    "title": "Wafer nominal curat (None)",
    "input": {"wafer_id": "W-105", "lot_id": "L-05"},
    "expected_type": "Fără defect; autorizare trecere la următorul pas"
  },
  {
    "id": 6,
    "title": "Încredere scăzută cu validare prin similaritate",
    "input": {"wafer_id": "W-106", "lot_id": "L-06"},
    "expected_type": "Apelare gaseste_similare pentru confirmarea clasei ambigue"
  }
]
```

---

## 8. Pasul 6: Testare și Validare Automată (`test_agent.py`)

Rulați acest script pentru a asigura stabilitatea livrabilelor:

```python
# llm/test_agent.py
import json
import os
from agent import analyze

def run_tests():
    with open("demo_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"Începere teste pentru {len(scenarios)} scenarii...\n")
    required_keys = {"explicatie", "actiune", "surse"}

    for item in scenarios:
        print(f"Testare Scenariu #{item['id']}: {item['title']}...")
        result = analyze(item["input"])
        
        # Validare schemă JSON de ieșire
        if not isinstance(result, dict) or not required_keys.issubset(result.keys()):
            print(f"❌ EȘEC: Răspunsul nu conține cheile obligatorii: {result}")
            return
        
        print(f"  ✓ Explicație: {result['explicatie'][:80]}...")
        print(f"  ✓ Acțiune:    {result['actiune'][:80]}...")
        print(f"  ✓ Surse:      {result['surse']}")
        print("  Status: OK\n")

    print("Toate scenariile au trecut cu succes!")

if __name__ == "__main__":
    run_tests()
```

---

## 9. Checklist de Integrare cu Restul Echipelor

- [ ] **Cu Backend (#4):** Colegul de la backend importă doar `from llm.agent import analyze`. Endpoint-ul `POST /analyze` pasează payload-ul către funcție și returnează rezultatul JSON direct către interfață.
- [ ] **Cu Modelul Vizual (#1):** În `llm/tools.py`, corpul funcției `tool_clasifica` se înlocuiește cu apelul real către model: `model.predict(wafer_matrix)`.
- [ ] **Cu Date & Evaluare (#2):** În `llm/tools.py`, se apelează `is_unknown(wafer_matrix)`. Dacă este adevărat, `tool_clasifica` returnează clasa `"necunoscut"`.