# Modulul 3: LLM Agent & RAG (Diagnostic Semiconductor WM-811K)

Acest modul implementează agentul autonom de diagnosticare și decizie pentru inspecția wafer-elor de siliciu, folosind RAG (Retrieval-Augmented Generation), clasificare vizuală prin ponderile antrenate din `weights.pt`, și raționament prin apelare de unelte (Tool / Function Calling).

## Structura Modulului

```text
llm/
├── causes.json            # Baza de cunoștințe (9 clase WM-811K: defecte, cauze fizice, verificări)
├── classifier.py          # Modelul CNN (WaferCNN) ce încarcă ponderile din weights.pt
├── rag.py                 # Indexare vectorială FAISS și căutare semantică în manualele tehnice
├── tools.py               # Cele 4 unelte specifice procesului de fabricație
├── agent.py               # Agentul ReAct și punctul de intrare: analyze(wafer_json)
├── demo_scenarios.json    # 6 scenarii de test prestabilite pentru demonstrație
├── test_agent.py          # Script de validare automată a scenariilor
├── requirements.txt       # Dependențele Python ale modulului
├── .env.example           # Șablon pentru configurarea cheilor de API și căilor
└── .env                   # Configurația locală (ignorat de git)
```

## Configurare

1. Instalați dependențele:
```bash
pip install -r requirements.txt
```

2. Configurați fișierul `.env`:
```env
LLM_API_KEY=rc-096ecf812a39bafd5551d273d4f7cedeb4c90eeea326187e
LLM_MODEL=Qwen3.8-27B
LLM_BASE_URL=https://api.openai.com/v1
WEIGHTS_PATH=E:/download/weights.pt
```

## Integrare cu celelalte module

- **Backend (#4):**
  Importă funcția principală:
  ```python
  from llm.agent import analyze
  rezultat = analyze({"wafer_id": "W-101", "lot_id": "L-01"})
  ```
  Răspunsul este un dicționar cu cheile obligatorii:
  ```json
  {
    "explicatie": "Descrierea defectului și starea lotului",
    "actiune": "Măsura tehnică recomandată",
    "surse": ["manual.pdf (p. X)", "causes.json"]
  }
  ```

- **Model Vizual (#1):**
  Ponderile antrenate din `weights.pt` sunt integrate prin `llm/classifier.py`. Pentru inferență directă pe matrici:
  ```python
  from llm.classifier import predict_wafer
  rezultat = predict_wafer(wafer_matrix)
  ```

## Rulare Teste

```bash
python test_agent.py
```
Toate cele 6 scenarii sunt validate automat cu succes.
