# WaferSight UI

A minimal Streamlit interface for wafer defect inspection. It includes local
synthetic samples and runs without the model, dataset, or API.

## Run locally

From this folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Choose a sample wafer and select **Analizează wafer**. When a defect is found,
the UI displays up to five matching reference-wafer maps and their probabilities.
Without an API configured, results and reference images are mock data.

## Connect the API

Choose **API backend** in the sidebar and enter the backend URL. You can also set
`WAFER_API_URL` before starting Streamlit to prefill the address:

```powershell
$env:WAFER_API_URL = "http://localhost:8000"
python -m streamlit run app.py
```

The UI calls `POST /analyze` with
`{"wafer_id": "...", "wafer": [[...], ...]}`. Expected response:

```json
{
  "wafer_id": "WF-0248-001",
  "class_name": "Edge-Ring",
  "is_defect": true,
  "confidence": 0.973,
  "explanation": "Defectele sunt concentrate spre marginea wafer-ului.",
  "action": "Verifică procesul la marginea wafer-ului.",
  "sources": ["Comparație cu setul de referință"],
  "similar_images": [
    {
      "wafer_id": "REF-EDGE-001",
      "probability": 98.1,
      "wafer": [[0, 1, 1], [1, 2, 1], [0, 1, 0]]
    }
  ]
}
```

`confidence` is in the 0–1 range. Each `similar_images[].probability` is a
percentage from 0 to 100; its `wafer` value is a 2D map using 0 (background),
1 (good die), and 2 (defective die). The UI sorts matches by probability and
shows the top five. `explanation`, `action`, and `sources` are optional;
`is_defect` may be omitted when `class_name` is one of the recognized normal
labels (`normal`, `none`, `no defect`, or `fără defect detectat`).
