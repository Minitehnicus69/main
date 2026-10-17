# Module 3: LLM Agent & RAG (Semiconductor Diagnostics WM-811K)

This module implements the autonomous decision and diagnostics agent for silicon wafer inspection, utilizing RAG (Retrieval-Augmented Generation), visual classification via trained weights from `weights.pt`, and reasoning through tool use (Tool / Function Calling).

## Module Structure

```text
llm/
├── causes.json            # Knowledge base (9 WM-811K classes: defects, physical causes, verification checks)
├── classifier.py          # CNN Model (WaferCNN) loading weights from weights.pt
├── rag.py                 # FAISS vector indexing and semantic search across technical manuals
├── tools.py               # 4 specialized tools for manufacturing process diagnostics
├── agent.py               # ReAct agent and main entry point: analyze(wafer_json)
├── demo_scenarios.json    # 6 preconfigured test scenarios for demonstration
├── test_agent.py          # Automated scenario validation script
├── requirements.txt       # Python dependencies for the module
├── .env.example           # Template for configuring API keys and paths
└── .env                   # Local configuration (ignored by git)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the `.env` file:
```env
LLM_API_KEY=rc-096ecf812a39bafd5551d273d4f7cedeb4c90eeea326187e
LLM_MODEL=Qwen3.8-27B
LLM_BASE_URL=https://api.openai.com/v1
WEIGHTS_PATH=E:/download/weights.pt
```

## Integration with Other Modules

- **Backend (#4):**
  Imports the main function:
  ```python
  from llm.agent import analyze
  result = analyze({"wafer_id": "W-101", "lot_id": "L-01"})
  ```
  The response is a dictionary containing the required keys:
  ```json
  {
    "explanation": "Defect description and lot status",
    "action": "Recommended technical action",
    "sources": ["manual.pdf (p. X)", "causes.json"]
  }
  ```

- **Visual Model (#1):**
  Trained weights from `weights.pt` are integrated via `llm/classifier.py`. For direct matrix inference:
  ```python
  from llm.classifier import predict_wafer
  result = predict_wafer(wafer_matrix)
  ```

## Running Tests

```bash
python test_agent.py
```
All 6 scenarios are automatically validated and verified.
