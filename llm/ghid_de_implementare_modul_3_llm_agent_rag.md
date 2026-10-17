# Step-by-Step Implementation Plan: Module 3 (LLM, Agent, RAG)

This document details the technical steps for building the decision and reasoning module in the `llm/` directory, using an external API-driven model (Claude / OpenAI / Gemini / Qwen), FAISS vector search for technical documentation, and Tool Calling / Function Calling.

---

## 1. File Structure

```text
llm/
├── causes.json            # Knowledge base: Defect -> Physical Causes -> Verification Check
├── rag.py                 # PDF processing, FAISS indexing, and semantic search
├── tools.py               # Definition of the 4 tools (schemas + mock/integrations)
├── agent.py               # ReAct loop / Tool Use via external API (analyze())
├── demo_scenarios.json    # 6 preconfigured test scenarios for demo
├── test_agent.py          # Automated scenario validation script
└── requirements.txt       # Module dependencies
```

---

## 2. Dependencies (`requirements.txt`)

Install the required packages:

```text
anthropic>=0.25.0
pypdf>=4.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0
```

Installation command:
```bash
pip install -r requirements.txt
```

---

## 3. Step 1: Structured Knowledge Base (`causes.json`)

Create `llm/causes.json` mapping WM-811K defect patterns to manufacturing root causes and recommended actions:

```json
{
  "Center": {
    "causes": [
      "Non-uniform layer deposition (CVD/PVD)",
      "Asymmetric pressure during chemical-mechanical polishing (CMP)",
      "Thermal variations across the center of the heating platen"
    ],
    "check": "Check the central gas nozzles in the process chamber and the planarity of the CMP head."
  },
  "Edge-Ring": {
    "causes": [
      "Focus ring erosion in plasma etching",
      "Deficient helium cooling on the edge of the electrostatic chuck (ESC)"
    ],
    "check": "Inspect the focus ring in the etching chamber and the edge helium pressure."
  },
  "Edge-Loc": {
    "causes": [
      "Aggressive mechanical contact on the wafer edge during transfer",
      "Localized thermal stress at the clamping pins"
    ],
    "check": "Check the aligner pins and the gripping force of the mechanical handling arm."
  },
  "Scratch": {
    "causes": [
      "Scratching caused by the robotic handling arm",
      "Residual abrasive particles on vacuum suction cups"
    ],
    "check": "Inspect the robotic arm trajectory and clean vacuum suction contact pads."
  },
  "Random": {
    "causes": [
      "Suspended particle contamination in the cleanroom",
      "HEPA/ULPA filter failure above the workstation"
    ],
    "check": "Measure particle concentration in the cleanroom enclosure and verify laminar airflow."
  },
  "none": {
    "causes": [
      "Nominal operating parameters, no deviations detected"
    ],
    "check": "Authorize batch transfer to the next technological processing step."
  }
}
```

---

## 4. Step 2: RAG Engine with Precise Citation (`rag.py`)

This file loads troubleshooting manuals in PDF format, extracts pages, indexes them in FAISS, and provides a search function that returns text along with the exact document name and page number.

```python
# llm/rag.py
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Compact and fast embedding model on CPU
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents_store = []
index = None

def build_index(pdf_paths: list[str], chunk_size_words: int = 350):
    """
    Reads PDFs, extracts page chunks, and builds the FAISS index.
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
        # Default mock index if no external PDFs are provided
        documents_store = [
            {
                "doc": "manual_cvd_standard.pdf",
                "page": 14,
                "text": "Central defects (Center) indicate CVD gas flow non-uniformities or asymmetric temperature of the heating platen."
            },
            {
                "doc": "manual_etch_troubleshooting.pdf",
                "page": 29,
                "text": "The Edge-Ring pattern is frequently caused by physical degradation and erosion of the focus ring."
            }
        ]
        chunks = [d["text"] for d in documents_store]

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

def search_manual(query: str, top_k: int = 1) -> list[dict]:
    """
    Returns the most relevant chunks along with citation metadata.
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

# Backwards compatibility alias
cauta_manual = search_manual
```

---

## 5. Step 3: Defining the 4 Tools (`tools.py`)

This file contains JSON Schema definitions for Tool Calling and function implementations (functional mocks that connect to modules #1, #2, and #4).

```python
# llm/tools.py
from rag import search_manual

# 1. Tool implementations (functional mocks)
def tool_classify(wafer_id: str) -> dict:
    # Connects directly to Team #1 CNN model
    mock_db = {
        "W-101": {"class": "Edge-Ring", "confidence": 0.91},
        "W-102": {"class": "Center", "confidence": 0.96},
        "W-103": {"class": "unknown", "confidence": 0.32},
        "W-104": {"class": "Scratch", "confidence": 0.88},
        "W-105": {"class": "none", "confidence": 0.99},
        "W-106": {"class": "Center", "confidence": 0.65}
    }
    return mock_db.get(wafer_id, {"class": "Center", "confidence": 0.85})

def tool_check_drift(lot_id: str) -> dict:
    # Checks if 3 of the last 5 wafers in the lot have the same defect class
    if lot_id == "LOT-CRITIC" or lot_id == "L-02":
        return {"lot_id": lot_id, "alarm": True, "repeated_class": "Center", "count": 3}
    return {"lot_id": lot_id, "alarm": False, "repeated_class": "none", "count": 0}

def tool_find_similar(wafer_id: str) -> list[dict]:
    # Queries the embedding vectors from embeddings.npy
    return [
        {"wafer_id": "W-HIST-441", "class": "Center", "similarity": 0.93},
        {"wafer_id": "W-HIST-812", "class": "Center", "similarity": 0.89}
    ]

def tool_search_manual(query: str) -> list[dict]:
    return search_manual(query, top_k=1)

# Name -> executable function routing
TOOL_ROUTER = {
    "classify": tool_classify,
    "check_drift": tool_check_drift,
    "find_similar": tool_find_similar,
    "search_manual": tool_search_manual
}

# Schemas passed to LLM (Anthropic API compatible)
TOOLS_DEFINITIONS = [
    {
        "name": "classify",
        "description": "Obtain defect class and confidence score for a wafer from the CNN model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wafer_id": {"type": "string", "description": "Unique identifier of the wafer"}
            },
            "required": ["wafer_id"]
        }
    },
    {
        "name": "check_drift",
        "description": "Check if a drift alarm has been triggered (repeated defect in current lot).",
        "input_schema": {
            "type": "object",
            "properties": {
                "lot_id": {"type": "string", "description": "Production lot ID"}
            },
            "required": ["lot_id"]
        }
    },
    {
        "name": "find_similar",
        "description": "Find similar wafers in historical database based on embedding space.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wafer_id": {"type": "string", "description": "Current wafer ID"}
            },
            "required": ["wafer_id"]
        }
    },
    {
        "name": "search_manual",
        "description": "Search intervention procedures and physical causes in process technical manuals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or defect description"}
            },
            "required": ["query"]
        }
    }
]
```

---

## 6. Step 4: Agent Loop (`agent.py`)

This file exports the `analyze(wafer_json)` function consumed by the Backend (`api/`).

```python
# llm/agent.py
import json
import os
from anthropic import Anthropic
from tools import TOOL_ROUTER, TOOLS_DEFINITIONS

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "MOCK_KEY"))

SYSTEM_PROMPT = """You are an expert agent in silicon wafer manufacturing process diagnostics.
Your objective:
1. Analyze wafer and lot status by calling the required tools (classify, check_drift, etc.).
2. If confidence is low, verify similar wafers. If a defect is found, search the manual for root cause and corrective action.
3. The FINAL response must STRICTLY be a valid JSON object without surrounding text:
{
  "explanation": "Defect description, confidence level, and lot drift status.",
  "action": "Recommended technical action for operator/engineer.",
  "sources": ["document_name.pdf (p. X)", "causes.json"]
}"""

def analyze(wafer_json: dict) -> dict:
    """
    Entry point required by integration contract: analyze(wafer_json) -> dict
    """
    wafer_id = wafer_json.get("wafer_id", "W-101")
    lot_id = wafer_json.get("lot_id", "L-01")

    messages = [
        {"role": "user", "content": f"Analyze wafer {wafer_id} from lot {lot_id}."}
    ]

    # ReAct loop limited to 5 iterations
    for _ in range(5):
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS_DEFINITIONS,
            messages=messages
        )

        # Case 1: LLM requests tool execution
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    fn_name = block.name
                    args = block.input
                    
                    fn = TOOL_ROUTER.get(fn_name)
                    res = fn(**args) if fn else {"error": f"Unknown tool {fn_name}"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(res)
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Case 2: LLM completed reasoning and provides final response
        for block in response.content:
            if hasattr(block, "text"):
                text_content = block.text.strip()
                if text_content.startswith("```"):
                    text_content = text_content.split("```")[1]
                    if text_content.startswith("json"):
                        text_content = text_content[4:]
                try:
                    res_json = json.loads(text_content.strip())
                    return res_json
                except json.JSONDecodeError:
                    return {
                        "explanation": text_content,
                        "action": "Manual inspection recommended.",
                        "sources": ["causes.json"]
                    }

    return {"error": "Maximum reasoning step limit reached."}
```

---

## 7. Step 5: The 6 Demo Scenarios (`demo_scenarios.json`)

Scenarios demonstrating agent adaptability across production contexts:

```json
[
  {
    "id": 1,
    "title": "Isolated Edge-Ring Defect",
    "input": {"wafer_id": "W-101", "lot_id": "L-01"},
    "expected_type": "Isolated Edge-Ring, verify focus ring at end of shift"
  },
  {
    "id": 2,
    "title": "Critical Drift Alarm (Repeated Center)",
    "input": {"wafer_id": "W-102", "lot_id": "LOT-CRITIC"},
    "expected_type": "Active drift alarm; immediate shutdown of CVD chamber"
  },
  {
    "id": 3,
    "title": "Unknown Pattern (Out-of-Distribution)",
    "input": {"wafer_id": "W-103", "lot_id": "L-03"},
    "expected_type": "Unknown defect; flag for labeling by process engineer"
  },
  {
    "id": 4,
    "title": "Mechanical Scratch",
    "input": {"wafer_id": "W-104", "lot_id": "L-04"},
    "expected_type": "Scratch defect; inspect robotic arm and transfer path"
  },
  {
    "id": 5,
    "title": "Nominal Clean Wafer (None)",
    "input": {"wafer_id": "W-105", "lot_id": "L-05"},
    "expected_type": "No defect; authorize transfer to next step"
  },
  {
    "id": 6,
    "title": "Low Confidence with Similarity Validation",
    "input": {"wafer_id": "W-106", "lot_id": "L-06"},
    "expected_type": "Call find_similar to confirm ambiguous class"
  }
]
```

---

## 8. Step 6: Testing and Automated Validation (`test_agent.py`)

Run this script to ensure stability:

```python
# llm/test_agent.py
import json
import os
from agent import analyze

def run_tests():
    with open("demo_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"Starting tests for {len(scenarios)} scenarios...\n")
    required_keys = {"explanation", "action", "sources"}

    for item in scenarios:
        print(f"Testing Scenario #{item['id']}: {item['title']}...")
        result = analyze(item["input"])
        
        # Output JSON schema validation
        if not isinstance(result, dict) or not required_keys.issubset(result.keys()):
            print(f"❌ FAILURE: Response missing required keys: {result}")
            return
        
        print(f"  ✓ Explanation: {result['explanation'][:80]}...")
        print(f"  ✓ Action:      {result['action'][:80]}...")
        print(f"  ✓ Sources:     {result['sources']}")
        print("  Status: OK\n")

    print("All scenarios passed successfully!")

if __name__ == "__main__":
    run_tests()
```

---

## 9. Integration Checklist with Other Teams

- [ ] **With Backend (#4):** Backend imports `from llm.agent import analyze`. The `POST /analyze` endpoint passes the payload to the function and returns the JSON result directly to the frontend.
- [ ] **With Visual Model (#1):** In `llm/tools.py`, the body of `tool_classify` connects directly to the real model: `model.predict(wafer_matrix)`.
- [ ] **With Data & Evaluation (#2):** In `llm/tools.py`, calls `is_unknown(wafer_matrix)`. If true, `tool_classify` returns the class `"unknown"`.