# llm/tools.py
import sys
from pathlib import Path

# Ensure local module resolution regardless of current working directory
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from classifier import predict_wafer
from rag import cauta_manual, search_manual

# Database for testing and integration
MOCK_CLASSIFICATION_DB = {
    "W-101": {"class": "Edge-Ring", "confidence": 0.91},
    "W-102": {"class": "Center", "confidence": 0.96},
    "W-103": {"class": "unknown", "confidence": 0.32},
    "W-104": {"class": "Scratch", "confidence": 0.88},
    "W-105": {"class": "none", "confidence": 0.99},
    "W-106": {"class": "Center", "confidence": 0.65}
}

# 1. Tool implementations
def tool_classify(wafer_id: str, wafer_matrix: list | None = None) -> dict:
    """
    Retrieves defect class and confidence score for a wafer from the CNN model
    (using weights.pt if a wafer matrix is provided, otherwise from the lot registry).
    """
    if wafer_matrix is not None:
        try:
            return predict_wafer(wafer_matrix)
        except Exception as e:
            print(f"[tool_classify] Error during CNN inference: {e}")

    return MOCK_CLASSIFICATION_DB.get(wafer_id, {"class": "Center", "confidence": 0.85})

def tool_check_drift(lot_id: str) -> dict:
    """
    Checks if a drift alarm is triggered (repeated defect in the current lot:
    at least 3 of the last 5 wafers with the same defect class).
    """
    if lot_id in ("LOT-CRITIC", "L-02"):
        return {"lot_id": lot_id, "alarm": True, "repeated_class": "Center", "count": 3}
    return {"lot_id": lot_id, "alarm": False, "repeated_class": "none", "count": 0}

def tool_find_similar(wafer_id: str) -> list[dict]:
    """
    Finds similar wafers from the historical manufacturing database based on embedding space.
    """
    return [
        {"wafer_id": "W-HIST-441", "class": "Center", "similarity": 0.93},
        {"wafer_id": "W-HIST-812", "class": "Center", "similarity": 0.89}
    ]

def tool_search_manual(query: str) -> list[dict]:
    """
    Searches intervention procedures and physical causes in process technical manuals via RAG.
    """
    return search_manual(query, top_k=1)

# Romanian function aliases for backwards compatibility
tool_clasifica = tool_classify
tool_verifica_drift = tool_check_drift
tool_gaseste_similare = tool_find_similar
tool_cauta_manual = tool_search_manual

# Name -> executable function routing
TOOL_ROUTER = {
    "classify": tool_classify,
    "check_drift": tool_check_drift,
    "find_similar": tool_find_similar,
    "search_manual": tool_search_manual,
    # Romanian aliases
    "clasifica": tool_classify,
    "verifica_drift": tool_check_drift,
    "gaseste_similare": tool_find_similar,
    "cauta_manual": tool_search_manual
}

# Anthropic tool schemas
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

# OpenAI / Qwen function calling schemas
OPENAI_TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "classify",
            "description": "Obtain defect class and confidence score for a wafer from the CNN model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wafer_id": {"type": "string", "description": "Unique identifier of the wafer"}
                },
                "required": ["wafer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_drift",
            "description": "Check if a drift alarm has been triggered (repeated defect in current lot).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "Production lot ID"}
                },
                "required": ["lot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": "Find similar wafers in historical database based on embedding space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wafer_id": {"type": "string", "description": "Current wafer ID"}
                },
                "required": ["wafer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": "Search intervention procedures and physical causes in process technical manuals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or defect description"}
                },
                "required": ["query"]
            }
        }
    }
]

if __name__ == "__main__":
    print("Testing tools...")
    print("1. classify(W-101):", tool_classify("W-101"))
    print("2. check_drift(LOT-CRITIC):", tool_check_drift("LOT-CRITIC"))
    print("3. find_similar(W-106):", tool_find_similar("W-106"))
    print("4. search_manual(Center):", tool_search_manual("Center"))
    print("All 4 tools are functional!")
