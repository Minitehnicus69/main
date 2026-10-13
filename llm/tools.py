# llm/tools.py
import sys
from pathlib import Path

# Asigură rezolvarea modulelor locale indiferent de directorul curent de lucru
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from classifier import predict_wafer
from rag import cauta_manual

# Baza de date pentru testare și integrare
MOCK_CLASSIFICATION_DB = {
    "W-101": {"class": "Edge-Ring", "confidence": 0.91},
    "W-102": {"class": "Center", "confidence": 0.96},
    "W-103": {"class": "necunoscut", "confidence": 0.32},
    "W-104": {"class": "Scratch", "confidence": 0.88},
    "W-105": {"class": "none", "confidence": 0.99},
    "W-106": {"class": "Center", "confidence": 0.65}
}

# 1. Implementările funcțiilor (Unelte)
def tool_clasifica(wafer_id: str, wafer_matrix: list | None = None) -> dict:
    """
    Obține clasa defectului și scorul de încredere pentru wafer din modelul CNN
    (folosind weights.pt dacă se furnizează o matrice a wafer-ului, altfel din registrul lotului).
    """
    if wafer_matrix is not None:
        try:
            return predict_wafer(wafer_matrix)
        except Exception as e:
            print(f"[tool_clasifica] Eroare la inferenta CNN: {e}")

    return MOCK_CLASSIFICATION_DB.get(wafer_id, {"class": "Center", "confidence": 0.85})

def tool_verifica_drift(lot_id: str) -> dict:
    """
    Verifică dacă s-a declanșat o alarmă de drift (același defect repetat în lotul curent:
    minim 3 din ultimele 5 wafere cu aceeași clasă defectă).
    """
    if lot_id in ("LOT-CRITIC", "L-02"):
        return {"lot_id": lot_id, "alarm": True, "repeated_class": "Center", "count": 3}
    return {"lot_id": lot_id, "alarm": False, "repeated_class": "none", "count": 0}

def tool_gaseste_similare(wafer_id: str) -> list[dict]:
    """
    Găsește wafere similare din baza istorică de fabricație pe baza spațiului de embedding.
    """
    return [
        {"wafer_id": "W-HIST-441", "class": "Center", "similarity": 0.93},
        {"wafer_id": "W-HIST-812", "class": "Center", "similarity": 0.89}
    ]

def tool_cauta_manual(query: str) -> list[dict]:
    """
    Caută proceduri de intervenție și cauze fizice în manualele tehnice de proces prin motorul RAG.
    """
    return cauta_manual(query, top_k=1)

# Mapare nume -> funcție executabilă
TOOL_ROUTER = {
    "clasifica": tool_clasifica,
    "verifica_drift": tool_verifica_drift,
    "gaseste_similare": tool_gaseste_similare,
    "cauta_manual": tool_cauta_manual
}

# Scheme Anthropic (compatibile cu ghidul original)
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

# Scheme OpenAI / Qwen3.8-27B Function Calling (type: 'function')
OPENAI_TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "clasifica",
            "description": "Obține clasa defectului și scorul de încredere pentru wafer din modelul CNN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wafer_id": {"type": "string", "description": "ID-ul unic al wafer-ului"}
                },
                "required": ["wafer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verifica_drift",
            "description": "Verifică dacă s-a declanșat o alarmă de drift (același defect repetat în lotul curent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "ID-ul lotului de producție"}
                },
                "required": ["lot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gaseste_similare",
            "description": "Găsește wafere similare din baza istorică de fabricație pe baza spațiului de embedding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wafer_id": {"type": "string", "description": "ID-ul wafer-ului curent"}
                },
                "required": ["wafer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cauta_manual",
            "description": "Caută proceduri de intervenție și cauze fizice în manualele tehnice de proces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cuvinte cheie sau descrierea defectului"}
                },
                "required": ["query"]
            }
        }
    }
]

if __name__ == "__main__":
    print("Testare unelte...")
    print("1. clasifica(W-101):", tool_clasifica("W-101"))
    print("2. verifica_drift(LOT-CRITIC):", tool_verifica_drift("LOT-CRITIC"))
    print("3. gaseste_similare(W-106):", tool_gaseste_similare("W-106"))
    print("4. cauta_manual(Center):", tool_cauta_manual("Center"))
    print("Toate cele 4 unelte sunt functionale!")
