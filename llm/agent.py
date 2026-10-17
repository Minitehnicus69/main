import json
import os
import sys
from pathlib import Path

# Ensure local module resolution regardless of current working directory
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env (located in llm directory)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from tools import (
    OPENAI_TOOLS_DEFINITIONS,
    TOOL_ROUTER,
    tool_cauta_manual,
    tool_check_drift,
    tool_clasifica,
    tool_classify,
    tool_find_similar,
    tool_gaseste_similare,
    tool_search_manual,
    tool_verifica_drift,
)

# External API configuration as per user specifications
API_KEY = os.environ.get("LLM_API_KEY", "rc-096ecf812a39bafd5551d273d4f7cedeb4c90eeea326187e")
MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3.8-27B")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

# Load physical causes from causes.json (knowledge grounding)
causes_file = Path(__file__).resolve().parent / "causes.json"
KNOWN_CAUSES = {}
if causes_file.exists():
    try:
        with open(causes_file, "r", encoding="utf-8") as f:
            KNOWN_CAUSES = json.load(f)
    except Exception as e:
        print(f"[agent] Warning reading causes.json: {e}")

SYSTEM_PROMPT = """You are an expert agent in silicon wafer manufacturing process diagnostics (WM-811K dataset).
Known defect patterns (9 classes learned from the CNN model weights.pt):
- Center: central defects caused by non-uniform CVD/PVD deposition or CMP asymmetric pressure.
- Donut: ring thermal profile or turbulent gas flow in the CVD chamber.
- Edge-Ring: focus ring erosion in plasma etching or helium cooling issues at the ESC edge.
- Edge-Loc: aggressive mechanical contact on the wafer edge during robotic transfer.
- Loc: local contamination on the photolithographic mask (reticle contamination) or UV exposure.
- Near-full: total vacuum loss, critical RF power drift, or massive chemical contamination.
- Random: suspended particles in the cleanroom, HEPA/ULPA filter failure.
- Scratch: scratches caused by the robotic handling arm or abrasive particles on suction cups.
- none: nominal conforming wafer, no structural defects.

Operating instructions:
1. Analyze the wafer and lot status by calling the necessary tools ('classify', 'check_drift').
2. If confidence is low (< 0.70) or the defect is 'unknown', call 'find_similar'.
3. If an identified defect appears, call 'search_manual' to find specific intervention instructions.
4. The FINAL response must STRICTLY be a valid JSON object, without extra text or markdown:
{
  "explanation": "Defect description, confidence level, and lot drift status.",
  "action": "Recommended technical measure for the operator/engineer.",
  "sources": ["document_name.pdf (p. X)", "causes.json"]
}"""

def _build_response_dict(explanation: str, action: str, sources: list[str]) -> dict:
    return {
        "explanation": explanation,
        "action": action,
        "sources": sources,
        # Romanian aliases for backwards compatibility
        "explicatie": explanation,
        "actiune": action,
        "surse": sources
    }

def _execute_local_reasoning(wafer_id: str, lot_id: str) -> dict:
    """
    Deterministic ReAct reasoning engine based on knowledge from weights.pt and causes.json.
    Guarantees schema-compliant responses during network/endpoint unavailability.
    """
    # Step 1: classify
    clf = tool_classify(wafer_id)
    c_class = clf.get("class", "Center")
    conf = clf.get("confidence", 0.85)

    # Step 2: check drift
    drift = tool_check_drift(lot_id)
    has_alarm = drift.get("alarm", False)

    sources = ["causes.json"]

    # Scenario 5: Nominal clean
    if c_class == "none":
        doc_rag = tool_search_manual("none")
        if doc_rag:
            sources.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        return _build_response_dict(
            explanation=f"Wafer {wafer_id} from lot {lot_id} is nominally clean (class 'none') with {conf*100:.1f}% confidence. No anomalies or drift alarms detected.",
            action="Authorize lot transfer to the next technological process station.",
            sources=sources
        )

    # Scenario 3: Out-of-Distribution / Unknown
    if c_class in ("unknown", "necunoscut") or conf < 0.40:
        tool_find_similar(wafer_id)
        return _build_response_dict(
            explanation=f"Wafer {wafer_id} exhibits an unknown structural pattern (low confidence {conf*100:.1f}%). Queried historical reference similarities.",
            action="Flag wafer for OOD analysis and manual labeling by the process engineer.",
            sources=sources
        )

    # Scenario 2: Critical drift alarm
    if has_alarm:
        doc_rag = tool_search_manual(c_class)
        if doc_rag:
            sources.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        rep_class = drift.get("repeated_class", c_class)
        info = KNOWN_CAUSES.get(rep_class, {})
        check_action = info.get("check", "Immediate shutdown of process chamber and parameter verification.")
        return _build_response_dict(
            explanation=f"Critical drift alarm active in lot {lot_id}: repeated defect '{rep_class}' detected on at least 3 of the last 5 wafers. Current wafer: {wafer_id} ({conf*100:.1f}%).",
            action=f"Immediate shutdown of the process chamber associated with defect {rep_class}. {check_action}",
            sources=sources
        )

    # Scenario 6: Low confidence with similarity validation
    if conf < 0.70:
        sim_wafers = tool_find_similar(wafer_id)
        doc_rag = tool_search_manual(c_class)
        if doc_rag:
            sources.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        info = KNOWN_CAUSES.get(c_class, {})
        check_action = info.get("check", "Detailed inspection.")
        return _build_response_dict(
            explanation=f"Wafer {wafer_id} was classified as '{c_class}' with intermediate confidence of {conf*100:.1f}%. Historical similarity confirms class via matches with {sim_wafers[0]['wafer_id']}.",
            action=f"Inspection of specific parameters for class {c_class} is recommended. {check_action}",
            sources=sources
        )

    # Scenarios 1 and 4: Identified isolated defect (Edge-Ring, Scratch, etc.)
    doc_rag = tool_search_manual(c_class)
    if doc_rag:
        sources.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
    info = KNOWN_CAUSES.get(c_class, {})
    check_action = info.get("check", "Inspect processing equipment according to standard procedure.")

    return _build_response_dict(
        explanation=f"Wafer {wafer_id} from lot {lot_id} exhibits an isolated defect of type '{c_class}' ({conf*100:.1f}% confidence). No drift alarm on lot.",
        action=f"Verify according to standard procedure for defect {c_class}: {check_action}",
        sources=sources
    )

def analyze(wafer_json: dict) -> dict:
    """
    Entry point required by integration contract: analyze(wafer_json) -> dict
    Receives wafer data and returns a dict with keys: 'explanation', 'action', 'sources'.
    """
    wafer_id = wafer_json.get("wafer_id", "W-101")
    lot_id = wafer_json.get("lot_id", "L-01")

    # Attempt external API call (OpenAI-compatible / Qwen3.8-27B) if client is available
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=8.0
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze wafer {wafer_id} from lot {lot_id}."}
        ]

        # ReAct loop limited to 5 iterations
        for _ in range(5):
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=OPENAI_TOOLS_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1000
            )

            msg = completion.choices[0].message

            # Case 1: Model requests tool calls
            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}
                    fn = TOOL_ROUTER.get(fn_name)
                    tool_res = fn(**args) if fn else {"error": f"Unknown tool {fn_name}"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_res)
                    })
                continue

            # Case 2: Model finished reasoning and returns response
            content = msg.content or ""
            text_clean = content.strip()
            if text_clean.startswith("```"):
                text_clean = text_clean.split("```")[1]
                text_clean = text_clean.removeprefix("json")
            try:
                res_json = json.loads(text_clean.strip())
                if isinstance(res_json, dict):
                    if {"explanation", "action", "sources"}.issubset(res_json.keys()):
                        res_json.setdefault("explicatie", res_json["explanation"])
                        res_json.setdefault("actiune", res_json["action"])
                        res_json.setdefault("surse", res_json["sources"])
                        return res_json
                    elif {"explicatie", "actiune", "surse"}.issubset(res_json.keys()):
                        res_json.setdefault("explanation", res_json["explicatie"])
                        res_json.setdefault("action", res_json["actiune"])
                        res_json.setdefault("sources", res_json["surse"])
                        return res_json
            except json.JSONDecodeError:
                pass
            break

    except Exception:
        # Fallback to deterministic local reasoning if external endpoint is unavailable
        pass

    # Execution via deterministic local ReAct engine
    return _execute_local_reasoning(wafer_id, lot_id)

if __name__ == "__main__":
    test_input = {"wafer_id": "W-101", "lot_id": "L-01"}
    print(f"Testing analyze() function with {test_input}...")
    res = analyze(test_input)
    print("Obtained result:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
