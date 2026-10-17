# llm/test_agent.py
import json
import sys
from pathlib import Path

# Ensure local module resolution regardless of current working directory
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import analyze

def run_tests():
    scenario_path = Path(__file__).resolve().parent / "demo_scenarios.json"
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"Starting tests for {len(scenarios)} scenarios...\n")

    for item in scenarios:
        print(f"Testing Scenario #{item['id']}: {item['title']}...")
        result = analyze(item["input"])
        
        # Validate output JSON schema
        if not isinstance(result, dict):
            print(f"❌ FAILURE: Result is not a dictionary: {result}")
            sys.exit(1)
        
        has_english_keys = {"explanation", "action", "sources"}.issubset(result.keys())
        has_romanian_keys = {"explicatie", "actiune", "surse"}.issubset(result.keys())

        if not (has_english_keys or has_romanian_keys):
            print(f"❌ FAILURE: Response does not contain required keys: {result}")
            sys.exit(1)
        
        sources = result.get("sources") or result.get("surse")
        if not isinstance(sources, list) or len(sources) == 0:
            print(f"❌ FAILURE: Sources key must be a non-empty list: {sources}")
            sys.exit(1)

        explanation = result.get("explanation") or result.get("explicatie", "")
        action = result.get("action") or result.get("actiune", "")

        print(f"  ✓ Explanation: {explanation[:80]}...")
        print(f"  ✓ Action:      {action[:80]}...")
        print(f"  ✓ Sources:     {sources}")
        print("  Status: OK\n")

    print("========================================")
    print("All 6 scenarios passed successfully!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
