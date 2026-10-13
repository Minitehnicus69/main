# llm/test_agent.py
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import analyze

def run_tests():
    scenario_path = Path(__file__).resolve().parent / "demo_scenarios.json"
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"Începere teste pentru {len(scenarios)} scenarii...\n")
    required_keys = {"explicatie", "actiune", "surse"}

    for item in scenarios:
        print(f"Testare Scenariu #{item['id']}: {item['title']}...")
        result = analyze(item["input"])
        
        # Validare schemă JSON de ieșire
        if not isinstance(result, dict) or not required_keys.issubset(result.keys()):
            print(f"❌ EȘEC: Răspunsul nu conține cheile obligatorii: {result}")
            sys.exit(1)
        
        if not isinstance(result["surse"], list) or len(result["surse"]) == 0:
            print(f"❌ EȘEC: Cheia 'surse' trebuie să fie o listă nevidă: {result['surse']}")
            sys.exit(1)

        print(f"  ✓ Explicație: {result['explicatie'][:80]}...")
        print(f"  ✓ Acțiune:    {result['actiune'][:80]}...")
        print(f"  ✓ Surse:      {result['surse']}")
        print("  Status: OK\n")

    print("========================================")
    print("Toate cele 6 scenarii au trecut cu succes!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
