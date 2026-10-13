# llm/test_agent.py
import json
import sys
from pathlib import Path

# Asigură rezolvarea modulelor locale indiferent de directorul curent de lucru
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

    print(f"Incepere teste pentru {len(scenarios)} scenarii...\n")
    required_keys = {"explicatie", "actiune", "surse"}

    for item in scenarios:
        print(f"Testare Scenariu #{item['id']}: {item['title']}...")
        result = analyze(item["input"])
        
        # Validare schema JSON de iesire
        if not isinstance(result, dict) or not required_keys.issubset(result.keys()):
            print(f"❌ ESEC: Raspunsul nu contine cheile obligatorii: {result}")
            sys.exit(1)
        
        if not isinstance(result["surse"], list) or len(result["surse"]) == 0:
            print(f"❌ ESEC: Cheia 'surse' trebuie sa fie o lista nevida: {result['surse']}")
            sys.exit(1)

        print(f"  ✓ Explicatie: {result['explicatie'][:80]}...")
        print(f"  ✓ Actiune:    {result['actiune'][:80]}...")
        print(f"  ✓ Surse:      {result['surse']}")
        print("  Status: OK\n")

    print("========================================")
    print("Toate cele 6 scenarii au trecut cu succes!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
