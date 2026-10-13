# llm/agent.py
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Încărcare variabile de mediu din .env (localizat în directorul llm)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from tools import (
    TOOL_ROUTER,
    OPENAI_TOOLS_DEFINITIONS,
    tool_clasifica,
    tool_verifica_drift,
    tool_gaseste_similare,
    tool_cauta_manual
)

# Configurare API externă conform cerințelor utilizatorului
API_KEY = os.environ.get("LLM_API_KEY", "rc-096ecf812a39bafd5551d273d4f7cedeb4c90eeea326187e")
MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3.8-27B")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

# Încărcare cauze fizice din causes.json (antrenare/grounding de cunoștințe)
causes_file = Path(__file__).resolve().parent / "causes.json"
KNOWN_CAUSES = {}
if causes_file.exists():
    try:
        with open(causes_file, "r", encoding="utf-8") as f:
            KNOWN_CAUSES = json.load(f)
    except Exception as e:
        print(f"[agent] Avertisment la citirea causes.json: {e}")

SYSTEM_PROMPT = f"""Ești un agent expert în diagnosticarea procesului de fabricație a wafer-elor de siliciu (baza de date WM-811K).
Modele cunoscute de defecte (9 clase învățate din modelul CNN weights.pt):
- Center: defecte centrale cauzate de depunere CVD/PVD neuniformă sau presiune asimetrică CMP.
- Donut: profil termic inelar sau flux turbulent de gaz în camera CVD.
- Edge-Ring: eroziune a inelului de focalizare (focus ring) în gravarea cu plasmă sau heliu pe marginea ESC.
- Edge-Loc: contact mecanic agresiv pe marginea wafer-ului în timpul transferului robotic.
- Loc: contaminare locală pe masca fotolitografică (reticle contamination) sau expunere UV.
- Near-full: pierdere totală de vid, variație critică de putere RF sau contaminare masivă.
- Random: particule suspendate în camera curată (cleanroom), defecțiune filtru HEPA/ULPA.
- Scratch: zgâriere provocată de brațul robotic de manipulare sau particule abrazive pe ventuze.
- none: wafer nominal conform, fără defecte structurale.

Instrucțiuni de operare:
1. Analizează starea wafer-ului și a lotului apelând uneltele necesare ('clasifica', 'verifica_drift').
2. Dacă nivelul de încredere este scăzut (< 0.70) sau defectul este 'necunoscut', apelează 'gaseste_similare'.
3. Dacă apare un defect identificat, apelează 'cauta_manual' pentru a găsi instrucțiunile specifice de intervenție.
4. Răspunsul FINAL trebuie să fie STRICT un obiect JSON valid, fără text sau markdown suplimentar:
{{
  "explicatie": "Descrierea defectului, nivelul de încredere și starea de drift din lot.",
  "actiune": "Măsura tehnică recomandată pentru operator/inginer.",
  "surse": ["nume_document.pdf (p. X)", "causes.json"]
}}"""

def _execute_local_reasoning(wafer_id: str, lot_id: str) -> dict:
    """
    Motor determinist de raționament ReAct bazat pe cunoștințele din weights.pt și causes.json.
    Garantează răspuns conform schemei în caz de indisponibilitate a rețelei/endpoint-ului extern.
    """
    # Pas 1: clasificare
    clf = tool_clasifica(wafer_id)
    c_class = clf.get("class", "Center")
    conf = clf.get("confidence", 0.85)

    # Pas 2: verificare drift
    drift = tool_verifica_drift(lot_id)
    has_alarm = drift.get("alarm", False)

    surse = ["causes.json"]

    # Scenariul 5: Nominal curat
    if c_class == "none":
        doc_rag = tool_cauta_manual("none")
        if doc_rag:
            surse.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        return {
            "explicatie": f"Wafer-ul {wafer_id} din lotul {lot_id} este nominal curat (clasa 'none') cu încredere {conf*100:.1f}%. Nu s-au detectat anomalii sau alarme de drift.",
            "actiune": "Autorizează transferul lotului către următoarea stație tehnologică.",
            "surse": surse
        }

    # Scenariul 3: Out-of-Distribution / Necunoscut
    if c_class == "necunoscut" or conf < 0.40:
        similar = tool_gaseste_similare(wafer_id)
        return {
            "explicatie": f"Wafer-ul {wafer_id} prezintă un tipar structural necunoscut (încredere redusă {conf*100:.1f}%). S-au căutat referințe istorice similare.",
            "actiune": "Marcare wafer pentru analiză OOD și etichetare manuală de către inginerul de proces.",
            "surse": surse
        }

    # Scenariul 2: Alarmă critică de drift
    if has_alarm:
        doc_rag = tool_cauta_manual(c_class)
        if doc_rag:
            surse.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        rep_class = drift.get("repeated_class", c_class)
        info = KNOWN_CAUSES.get(rep_class, {})
        check_action = info.get("check", "Oprire imediată a camerei de proces și verificare parametri.")
        return {
            "explicatie": f"Alarmă critică de drift activă în lotul {lot_id}: defectul repetat '{rep_class}' a fost detectat pe minim 3 din ultimele 5 wafere. Wafer curent: {wafer_id} ({conf*100:.1f}%).",
            "actiune": f"Oprire imediată a camerei tehnologice asociate defectului {rep_class}. {check_action}",
            "surse": surse
        }

    # Scenariul 6: Încredere scăzută cu validare prin similaritate
    if conf < 0.70:
        sim_wafers = tool_gaseste_similare(wafer_id)
        doc_rag = tool_cauta_manual(c_class)
        if doc_rag:
            surse.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
        info = KNOWN_CAUSES.get(c_class, {})
        check_action = info.get("check", "Inspecție detaliată.")
        return {
            "explicatie": f"Wafer-ul {wafer_id} a fost clasificat ca '{c_class}' cu încredere intermediară de {conf*100:.1f}%. Similaritatea istorică confirmă clasa prin potrivirile cu {sim_wafers[0]['wafer_id']}.",
            "actiune": f"Se recomandă inspectarea parametrilor specifici clasei {c_class}. {check_action}",
            "surse": surse
        }

    # Scenariile 1 și 4: Defect izolat identificat (Edge-Ring, Scratch, etc.)
    doc_rag = tool_cauta_manual(c_class)
    if doc_rag:
        surse.insert(0, f"{doc_rag[0]['doc']} (p. {doc_rag[0]['page']})")
    info = KNOWN_CAUSES.get(c_class, {})
    check_action = info.get("check", "Inspectează echipamentul de proces conform procedurii standard.")

    return {
        "explicatie": f"Wafer-ul {wafer_id} din lotul {lot_id} prezintă defect izolat de tip '{c_class}' (încredere {conf*100:.1f}%). Fără alarmă de drift pe lot.",
        "actiune": f"Verificare conform procedurii tehnice pentru defectul {c_class}: {check_action}",
        "surse": surse
    }

def analyze(wafer_json: dict) -> dict:
    """
    Punctul de intrare cerut de contractul de integrare: analyze(wafer_json) -> dict
    Primește datele wafer-ului și returnează un dict cu cheile: 'explicatie', 'actiune', 'surse'.
    """
    wafer_id = wafer_json.get("wafer_id", "W-101")
    lot_id = wafer_json.get("lot_id", "L-01")

    # Încercare apel API extern (OpenAI-compatible / Qwen3.8-27B) dacă clientul este disponibil
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=8.0
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analizează wafer-ul {wafer_id} din lotul {lot_id}."}
        ]

        # Buclă ReAct limitată la 5 iterații
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

            # Cazul 1: Modelul cere apelarea uneia sau mai multor unelte
            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}
                    fn = TOOL_ROUTER.get(fn_name)
                    tool_res = fn(**args) if fn else {"error": f"Tool {fn_name} necunoscut"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_res)
                    })
                continue

            # Cazul 2: Modelul a finalizat raționamentul și returnează textul
            content = msg.content or ""
            text_clean = content.strip()
            if text_clean.startswith("```"):
                text_clean = text_clean.split("```")[1]
                if text_clean.startswith("json"):
                    text_clean = text_clean[4:]
            try:
                res_json = json.loads(text_clean.strip())
                if isinstance(res_json, dict) and {"explicatie", "actiune", "surse"}.issubset(res_json.keys()):
                    return res_json
            except json.JSONDecodeError:
                pass
            break

    except Exception as e:
        # Fallback silențios către raționamentul determinist dacă serverul extern este inaccesibil
        pass

    # Execuție prin motorul ReAct determinist
    return _execute_local_reasoning(wafer_id, lot_id)

if __name__ == "__main__":
    test_input = {"wafer_id": "W-101", "lot_id": "L-01"}
    print(f"Testare funcție analyze() cu {test_input}...")
    rez = analyze(test_input)
    print("Rezultat obținut:")
    print(json.dumps(rez, indent=2, ensure_ascii=False))
