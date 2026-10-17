"""Contractul JSON dintre componente ("legea"). UI-ul (ui/app.py) citește cheile în engleză."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Exemplele din /docs sunt răspunsuri reale ale API-ului (lotul B, wafer-ul B_w10), cu hărțile scurtate.
_MAP = [[0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 1], [0, 1, 1, 0]]
_COST = {"cost_rebut": 13500.0, "cost_oprire": 10000.0, "p_defect": 0.6, "wafere_ramase": 15, "recomandare": "opreste"}
_DRIFT = {"alarm": True, "repeated_class": "Edge-Ring", "affected": 3,
          "window": ["none", "none", "Edge-Ring", "Edge-Ring", "Edge-Ring"]}
_ANALYZE = {
    "wafer_id": "B_w10", "lot_id": "B", "class_name": "Edge-Ring", "is_defect": True, "confidence": 0.999,
    "is_unknown": False, "uncertain": False,
    "explanation": "Clasa Edge-Ring este confirmată cu încredere maximă (0.999) și istoricul este perfect relevant "
                   "(acord 1.0). Cauzele probabile sunt eroziunea inelului de focalizare în gravarea cu plasmă sau "
                   "răcirea deficitară cu heliu pe marginea ESC. Alarma de drift este activă (aceeași clasă pe 3 din "
                   "ultimele 5 wafere), deci problema este sistemică și afectează tot lotul.",
    "action": "Verifică imediat inelul de focalizare din camera de gravare și presiunea de heliu pe marginea ESC. "
              "Oprește procesarea și inspectează tot lotul B.",
    "sources": ["causes.json"],
    "similar_images": [{"wafer_id": "history_12889", "class_name": "Edge-Ring", "probability": 99.5, "wafer": _MAP},
                       {"wafer_id": "history_1262", "class_name": "Edge-Ring", "probability": 99.3, "wafer": _MAP}],
    "drift": _DRIFT, "cost": _COST, "wafer": _MAP, "gradcam_png": "iVBORw0KGgoAAAANSUhEUgAA...",
}


# ---------- cost ----------

class CostParams(BaseModel):
    """Ipotezele motorului de cost; UI-ul le poate schimba din slidere."""
    pret_wafer: float = Field(1500.0, ge=0, description="Preț per wafer (USD)")
    ore_oprire: float = Field(2.0, ge=0, description="Câte ore durează oprirea și verificarea echipamentului")
    cost_ora: float = Field(5000.0, ge=0, description="Cost per oră de linie oprită (USD)")
    wafere_ramase: Optional[int] = Field(None, ge=0, description="Câte wafere mai urmează; implicit restul lotului")


class CostResult(BaseModel):
    cost_rebut: float = Field(description="wafere_ramase × pret_wafer × p_defect")
    cost_oprire: float = Field(description="ore_oprire × cost_ora")
    p_defect: float
    wafere_ramase: int
    recomandare: str = Field(description="'opreste' sau 'continua'")

    model_config = ConfigDict(json_schema_extra={"examples": [_COST]})


# ---------- /analyze ----------

class AnalyzeRequest(BaseModel):
    wafer_id: str = Field(description="ID din data/lots.json (ex. 'B_w10'), un history_id (ex. '123') sau un ID liber")
    wafer: Optional[list[list[int]]] = Field(None, description="Harta wafer-ului (0 fundal, 1 die bun, 2 die picat). "
                                                              "Lipsă = se ia din istoric după wafer_id")
    lot_id: Optional[str] = Field(None, description="Lotul; pentru wafere trimise ca matrice ține evidența driftului")
    use_llm: bool = Field(True, description="False = răspuns determinist, fără apel la LLM")
    include_gradcam: bool = True
    cost: CostParams = Field(default_factory=CostParams)
    lang: Literal["ro", "en"] = Field("ro", description="Limba pentru explicație și acțiune")

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"wafer_id": "B_w10", "use_llm": True, "include_gradcam": True,
         "cost": {"pret_wafer": 1500, "ore_oprire": 2, "cost_ora": 5000}},
    ]})


class SimilarImage(BaseModel):
    wafer_id: str
    class_name: str
    probability: float = Field(ge=0, le=100, description="Similaritate cosinus × 100")
    wafer: list[list[int]]


class Drift(BaseModel):
    alarm: bool
    repeated_class: Optional[str] = None
    affected: int = 0
    window: list[str] = []


class AnalyzeResponse(BaseModel):
    wafer_id: str
    lot_id: Optional[str] = None
    class_name: str = Field(description="Clasa prezisă sau 'necunoscut'")
    is_defect: bool
    confidence: float = Field(ge=0, le=1)
    is_unknown: bool = Field(description="Pattern în afara claselor cunoscute (detectorul de necunoscut)")
    uncertain: bool = Field(description="Încredere mică și vecinii din istoric nu confirmă clasa")
    explanation: str
    action: str
    sources: list[str]
    similar_images: list[SimilarImage]
    drift: Drift
    cost: CostResult
    wafer: list[list[int]] = Field(description="Harta 64x64 analizată")
    gradcam_png: Optional[str] = Field(None, description="PNG base64 cu zona care a decis clasa")
    ai_generated: bool = Field(False, description="Explicația vine de la LLM (False = răspuns determinist)")

    model_config = ConfigDict(json_schema_extra={"examples": [_ANALYZE]})


# ---------- /lot ----------

class LotWafer(BaseModel):
    wafer_id: str
    wafer: Optional[list[list[int]]] = None


class LotRequest(BaseModel):
    lot_id: str = Field(description="Un lot din data/lots.json (A, B, C, D) sau un ID nou")
    wafers: Optional[list[LotWafer]] = Field(None, description="Lipsă = wafere luate din data/lots.json")
    use_llm: bool = Field(True, description="Se aplică doar analizei detaliate a primului wafer cu alarmă")
    include_maps: bool = True
    cost: CostParams = Field(default_factory=CostParams)
    lang: Literal["ro", "en"] = Field("ro", description="Limba pentru analiza agentului la alarmă")

    model_config = ConfigDict(json_schema_extra={"examples": [{"lot_id": "B", "use_llm": True, "include_maps": False}]})


class LotItem(BaseModel):
    index: int
    wafer_id: str
    class_name: str
    confidence: float
    is_unknown: bool
    alarm: bool = Field(description="Alarma de drift era activă la acest wafer (încadrat cu roșu în UI)")
    wafer: Optional[list[list[int]]] = None


class LotResponse(BaseModel):
    lot_id: str
    n_wafers: int
    alarm: bool
    first_alarm_index: Optional[int] = Field(None, description="Poziția (1..N) la care a pornit alarma")
    repeated_class: Optional[str] = None
    wafers: list[LotItem]
    cost: Optional[CostResult] = Field(None, description="Calculat la primul wafer cu alarmă")
    alarm_analysis: Optional[AnalyzeResponse] = Field(None, description="Analiza agentului pe primul wafer cu alarmă")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "lot_id": "B", "n_wafers": 25, "alarm": True, "first_alarm_index": 10, "repeated_class": "Edge-Ring",
        "wafers": [
            {"index": 9, "wafer_id": "B_w09", "class_name": "Edge-Ring", "confidence": 1.0, "is_unknown": False,
             "alarm": False, "wafer": None},
            {"index": 10, "wafer_id": "B_w10", "class_name": "Edge-Ring", "confidence": 0.999, "is_unknown": False,
             "alarm": True, "wafer": None},
            {"index": 11, "wafer_id": "B_w11", "class_name": "Edge-Ring", "confidence": 0.981, "is_unknown": False,
             "alarm": True, "wafer": None},
        ],
        "cost": _COST, "alarm_analysis": _ANALYZE,
    }]})


# ---------- /feedback ----------

class FeedbackRequest(BaseModel):
    wafer_id: str
    predicted_class: str
    correct: bool
    true_class: Optional[str] = Field(None, description="Clasa corectă, dacă predicția e greșită")
    wafer: Optional[list[list[int]]] = Field(None, description="Necesar doar pentru wafere care nu sunt în istoric")
    comment: Optional[str] = None

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"wafer_id": "B_w10", "predicted_class": "Edge-Ring", "correct": True},
        {"wafer_id": "D_w11", "predicted_class": "Edge-Loc", "correct": False, "true_class": "Loc",
         "comment": "defect local, nu pe margine"},
    ]})


class FeedbackResponse(BaseModel):
    id: int
    stored: bool
    used_for_similar: bool = Field(description="Wafer-ul intră în căutarea de similare cu eticheta confirmată")

    model_config = ConfigDict(json_schema_extra={"examples": [{"id": 1, "stored": True, "used_for_similar": True}]})


class FeedbackItem(BaseModel):
    id: int
    ts: float = Field(description="Unix timestamp")
    wafer_id: str
    history_id: Optional[int] = None
    predicted_class: str
    correct: bool
    true_class: Optional[str] = None
    comment: Optional[str] = None

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"id": 1, "ts": 1792252800.0, "wafer_id": "B_w10", "history_id": 812, "predicted_class": "Edge-Ring",
         "correct": True, "true_class": None, "comment": None},
    ]})


# ---------- /cost ----------

class CostRequest(BaseModel):
    p_defect: float = Field(ge=0, le=1)
    wafere_ramase: int = Field(ge=0)
    params: CostParams = Field(default_factory=CostParams)

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"p_defect": 0.6, "wafere_ramase": 15, "params": {"pret_wafer": 1500, "ore_oprire": 2, "cost_ora": 5000}},
    ]})


# ---------- /health, /lots ----------

class HealthResponse(BaseModel):
    mode: str = Field(description="'model' dacă weights.pt și history.npz sunt încărcate, altfel 'mock'")
    error: Optional[str] = None
    history: int = Field(description="Câte wafere sunt în history.npz")
    lots: dict[str, int] = Field(description="Lot -> număr de wafere")
    llm_model: Optional[str] = None
    rag: bool = Field(False, description="Indexul RAG (llm/rag.py) e construit; altfel explicațiile vin din causes.json")
    rag_error: Optional[str] = None

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"mode": "model", "error": None, "history": 30519, "lots": {"A": 25, "B": 25, "C": 25, "D": 25},
         "llm_model": "Qwen3.8-Flash-Next", "rag": True, "rag_error": None},
    ]})


class LotInfo(BaseModel):
    lot_id: str
    n_wafers: int
    echipament: Optional[str] = None
    wafer_ids: list[str]

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"lot_id": "B", "n_wafers": 25, "echipament": "ETCH-03", "wafer_ids": ["B_w01", "B_w02", "...", "B_w25"]},
    ]})


# ---------- /upload ----------

class UploadedWafer(BaseModel):
    wafer_id: str = Field(description="Numele fișierului (sau ID-ul din fișier), unic în sesiune")
    source_file: str
    format: str = Field(description="JSON, CSV, TXT, TSV, NPY, NPZ sau Image")
    shape: list[int] = Field(description="Dimensiunea originală a hărții")
    wafer: list[list[int]] = Field(description="Harta 0/1/2 (micșorată la 64x64 dacă e mai mare de 128)")
    warnings: list[str] = []
    stats: dict[str, float] = Field(description="dies, fail, fail_pct")


class UploadFileError(BaseModel):
    file: str
    error: str


class UploadResponse(BaseModel):
    wafers: list[UploadedWafer]
    errors: list[UploadFileError]

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "wafers": [{"wafer_id": "my_wafer", "source_file": "my_wafer.json", "format": "JSON", "shape": [4, 4],
                    "wafer": _MAP, "warnings": [], "stats": {"dies": 12, "fail": 4, "fail_pct": 33.3}}],
        "errors": [{"file": "notes.docx", "error": "unsupported file type '.docx'. Use JSON, CSV, TXT, NPY, NPZ, PNG or JPG"}],
    }]})
