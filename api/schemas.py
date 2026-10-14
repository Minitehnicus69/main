"""Contractul JSON dintre componente ("legea"). UI-ul (ui/app.py) citește cheile în engleză."""
from typing import Optional

from pydantic import BaseModel, Field


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


# ---------- /analyze ----------

class AnalyzeRequest(BaseModel):
    wafer_id: str = Field(description="ID din data/lots.json (ex. 'B_w10'), un history_id (ex. '123') sau un ID liber")
    wafer: Optional[list[list[int]]] = Field(None, description="Harta wafer-ului (0 fundal, 1 die bun, 2 die picat). "
                                                              "Lipsă = se ia din istoric după wafer_id")
    lot_id: Optional[str] = Field(None, description="Lotul; pentru wafere trimise ca matrice ține evidența driftului")
    use_llm: bool = Field(True, description="False = răspuns determinist, fără apel la LLM")
    include_gradcam: bool = True
    cost: CostParams = Field(default_factory=CostParams)


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


# ---------- /feedback ----------

class FeedbackRequest(BaseModel):
    wafer_id: str
    predicted_class: str
    correct: bool
    true_class: Optional[str] = Field(None, description="Clasa corectă, dacă predicția e greșită")
    wafer: Optional[list[list[int]]] = Field(None, description="Necesar doar pentru wafere care nu sunt în istoric")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    stored: bool
    used_for_similar: bool = Field(description="Wafer-ul intră în căutarea de similare cu eticheta confirmată")


# ---------- /cost ----------

class CostRequest(BaseModel):
    p_defect: float = Field(ge=0, le=1)
    wafere_ramase: int = Field(ge=0)
    params: CostParams = Field(default_factory=CostParams)
