"""Backend WaferSight. Pornire (din rădăcina proiectului):  uvicorn api.main:app --reload"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import service as svc
from api.schemas import (AnalyzeRequest, AnalyzeResponse, CostRequest, CostResult, FeedbackRequest,
                         FeedbackResponse, LotRequest, LotResponse)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app):
    svc._boot()            # încărcăm modelul la pornire, nu la prima cerere
    yield


app = FastAPI(title="WaferSight API", version="1.0", lifespan=lifespan,
              description="Clasificare wafer map, detector de necunoscut, drift pe lot, agent LLM și cost.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    """'mode': 'model' dacă weights.pt și history.npz sunt încărcate, altfel 'mock'."""
    return svc.status()


@app.get("/lots")
def lots():
    """Loturile sintetice din data/lots.json (pentru selectorul din UI)."""
    if not svc._boot():
        return []
    return svc.lots()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not svc._boot():
        return svc.mock_response(req.wafer_id)
    try:
        return svc.analyze(req.wafer_id, req.wafer, req.lot_id, req.use_llm, req.include_gradcam, req.cost)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/lot", response_model=LotResponse)
def lot(req: LotRequest):
    if not svc._boot():
        raise HTTPException(503, f"modelul nu e încărcat: {svc._state['error']}")
    wafers = [w.model_dump() for w in req.wafers] if req.wafers is not None else None
    try:
        return svc.analyze_lot(req.lot_id, wafers, req.use_llm, req.include_maps, req.cost)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Salvează 'corect/greșit'. Eticheta confirmată se folosește la căutarea de wafere similare."""
    return svc.add_feedback(req)


@app.get("/feedback")
def feedback_list(limit: int = 50):
    return svc.list_feedback(limit)


@app.post("/cost", response_model=CostResult)
def cost(req: CostRequest):
    """Recalculează costul când utilizatorul mută sliderele, fără să mai ruleze modelul."""
    return svc.cost(req.p_defect, req.wafere_ramase, req.params)
