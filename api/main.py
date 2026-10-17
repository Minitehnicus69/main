"""Backend WaferSight. Pornire (din rădăcina proiectului):  uvicorn api.main:app --reload"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import service as svc
from api import upload as up
from api.schemas import (AnalyzeRequest, AnalyzeResponse, CostRequest, CostResult, FeedbackItem, FeedbackRequest,
                         FeedbackResponse, HealthResponse, LotInfo, LotRequest, LotResponse, UploadResponse)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "faiss.loader", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app):
    svc._boot()            # încărcăm modelul la pornire, nu la prima cerere
    yield


app = FastAPI(title="WaferSight API", version="1.0", lifespan=lifespan,
              description="Clasificare wafer map, detector de necunoscut, drift pe lot, agent LLM și cost.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health():
    """'mode': 'model' dacă weights.pt și history.npz sunt încărcate, altfel 'mock'."""
    return svc.status()


@app.get("/lots", response_model=list[LotInfo])
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
        return svc.analyze(req.wafer_id, req.wafer, req.lot_id, req.use_llm, req.include_gradcam, req.cost,
                           lang=req.lang)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/lot", response_model=LotResponse)
def lot(req: LotRequest):
    if not svc._boot():
        raise HTTPException(503, f"modelul nu e încărcat: {svc._state['error']}")
    wafers = [w.model_dump() for w in req.wafers] if req.wafers is not None else None
    try:
        return svc.analyze_lot(req.lot_id, wafers, req.use_llm, req.include_maps, req.cost, lang=req.lang)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Salvează 'corect/greșit'. Eticheta confirmată se folosește la căutarea de wafere similare."""
    return svc.add_feedback(req)


@app.get("/feedback", response_model=list[FeedbackItem])
def feedback_list(limit: int = 50):
    return svc.list_feedback(limit)


@app.post("/cost", response_model=CostResult)
def cost(req: CostRequest):
    """Recalculează costul când utilizatorul mută sliderele, fără să mai ruleze modelul."""
    return svc.cost(req.p_defect, req.wafere_ramase, req.params)


@app.post("/upload", response_model=UploadResponse)
def upload_wafers(files: list[UploadFile] = File(..., description="JSON, CSV/TXT, NPY/NPZ sau imagini PNG/JPG"),
                  existing: str = Form("", description="ID-uri deja folosite în UI, separate prin virgulă")):
    """Citește hărțile încărcate și le întoarce validate (0/1/2), fără să ruleze modelul.
    Hărțile se analizează apoi cu /analyze (un wafer) sau /lot (toate, în ordine)."""
    wafers, errors = [], []
    for f in files:
        name = f.filename or "upload"
        try:
            wafers += up.parse_file(name, f.file.read(up.MAX_BYTES + 1))
        except up.UploadError as e:
            errors.append({"file": name, "error": str(e)})
        except Exception as e:
            logging.getLogger("api").warning("upload %s: %r", name, e)
            errors.append({"file": name, "error": f"could not read file ({type(e).__name__})"})
    if len(wafers) > up.MAX_WAFERS:
        errors.append({"file": "*", "error": f"{len(wafers)} wafers found; only the first {up.MAX_WAFERS} were kept"})
        wafers = wafers[:up.MAX_WAFERS]
    up.unique_ids(wafers, [s for s in existing.split(",") if s])
    return {"wafers": wafers, "errors": errors}


@app.get("/upload/sample")
def upload_sample(fmt: Literal["json", "csv"] = "json"):
    """Fișier de exemplu cu câte un wafer real din istoric pentru fiecare clasă."""
    if not svc._boot():
        raise HTTPException(503, f"modelul nu e încărcat: {svc._state['error']}")
    wafers = svc.sample_wafers()
    body = up.to_csv(wafers) if fmt == "csv" else json.dumps({"wafers": wafers})
    return Response(body, media_type="text/csv" if fmt == "csv" else "application/json",
                    headers={"Content-Disposition": f'attachment; filename="wafersight_sample.{fmt}"'})


# UI-ul (ui/index.html) servit de același server, la "/". Montat ultimul, ca să nu acopere rutele API.
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
