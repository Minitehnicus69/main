# WaferSight: backend FastAPI + UI (ui/ servit la "/"). Un singur container, un singur port.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# torch doar pentru CPU (de câteva ori mai mic decât varianta cu CUDA); restul găsește torch deja instalat
COPY api/requirements.txt api/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r api/requirements.txt

# modelul de embedding pentru RAG intră în imagine, ca pornirea să nu depindă de HuggingFace
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .

# Railway (și alte platforme) dau portul în $PORT
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
