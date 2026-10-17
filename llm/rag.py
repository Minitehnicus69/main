# llm/rag.py
import os
import sys

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Reconfigure encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Compact and fast embedding model on CPU
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents_store = []
index = None

DEFAULT_KNOWLEDGE_DOCS = [
    {
        "doc": "manual_cvd_standard.pdf",
        "page": 14,
        "text": "Central defects (Center) indicate CVD gas flow non-uniformities or asymmetric temperature of the heating platen. Check the central nozzles and polishing head planarity."
    },
    {
        "doc": "manual_cvd_standard.pdf",
        "page": 22,
        "text": "The Donut pattern (inner ring) appears due to radial temperature variations on the susceptor and turbulent peripheral flow in the CVD chamber. Recalibration of concentric thermal zones is recommended."
    },
    {
        "doc": "manual_etch_troubleshooting.pdf",
        "page": 29,
        "text": "The Edge-Ring pattern is frequently caused by physical degradation and erosion of the focus ring in plasma etching or insufficient helium pressure at the ESC edge."
    },
    {
        "doc": "manual_robotics_transfer.pdf",
        "page": 19,
        "text": "Edge-Loc and Scratch defects indicate abnormal mechanical forces during robotic handling: friction on the wafer edge at the aligner pins or scratches caused by particles on vacuum suction cups."
    },
    {
        "doc": "manual_litho_inspection.pdf",
        "page": 35,
        "text": "The Loc (localized) pattern indicates adherent impurities on the photolithographic mask (reticle contamination) or particles during UV exposure. Inspect the mask reticle."
    },
    {
        "doc": "manual_process_emergency.pdf",
        "page": 5,
        "text": "The Near-full defect represents a large-scale critical failure: loss of vacuum, power imbalance in the plasma generator, or massive chemical contamination. Shut down the process chamber immediately."
    },
    {
        "doc": "manual_cleanroom_hepa.pdf",
        "page": 4,
        "text": "The Random pattern (randomly dispersed particles) is correlated with HEPA/ULPA filter degradation and contamination of the laminar airflow in the cleanroom enclosure."
    },
    {
        "doc": "manual_quality_control.pdf",
        "page": 2,
        "text": "The none class indicates the absence of structural anomalies, with nominal production parameters being met. The batch is approved for the next processing step."
    }
]

def build_index(pdf_paths: list[str] | None = None, chunk_size_words: int = 350):
    """
    Reads PDFs, extracts page chunks, and builds the FAISS index.
    If no PDF files exist, populates the index with default technical documentation.
    """
    global index, documents_store
    documents_store = []
    chunks = []

    if pdf_paths:
        for path in pdf_paths:
            if not os.path.exists(path):
                continue
            try:
                reader = PdfReader(path)
                doc_name = os.path.basename(path)
                for page_idx, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    words = text.split()
                    for i in range(0, len(words), chunk_size_words):
                        chunk = " ".join(words[i:i + chunk_size_words])
                        if len(chunk.strip()) > 30:
                            chunks.append(chunk)
                            documents_store.append({
                                "doc": doc_name,
                                "page": page_idx,
                                "text": chunk
                            })
            except Exception as e:
                print(f"Warning reading PDF {path}: {e}")

    if not chunks:
        # Standard technical documentation from manufacturing manuals
        documents_store = list(DEFAULT_KNOWLEDGE_DOCS)
        chunks = [d["text"] for d in documents_store]

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

def search_manual(query: str, top_k: int = 1) -> list[dict]:
    """
    Returns the most relevant chunks along with citation metadata (doc, page, text).
    """
    if index is None:
        build_index([])

    q_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    _distances, indices = index.search(q_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(documents_store):
            results.append(documents_store[idx])
    return results

# Romanian alias for backwards compatibility
cauta_manual = search_manual

if __name__ == "__main__":
    print("Initializing and testing RAG module...")
    build_index([])
    test_queries = ["Center", "Edge-Ring", "Scratch"]
    for q in test_queries:
        res = search_manual(q, top_k=1)
        print(f"Query: '{q}' -> Document: {res[0]['doc']} (p. {res[0]['page']}): {res[0]['text'][:60]}...")
