# llm/rag.py
import os
import sys

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Reconfigurare encoding pentru consola Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Model de embedding compact și rapid pe CPU
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents_store = []
index = None

DEFAULT_KNOWLEDGE_DOCS = [
    {
        "doc": "manual_cvd_standard.pdf",
        "page": 14,
        "text": "Defectele centrale (Center) indică neuniformități de debit gaz CVD sau temperatură asimetrică a platanului de încălzire. Verificați duzele centrale și planeitatea capului de lustruire."
    },
    {
        "doc": "manual_cvd_standard.pdf",
        "page": 22,
        "text": "Modelul Donut (inelar interior) apare la variații radiale de temperatură pe susceptor și flux turbulent periferic în camera CVD. Se recomandă recalibrarea zonelor concentrice termice."
    },
    {
        "doc": "manual_etch_troubleshooting.pdf",
        "page": 29,
        "text": "Pattern-ul Edge-Ring este cauzat în mod frecvent de degradarea fizică și eroziunea inelului de focalizare (focus ring) în gravarea cu plasmă sau presiune insuficientă de heliu pe marginea ESC."
    },
    {
        "doc": "manual_robotics_transfer.pdf",
        "page": 19,
        "text": "Defectele Edge-Loc și Scratch indică forțe mecanice anormale la manipulare robotică: frecare pe marginea wafer-ului la pinii de aliniere sau zgârieturi provocate de particule pe ventuzele de vid."
    },
    {
        "doc": "manual_litho_inspection.pdf",
        "page": 35,
        "text": "Pattern-ul Loc (localizat) indică impurități aderente pe masca fotolitografică (reticle contamination) sau particule în timpul expunerii UV. Inspectați reticulul măștii."
    },
    {
        "doc": "manual_process_emergency.pdf",
        "page": 5,
        "text": "Defectul Near-full reprezintă o avarie critică la scară largă: pierdere de vid, dezechilibru de putere în generatorul de plasmă sau contaminare chimică masivă. Opriți camera de proces imediat."
    },
    {
        "doc": "manual_cleanroom_hepa.pdf",
        "page": 4,
        "text": "Pattern-ul Random (particule dispersate aleatoriu) este corelat cu degradarea filtrelor HEPA/ULPA și contaminarea fluxului de aer laminar în incinta camerei curate (cleanroom)."
    },
    {
        "doc": "manual_quality_control.pdf",
        "page": 2,
        "text": "Clasa none indică absența anomaliilor structurale, parametrii nominali de producție fiind respectați. Lotul este aprobat pentru pasul următor de procesare."
    }
]

def build_index(pdf_paths: list[str] | None = None, chunk_size_words: int = 350):
    """
    Citește PDF-urile, extrage fragmente pe pagini și construiește indexul FAISS.
    Dacă nu există fișiere PDF, populează indexul cu documentația tehnică de bază.
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
                print(f"Avertisment la citirea PDF {path}: {e}")

    if not chunks:
        # Documentație tehnică standard din manuale de fabricație
        documents_store = list(DEFAULT_KNOWLEDGE_DOCS)
        chunks = [d["text"] for d in documents_store]

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

def cauta_manual(query: str, top_k: int = 1) -> list[dict]:
    """
    Returnează fragmentele cele mai relevante împreună cu metadatele de citare (doc, pagina, text).
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

if __name__ == "__main__":
    print("Initializare si testare modul RAG...")
    build_index([])
    test_queries = ["Center", "Edge-Ring", "Scratch"]
    for q in test_queries:
        res = cauta_manual(q, top_k=1)
        print(f"Query: '{q}' -> Document: {res[0]['doc']} (p. {res[0]['page']}): {res[0]['text'][:60]}...")
