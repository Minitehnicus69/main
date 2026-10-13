import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


st.set_page_config(page_title="WaferSight", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    .stApp { background: radial-gradient(ellipse at 78% -15%, #18264a 0, #0b1020 48%); color: #eef3ff; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    [data-testid="stSidebar"] { background: #0e1527; border-right: 1px solid #202c45; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .eyebrow { color: #53e0d0; font: 500 11px 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
    .title { font-size: 36px; font-weight: 800; letter-spacing: -.055em; margin: 4px 0 2px; }
    .subtitle { color: #93a4c3; font-size: 14px; margin: 0 0 22px; }
    .section-label { color: #93a4c3; font: 500 10px 'DM Mono', monospace; letter-spacing: .12em; text-transform: uppercase; }
    .muted { color: #93a4c3; font-size: 13px; }
    .status-pill { display:inline-block; border-radius:99px; padding:6px 11px; background:rgba(83,224,208,.1); border:1px solid rgba(83,224,208,.25); color:#53e0d0; font:500 11px 'DM Mono',monospace; }
    .status-warn { background:rgba(255,107,122,.12); border-color:rgba(255,107,122,.3); color:#ff6b7a; }
    .metric-card { background:#111a2c; border:1px solid #24314b; border-radius:14px; padding:14px 16px; min-height:88px; }
    .metric-value { font-size:22px; font-weight:800; margin-top:6px; }
    .metric-caption { color:#93a4c3; font-size:11px; }
    div[data-testid="stMetric"] { background:#111a2c; border:1px solid #24314b; border-radius:14px; padding:12px 15px; }
    div[data-testid="stMetricLabel"] { color:#93a4c3; }
    div[data-testid="stButton"] button { border-radius:10px; font-weight:700; min-height:42px; }
    .stTabs [data-baseweb="tab-list"] { gap:20px; }
    hr { border-color:#24314b; }
    </style>
    """,
    unsafe_allow_html=True,
)


SAMPLES = {
    "WF-0248-001 · Edge-Ring": ("WF-0248-001", "Edge-Ring"),
    "WF-0248-002 · Normal": ("WF-0248-002", "Normal"),
    "WF-0248-003 · Center": ("WF-0248-003", "Center"),
    "WF-0248-004 · Cluster": ("WF-0248-004", "Cluster"),
}
PATTERN_NAMES = {
    "Edge-Ring": "Edge-Ring",
    "Center": "Defect centru",
    "Scratch": "Scratch",
    "Cluster": "Cluster",
    "Normal": "Fără defect detectat",
}
PALETTE = ["#10182a", "#526c8e", "#ff6578"]


def make_wafer(pattern: str, seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = size * 0.46
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    inside = distance <= radius
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[inside] = 1

    if pattern == "Edge-Ring":
        defects = inside & (distance > radius * 0.78) & (rng.random((size, size)) > 0.68)
    elif pattern == "Center":
        defects = inside & (distance < radius * 0.24) & (rng.random((size, size)) > 0.62)
    elif pattern == "Scratch":
        defects = inside & (np.abs((y - center) - 0.38 * (x - center)) < 2.4)
        defects &= rng.random((size, size)) > 0.35
    elif pattern == "Cluster":
        cluster = np.sqrt((x - size * 0.38) ** 2 + (y - size * 0.62) ** 2) < size * 0.17
        defects = inside & cluster & (rng.random((size, size)) > 0.55)
    else:
        defects = np.zeros((size, size), dtype=bool)
    wafer[defects] = 2
    return wafer


def mock_result(wafer_id: str, pattern: str) -> dict:
    if pattern == "Normal":
        return {
            "wafer_id": wafer_id,
            "class_name": PATTERN_NAMES[pattern],
            "is_defect": False,
            "confidence": 0.991,
            "explanation": "Harta nu prezintă un tipar semnificativ de die-uri defecte.",
            "action": "Continuă monitorizarea standard a lotului.",
            "sources": ["Clasificare demo · set de imagini sintetice"],
            "similar_images": [],
        }

    probabilities = [98.1, 95.7, 93.4, 90.8, 87.6]
    matches = [
        {
            "wafer_id": f"REF-{pattern.upper()}-{index + 1:03}",
            "probability": probability,
            "wafer": make_wafer(pattern, seed=400 + index),
        }
        for index, probability in enumerate(probabilities)
    ]
    return {
        "wafer_id": wafer_id,
        "class_name": PATTERN_NAMES[pattern],
        "is_defect": True,
        "confidence": 0.973,
        "explanation": (
            f"Distribuția defectelor este compatibilă cu tiparul {PATTERN_NAMES[pattern]}. "
            "Clasificarea este demonstrativă și se bazează pe imagini sintetice."
        ),
        "action": (
            "Compară cu waferele similare de mai jos și verifică echipamentul "
            "corespunzător înainte de a decide oprirea liniei."
        ),
        "sources": ["Comparație demo · galerie sintetică de wafer-e de referință"],
        "similar_images": matches,
    }


def draw_wafer(wafer: np.ndarray, title: str | None = None, compact: bool = False):
    color_map = plt.matplotlib.colors.ListedColormap(PALETTE)
    figure, axis = plt.subplots(
        figsize=(1.7, 1.7) if compact else (5.2, 5.2),
        facecolor="#0b1020",
    )
    axis.set_facecolor("#0b1020")
    axis.imshow(wafer, cmap=color_map, vmin=0, vmax=2, interpolation="nearest")
    axis.set_axis_off()
    if title:
        axis.set_title(title, color="#eef3ff", fontsize=10, pad=8)
    figure.tight_layout(pad=0)
    return figure


def show_wafer(wafer: np.ndarray, title: str | None = None, compact: bool = False):
    figure = draw_wafer(wafer, title, compact)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def analyze_with_api(api_url: str, wafer: np.ndarray, wafer_id: str) -> dict:
    response = requests.post(
        api_url.rstrip("/") + "/analyze",
        json={"wafer_id": wafer_id, "wafer": wafer.tolist()},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()
    matches = result.get("similar_images", [])
    if not isinstance(matches, list):
        raise ValueError("'similar_images' trebuie să fie o listă.")
    for match in matches:
        if not isinstance(match, dict) or "wafer" not in match or "probability" not in match:
            raise ValueError("Fiecare imagine similară trebuie să conțină 'wafer' și 'probability'.")
        image = np.asarray(match["wafer"])
        if image.ndim != 2:
            raise ValueError("Imaginea wafer din 'similar_images' trebuie să fie o matrice 2D.")
        probability = float(match["probability"])
        if not 0 <= probability <= 100:
            raise ValueError("Probabilitatea imaginii similare trebuie să fie între 0 și 100.")
    confidence = float(result["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("'confidence' trebuie să fie între 0 și 1.")
    class_name = result["class_name"]
    return {
        "wafer_id": result.get("wafer_id", wafer_id),
        "class_name": class_name,
        "is_defect": bool(
            result.get(
                "is_defect",
                class_name.casefold() not in {"normal", "none", "no defect", "fără defect detectat"},
            )
        ),
        "confidence": confidence,
        "explanation": result.get("explanation", ""),
        "action": result.get("action", ""),
        "sources": result.get("sources", []),
        "similar_images": sorted(
            matches, key=lambda match: float(match["probability"]), reverse=True
        )[:5],
    }


st.sidebar.markdown('<div class="eyebrow">AMD HACKATHON · 2026</div>', unsafe_allow_html=True)
st.sidebar.markdown("## WaferSight")
st.sidebar.caption("AI-assisted wafer inspection")
st.sidebar.divider()
mode = st.sidebar.radio("Sursă date", ["Demo · mock local", "API backend"])
api_url = st.sidebar.text_input(
    "Adresa API",
    value=os.getenv("WAFER_API_URL", "http://localhost:8000"),
    disabled=mode.startswith("Demo"),
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Lot activ**")
st.sidebar.caption("LOT-AMD-0248 · Linia 03")
st.sidebar.caption("Camera CAM-07 · Unealta ETCH-02")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Date wafer sintetice · fără API"
    if mode.startswith("Demo")
    else f"API: {api_url}"
)

st.markdown('<div class="eyebrow">INSPECȚIE DE PROCES / DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Wafer intelligence, la prima vedere.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detectarea defectelor, compararea cu wafere similare și sprijin pentru decizia tehnicianului.</div>',
    unsafe_allow_html=True,
)

selected = st.selectbox("Alege un wafer demonstrativ", list(SAMPLES))
wafer_id, pattern = SAMPLES[selected]
wafer = make_wafer(pattern, seed=int(wafer_id[-3:]))

if "analysis" not in st.session_state:
    st.session_state.analysis = None
analyze_clicked = st.button("▶  Analizează wafer", type="primary")
if analyze_clicked:
    st.session_state.analysis = None
    try:
        st.session_state.analysis = (
            analyze_with_api(api_url, wafer, wafer_id)
            if mode.startswith("API")
            else mock_result(wafer_id, pattern)
        )
    except requests.RequestException as exc:
        st.error(f"Nu s-a putut contacta API-ul: {exc}")
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"Răspunsul API este invalid: {exc}")

result = st.session_state.analysis
if result is not None and result["wafer_id"] == wafer_id:
    st.markdown("---")
    defect_count = int(np.count_nonzero(wafer == 2))
    active_dies = int(np.count_nonzero(wafer))
    defect_rate = defect_count / active_dies if active_dies else 0.0
    overview = st.columns(4)
    with overview[0]:
        st.metric("Wafer analizat", wafer_id)
    with overview[1]:
        st.metric("Încredere", f"{result['confidence']:.1%}")
    with overview[2]:
        st.metric("Die-uri defecte", f"{defect_count:,}")
    with overview[3]:
        st.metric("Rată defecte", f"{defect_rate:.1%}")

    preview, classification = st.columns([1.05, 1], gap="large")
    with preview:
        st.markdown('<div class="section-label">Hartă wafer</div>', unsafe_allow_html=True)
        st.caption(f"{wafer_id} · 64 × 64 die-uri")
        show_wafer(wafer)
        st.caption("🟦 Die funcțional  ·  🟥 Die defect  ·  ⬛ În afara wafer-ului")
    with classification:
        badge = (
            '<span class="status-pill status-warn">DEFECT DETECTAT</span>'
            if result["is_defect"]
            else '<span class="status-pill">ANALIZĂ COMPLETĂ</span>'
        )
        st.markdown(badge, unsafe_allow_html=True)
        st.subheader(result["class_name"])

        with st.container(border=True):
            st.markdown("**Explicația analizei**")
            st.write(
                result["explanation"]
                or "Nu este disponibilă o explicație pentru această predicție."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Acțiune recomandată</div>', unsafe_allow_html=True)
        st.info(result["action"] or "Nu a fost furnizată o recomandare.", icon="🛠️")

        if result["sources"]:
            st.markdown('<div class="section-label">Baza analizei</div>', unsafe_allow_html=True)
            for source in result["sources"]:
                st.caption(f"· {source}")

    matches = result["similar_images"] if result["is_defect"] else []
    if matches:
        st.markdown("---")
        st.markdown("### Wafere similare")
        st.caption("Primele cinci potriviri din setul de referință, ordonate după probabilitatea estimată.")
        columns = st.columns(5, gap="small")
        for column, match in zip(columns, matches):
            with column:
                with st.container(border=True):
                    show_wafer(np.asarray(match["wafer"]), compact=True)
                    st.progress(
                        float(match["probability"]) / 100,
                        text=f"{float(match['probability']):.1f}% probabilitate",
                    )
                    st.markdown(f"**{match.get('wafer_id', 'Imagine similară')}**")
    elif result["is_defect"]:
        st.info("Nu au fost returnate imagini de referință pentru acest defect.")
    else:
        st.success("Nu a fost detectat niciun defect; nu sunt necesare imagini de comparație.")
    st.markdown("---")
    st.markdown("### Context lot")
    st.caption("LOT-AMD-0248 · Linia 03 · Cameră CAM-07 · Unealtă ETCH-02")
    st.caption("Metadate demonstrative până la conectarea backend-ului.")
elif result is None:
    st.markdown("---")
    st.markdown(
        '<div class="muted">Alege un wafer și pornește analiza pentru a vedea '
        'predicția, statisticile și imaginile de referință.</div>',
        unsafe_allow_html=True,
    )
