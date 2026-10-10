"""
Rulare:
    pip install -r requirements.txt
    streamlit run app.py
"""
import numpy as np
import pydicom
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PinkLens",
    page_icon="🎀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { color: #A8506B; }
    .card {
        background: #FFFFFF;
        border: 1px solid #F6CAD7;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .result-malign { background:#FADADD; border:1px solid #E58FA8; }
    .result-benign { background:#E9F5EC; border:1px solid #A9D5B5; }
    .result-title  { font-size: 1.5rem; font-weight: 700; margin: 0; }
    .muted { color: #8E6F7A; font-size: 0.9rem; }
    [data-testid="stSidebar"][aria-expanded="true"] { min-width: 21rem; max-width: 21rem; }
    [data-testid="stSidebar"][aria-expanded="false"] .sidebar-footer { display: none; }
    [data-testid="stSidebar"] h2 { padding-top: 0; margin-top: 0; }
    .sidebar-footer {
        position: fixed;
        bottom: 1rem;
        left: 1.5rem;
        width: 18rem;
        color: #8E6F7A;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .stProgress > div > div > div > div { background-color: #E58FA8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Citire imagini
def _to_float(value, default=None):
    """DICOM poate returna valori multiple (MultiValue) - le luăm pe primul."""
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except Exception:
        return default


def load_dicom(file) -> tuple[np.ndarray, dict]:
    """Returnează (imagine uint8 pentru afișare, metadate)."""
    ds = pydicom.dcmread(file)
    pixels = ds.pixel_array.astype(np.float32)

    # Dacă fișierul are mai multe cadre (ex. tomosinteză), luăm cadrul din mijloc
    if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
        pixels = pixels[pixels.shape[0] // 2]

    slope = _to_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _to_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    pixels = pixels * slope + intercept

    center = _to_float(getattr(ds, "WindowCenter", None))
    width = _to_float(getattr(ds, "WindowWidth", None))
    if center is not None and width:
        lo, hi = center - width / 2, center + width / 2
    else:
        lo, hi = np.percentile(pixels, (0.5, 99.5))
    pixels = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    img = (pixels * 255).astype(np.uint8)

    meta = {
        "Lateralitate": str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "-"))),
        "Proiecție": str(getattr(ds, "ViewPosition", "-")),
        "Modalitate": str(getattr(ds, "Modality", "-")),
        "Dimensiune": f"{img.shape[1]} x {img.shape[0]} px",
        "Vârstă pacient": str(getattr(ds, "PatientAge", "-")),
        "Data studiului": str(getattr(ds, "StudyDate", "-")),
    }
    
    # Păstrăm doar câmpurile care chiar există în fișier
    meta = {k: v for k, v in meta.items() if v.strip() not in ("", "-", "None")}
    return img, meta


# Modelul AI
def predict(image: np.ndarray) -> dict:
    """
    Predicția este hard-codată.
    Primește imaginea (numpy uint8, 2D) și returnează un dict cu probabilitățile pe clase.
    """
    prob_malign = 0.87
    return {
        "Malign": prob_malign,
        "Benign": 1.0 - prob_malign,
    }

# Interfata
with st.sidebar:
    st.header("Despre aplicație")
    st.write(
        "Prototip de cercetare care explorează utilizarea inteligenței "
        "artificiale pentru analiza imaginilor mamografice."
    )
    st.divider()
    threshold = st.slider(
        "Prag de decizie pentru „Malign”",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Dacă probabilitatea de malign depășește pragul, cazul este marcat ca malign.",
    )
    st.markdown(
        "<div class='sidebar-footer'>Instrument de cercetare. Rezultatul nu "
        "înlocuiește evaluarea unui medic radiolog.</div>",
        unsafe_allow_html=True,
    )

st.title("PinkLens")
st.markdown(
    "<p class='muted'>"
    "Încarcă o imagine mamografică pentru a vizualiza datele acesteia "
    "și a testa clasificarea experimentală realizată de un model AI."
    "</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "",
    type=["dcm", "dicom"],
    help="Format: DICOM (.dcm).",
)

if uploaded is None:
    st.info("Încarcă o mamografie pentru a începe.")
    st.stop()

# Citire fișier
try:
    if uploaded.name.lower().endswith((".dcm", ".dicom")):
        image, meta = load_dicom(uploaded)
except Exception as e:
    st.error(
        f"Fișierul nu a putut fi citit: {e}. Verifică dacă este un fișier DICOM valid."
    )
    st.stop()

col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    st.subheader("Mamografie")
    st.image(image, use_container_width=True, clamp=True)

with col_res:
    st.subheader("Detalii imagine")
    for k, v in meta.items():
        st.markdown(f"**{k}:** {v}")

    analyze = st.button("Analizează", type="primary", use_container_width=True)

    if analyze:
        with st.spinner("Se analizează imaginea..."):
            probs = predict(image)

        p_malign = probs["Malign"]
        is_malign = p_malign >= threshold
        css = "result-malign" if is_malign else "result-benign"
        label = "Suspect de malign" if is_malign else "Probabil benign"

        st.markdown(
            f"""
            <div class='card {css}'>
                <p class='result-title'>{label}</p>
                <p class='muted'>Probabilitate malign: {p_malign:.0%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Probabilități pe clase**")
        for cls, p in probs.items():
            st.write(f"{cls}: {p:.0%}")
            st.progress(float(p))

        if is_malign:
            st.warning(
                "Se recomandă evaluare suplimentară de către un medic radiolog "
                "(ex. biopsie, ecografie, RMN)."
            )
        else:
            st.success("Nu au fost detectate semne suspecte la acest prag de decizie.")