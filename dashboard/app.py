"""
Flood Risk Dashboard (Streamlit)
================================
River Murray flood-risk prototype — serves the team's trained Logistic
Regression model (next-day high-river-level risk for Murray Bridge).

Prediction source (in priority order):
1. The FastAPI backend, if an API URL is set (sidebar or API_URL secret) -> /predict_series
2. The local model file (backend/models/logistic_regression_real.joblib), loaded directly
3. A transparent fallback, so the dashboard still runs as a public link either way

The four model features are derived from a recent daily river-level series:
    level_lag1     most recent level (m)
    level_lag2     level the day before (m)
    level_roll7    mean of the 7 most recent levels (m)
    level_change3  most recent level minus the level 4 days earlier (m)
"""

import os
from pathlib import Path

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

TRAIN_RISK_THRESHOLD_M = 0.806
BAND_COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}
BAND_HEX = {"Low": "#12a150", "Moderate": "#e0982b", "High": "#d64545"}

# Murray Bridge gauging station (A4261162).
STATION = {"name": "Murray Bridge", "id": "A4261162", "lat": -35.12, "lon": 139.27}

# Real recent Murray Bridge daily levels (m) as a sensible default series.
DEFAULT_LEVELS = [0.67, 0.65, 0.65, 0.67, 0.69, 0.69, 0.73]


@st.cache_resource
def load_local_model():
    """Load the trained model from the repo, if available."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "backend" / "models" / "logistic_regression_real.joblib",
        here.parent / "notebooks" / "logistic_regression_real.joblib",
    ]
    for p in candidates:
        if p.exists():
            try:
                import joblib
                return joblib.load(p), f"local:{p.name}"
            except Exception:
                pass
    return None, "fallback"


def features_from_levels(levels):
    lag1 = levels[-1]
    lag2 = levels[-2]
    window = levels[-7:]
    roll7 = sum(window) / len(window)
    change3 = levels[-1] - levels[-4]
    return {"level_lag1": lag1, "level_lag2": lag2, "level_roll7": roll7, "level_change3": change3}


def band_of(p):
    return "High" if p >= 0.66 else "Moderate" if p >= 0.33 else "Low"


def fallback_probability(f):
    level = max(f["level_lag1"], f["level_roll7"])
    base = level / (TRAIN_RISK_THRESHOLD_M * 1.25)
    trend = max(0.0, f["level_change3"]) * 0.8
    return max(0.0, min(1.0, base + trend))


def predict(api_url, levels, model, model_src):
    feats = features_from_levels(levels)
    if api_url:
        try:
            r = requests.post(api_url.rstrip("/") + "/predict_series",
                              json={"levels": levels, "station_id": STATION["id"]}, timeout=30)
            r.raise_for_status()
            b = r.json()
            return b["flood_probability"], b["risk_band"], feats, b.get("model_source", "api")
        except Exception as exc:
            st.warning(f"API unreachable, using local model/fallback ({exc}).")
    if model is not None:
        import pandas as pd
        cols = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]
        row = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
        p = float(model.predict_proba(row)[0][1])
        return p, band_of(p), feats, model_src
    p = fallback_probability(feats)
    return p, band_of(p), feats, "fallback"


# --------------------------------------------------------------------------
st.set_page_config(page_title="River Murray Flood Risk", page_icon="🌊", layout="wide")
st.title("🌊 River Murray Flood Risk — Prototype")
st.caption(
    "Next-day flood-risk (high river level) for Murray Bridge, from the trained "
    "Logistic Regression model. Alerts require human authorisation — this view is advisory."
)

model, model_src = load_local_model()
try:
    default_api = st.secrets.get("API_URL", os.getenv("API_URL", ""))
except Exception:
    default_api = os.getenv("API_URL", "")

with st.sidebar:
    st.header("Recent river levels (m)")
    st.caption("Most recent 7 daily readings, oldest first. Adjust to simulate a rising river.")
    levels = []
    for i, d in enumerate(DEFAULT_LEVELS):
        levels.append(st.slider(f"Day -{len(DEFAULT_LEVELS)-i}", 0.0, 3.0, float(d), 0.01))
    shift = st.slider("Shift whole series (m)", -0.5, 2.0, 0.0, 0.05,
                      help="Raise or lower all readings to test scenarios.")
    levels = [round(x + shift, 3) for x in levels]
    st.divider()
    api_url = st.text_input("API URL (optional)", value=default_api,
                            placeholder="https://flood-risk-api.onrender.com")
    st.caption("Blank = use the local model bundled in the repo.")

prob, band, feats, source = predict(api_url, levels, model, model_src)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Prediction")
    st.metric("Flood probability", f"{prob*100:.0f}%")
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;color:white;"
        f"background:{BAND_HEX[band]};font-weight:600;display:inline-block'>Risk band: {band}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Source: {source}")
    if band == "High":
        st.error("⚠️ High risk — would require operator review before any alert is sent.")
    st.markdown("**Model features (derived)**")
    st.table({k: [round(v, 3)] for k, v in feats.items()})
    st.caption(f"Training risk threshold: {TRAIN_RISK_THRESHOLD_M} m (0.80 quantile).")

with col2:
    st.subheader("Station")
    fmap = folium.Map(location=[STATION["lat"], STATION["lon"]], zoom_start=9, tiles="CartoDB positron")
    folium.CircleMarker(
        location=[STATION["lat"], STATION["lon"]],
        radius=12, color=BAND_COLOR[band], fill=True, fill_color=BAND_COLOR[band], fill_opacity=0.9,
        popup=f"{STATION['name']}: {band} ({prob*100:.0f}%)", tooltip=STATION["name"],
    ).add_to(fmap)
    st_folium(fmap, height=430, use_container_width=True)

st.divider()
st.caption("Prototype for academic assessment (ITA602). Not for operational flood-warning use.")
