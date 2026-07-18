"""
Flood Risk Prediction API
=========================
FastAPI service for the flood-risk prototype (River Murray catchment).

Serves the team's trained Logistic Regression model
(notebooks/FloodRiskPrediction_LogisticRegression.ipynb ->
logistic_regression_real.joblib). The model predicts next-day high-river-level
("flood") risk for Murray Bridge from four river-level features:

    level_lag1     water level 1 day ago (m)
    level_lag2     water level 2 days ago (m)
    level_roll7    mean of the 7 most recent prior days (m)
    level_change3  level_lag1 minus the level 4 days ago (m)

Two ways to call it:
  POST /predict         -> pass the 4 features directly (matches the notebook)
  POST /predict_series  -> pass recent daily levels; the API derives the features

If the model file is missing, a transparent fallback keeps the endpoint working
so the service still deploys and demos end to end.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import joblib
    import pandas as pd
except Exception:  # pragma: no cover
    joblib = None
    pd = None

# Feature order MUST match the notebook's training order.
FEATURE_ORDER = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]

# Risk threshold used to label the training data (0.80 quantile of Murray Bridge
# water level). Reported for context; scoring uses the model's probability.
TRAIN_RISK_THRESHOLD_M = 0.806

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "logistic_regression_real.joblib"))

_model = None
_model_source = "stub"


def _load_model() -> None:
    global _model, _model_source
    if joblib is not None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            _model_source = f"joblib:{MODEL_PATH.name}"
        except Exception as exc:  # pragma: no cover
            _model = None
            _model_source = f"stub (model load failed: {exc})"
    else:
        _model = None
        _model_source = "stub (no model file found)"


def _stub_probability(f: "FloodFeatures") -> float:
    """Transparent fallback: risk rises as recent level approaches the threshold."""
    level = max(f.level_lag1, f.level_roll7)
    base = level / (TRAIN_RISK_THRESHOLD_M * 1.25)
    trend = max(0.0, f.level_change3) * 0.8
    return max(0.0, min(1.0, base + trend))


def _predict_probability(f: "FloodFeatures") -> float:
    if _model is not None and pd is not None:
        row = pd.DataFrame([[getattr(f, name) for name in FEATURE_ORDER]], columns=FEATURE_ORDER)
        return float(_model.predict_proba(row)[0][1])
    return _stub_probability(f)


def _risk_band(p: float) -> Literal["Low", "Moderate", "High"]:
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Moderate"
    return "Low"


def _features_from_series(levels: List[float]) -> "FloodFeatures":
    """Derive the 4 model features from recent daily levels (most recent last).

    Predicts the NEXT day, so the last provided value is treated as yesterday.
    """
    if len(levels) < 4:
        raise HTTPException(
            status_code=422,
            detail="Provide at least 4 recent daily levels (7+ preferred for level_roll7).",
        )
    lag1 = levels[-1]
    lag2 = levels[-2]
    window = levels[-7:]
    roll7 = sum(window) / len(window)
    change3 = levels[-1] - levels[-4]
    return FloodFeatures(
        level_lag1=lag1, level_lag2=lag2, level_roll7=roll7, level_change3=change3
    )


class FloodFeatures(BaseModel):
    level_lag1: float = Field(..., description="Water level 1 day ago (m)")
    level_lag2: float = Field(..., description="Water level 2 days ago (m)")
    level_roll7: float = Field(..., description="Mean of 7 most recent prior days (m)")
    level_change3: float = Field(..., description="level_lag1 minus level 4 days ago (m)")


class SeriesRequest(BaseModel):
    levels: List[float] = Field(
        ..., min_length=4,
        description="Recent daily water levels in metres, oldest first, most recent last.",
        examples=[[0.72, 0.70, 0.67, 0.65, 0.69, 0.69, 0.73]],
    )
    station_id: str = Field("A4261162", description="Gauging station id (Murray Bridge default).")


class PredictionResponse(BaseModel):
    flood_probability: float
    risk_band: Literal["Low", "Moderate", "High"]
    features: dict
    model_source: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Flood Risk Prediction API",
    description="Next-day river-flood risk for the River Murray catchment (ITA602 prototype).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_source": _model_source,
        "features": FEATURE_ORDER,
        "train_risk_threshold_m": TRAIN_RISK_THRESHOLD_M,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: FloodFeatures) -> PredictionResponse:
    p = _predict_probability(features)
    return PredictionResponse(
        flood_probability=round(p, 4),
        risk_band=_risk_band(p),
        features=features.model_dump(),
        model_source=_model_source,
    )


@app.post("/predict_series", response_model=PredictionResponse)
def predict_series(req: SeriesRequest) -> PredictionResponse:
    features = _features_from_series(req.levels)
    p = _predict_probability(features)
    return PredictionResponse(
        flood_probability=round(p, 4),
        risk_band=_risk_band(p),
        features={k: round(v, 4) for k, v in features.model_dump().items()},
        model_source=_model_source,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
