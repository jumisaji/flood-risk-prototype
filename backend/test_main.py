"""
Automated tests for the Flood Risk Prediction API.
Run from the backend/ folder:  pytest -q

Passes whether the real model is present or the stub fallback is active.
"""

from fastapi.testclient import TestClient

from main import app, _risk_band, _features_from_series

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["features"] == ["level_lag1", "level_lag2", "level_roll7", "level_change3"]


def test_predict_low_risk_recent_levels():
    payload = {"level_lag1": 0.69, "level_lag2": 0.69, "level_roll7": 0.69, "level_change3": 0.03}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["flood_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Moderate", "High"}


def test_predict_high_level_scenario_is_riskier():
    low = client.post("/predict", json={"level_lag1": 0.5, "level_lag2": 0.5, "level_roll7": 0.5, "level_change3": 0.0}).json()
    high = client.post("/predict", json={"level_lag1": 1.6, "level_lag2": 1.5, "level_roll7": 1.4, "level_change3": 0.6}).json()
    assert high["flood_probability"] >= low["flood_probability"]


def test_predict_series_derives_features():
    r = client.post("/predict_series", json={"levels": [0.72, 0.70, 0.67, 0.65, 0.69, 0.69, 0.73]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["features"]) == {"level_lag1", "level_lag2", "level_roll7", "level_change3"}
    assert body["features"]["level_lag1"] == 0.73


def test_series_too_short_rejected():
    r = client.post("/predict_series", json={"levels": [0.7, 0.7]})
    assert r.status_code == 422


def test_feature_helper_math():
    f = _features_from_series([0.6, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72])
    assert f.level_lag1 == 0.72
    assert f.level_lag2 == 0.70
    assert round(f.level_change3, 2) == round(0.72 - 0.66, 2)


def test_risk_band_thresholds():
    assert _risk_band(0.10) == "Low"
    assert _risk_band(0.50) == "Moderate"
    assert _risk_band(0.90) == "High"
