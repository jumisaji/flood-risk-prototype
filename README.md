# flood-risk-prototype

AI-based flood risk prediction and early warning prototype for the River Murray, South Australia.

ITA602 Industry Project. Aide Alejandra Gomez Navarro, Manuela Munoz Ramirez,
Julieth Milena Sanchez Jimenez, Furku Ghale.

**Live**

| | |
|---|---|
| Dashboard (React, delivered) | https://flood-risk-prototype.vercel.app |
| API (FastAPI on Render) | https://flood-risk-prototype-week10.onrender.com |
| Dashboard (Streamlit, earlier client) | https://flood-risk-prototype-week11.streamlit.app |

The Render instance is on the free tier and sleeps after about 15 minutes idle. Run
`python backend/warmup.py` a few minutes before a demo so the first prediction is instant.

## What it does

The system answers one question: **will tomorrow be a high river level day at Murray Bridge?**
A day counts as high risk when the level reaches the 0.80 quantile of the historical record,
which is 0.806 m. About 20% of days meet that condition, so the problem is imbalanced.

A prediction becomes an alert only when a human authorises it, and every authorised alert is
written to a tamper-evident log. The model advises; a person decides.

This is not a competitor to the Bureau of Meteorology's official warnings. It is a
council-level decision support layer around a forecast.

## Architecture

```
data/               river level and rainfall CSVs, LGA boundaries
notebooks/          shared training base (common.py), model notebooks, evaluation, SHAP
backend/            FastAPI service, model registry, alert audit chain, tests, Dockerfile
frontend/           React + Vite dashboard (deployed to Vercel)
dashboard/          Streamlit dashboard, consumes the same API
docs/               API contract, metrics, models and explainability report
shap/               precomputed SHAP importances
db/                 Supabase schema for the alert audit log
.github/workflows/  CI tests and the Render keep-alive job
```

Both dashboards talk to the same API through one versioned contract
(`docs/API_CONTRACT.md`), so they tell the same story.

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | service status and which models are loaded |
| `GET /models` | model registry with a five metric summary each |
| `POST /predict` | score the four features directly |
| `POST /predict_series` | derive the features from a level series, then score |
| `POST /alerts` | authorised alert, hash chained and emailed |
| `GET /alerts/verify` | recompute the whole chain and report the first broken link |

23 automated tests in `backend/test_main.py` cover the endpoints, the ensemble soft vote and
the audit chain, including tampering and removed link detection. They pass on a fresh
checkout with no model files present.


### Architecture Diagram

The architecture diagram from the project report will be added here.

![Flood Risk Prediction System Architecture](docs/architecture-diagram.png)

## Repository Structure

    flood-risk-prototype/
    ├── data/          # Datasets and processed data
    ├── notebooks/     # Data analysis and machine learning experiments
    ├── backend/       # FastAPI backend and model integration
    ├── dashboard/     # User interface and visualisation dashboard
    ├── docs/          # Project documentation and architecture diagrams
    ├── .gitignore
    └── README.md

## Technologies

- Python
- Machine Learning / Artificial Intelligence
- FastAPI
- React
- Docker
- Cloud technologies

## Data

| | |
|---|---|
| Modelled station | Murray Bridge |
| Also supported | Morgan, Mannum (loaded, no trained model yet) |
| Sources | Bureau of Meteorology, Water Data SA, SA Department for Environment and Water |
| Licence | CC BY 4.0, © Commonwealth of Australia |

Rainfall records are in `data/` but the deployed model does not use them. The four features
are all derived from river level. Rainfall is reserved for future work, and the reason is in
the limitations below.

### Features

All four look only at days before the prediction day, which is what keeps the evaluation
honest.

| Feature | Meaning |
|---|---|
| `level_lag1` | water level 1 day ago |
| `level_lag2` | water level 2 days ago |
| `level_roll7` | mean level over the 7 previous days |
| `level_change3` | level 1 day ago minus level 4 days ago (the trend) |

Every model imports `notebooks/common.py`, so they all train on the same data, the same
label and the same split. The split is chronological, training on the earlier period and
testing on the most recent 954 days. It is never shuffled.

An earlier shuffled split produced a Random Forest F1 of 0.932. With lagged features a
random split lets the model see days that sit between two training days, so that result was
leakage. It was discarded, and the chronological split has been in `common.py` ever since.

## Models

- **Logistic Regression**: linear baseline, standardised inputs, balanced class weights.
- **Random Forest**: 400 trees, depth capped at 10, chosen with an expanding window
  backtest inside the training period only.
- **XGBoost**: gradient boosting, tuned with Optuna.
- **LSTM**: sequence model over the last 14 daily levels, decision threshold 0.72.
- **Ensemble (deployed default)**: equal weight soft vote over the three tabular models.

### Results

Chronological test fold, 954 days, positive class = high risk day.

| Model | F1 | MCC | RMSE | Brier | NSE |
|---|---|---|---|---|---|
| Persistence baseline | 0.796 | 0.732 | – | – | – |
| Logistic Regression | 0.800 | 0.741 | 0.300 | 0.090 | 0.505 |
| Random Forest | 0.799 | 0.734 | 0.264 | **0.070** | **0.618** |
| XGBoost | 0.802 | 0.738 | 0.276 | 0.076 | 0.582 |
| LSTM | 0.804 | 0.741 | 0.338 | 0.114 | 0.371 |
| **Ensemble (deployed)** | **0.806** | **0.743** | 0.271 | 0.074 | 0.596 |

The four tabular rows are generated by `notebooks/eval_metrics.py` into `docs/metrics.json`,
so the table, the `/models` endpoint and the report cannot drift apart.

The LSTM row is different and we would rather say so than have it noticed. It comes from
`notebooks/FloodRiskPrediction_LSTM.ipynb`, because the model needs a 14 day window and
drops more early rows: its fold is 953 days against 954. Comparable in period and size, but
not byte identical. It is therefore **not in `docs/metrics.json` and not returned by
`GET /models`**, which reports it without metrics. Folding it into the shared script is
listed in future work.

**Every model sits within about one F1 point of the persistence baseline**, which simply
predicts that tomorrow repeats today and scores 0.796. River level is highly autocorrelated,
so yesterday's level is already an excellent predictor. The machine learning models add a
small margin, not a dramatic one.

Within that narrow band the ordering still means something. The ensemble gives the best F1
and MCC, which is why it is served by default. Random Forest gives the best probability
quality, which matters because the dashboard shows a probability rather than a yes or no.
The LSTM matches on F1 but its probabilities are the least calibrated.

## Explainability

SHAP importances are precomputed by `notebooks/precompute_shap.py` and stored in `shap/`,
because computing SHAP over the test fold on every page load is far too slow for an
interactive screen. Full analysis is in `docs/models_and_explainability.md`.

Mean absolute impact, share of total:

| Feature | Random Forest | XGBoost |
|---|---|---|
| `level_lag1` (yesterday) | 53% | 58% |
| `level_roll7` (7-day mean) | 30% | 26% |
| `level_lag2` (2 days ago) | 10% | 10% |
| `level_change3` (trend) | 6% | 6% |

Both tree models agree that yesterday's level dominates and the 7-day average is second.
Together they carry over 80% of the attribution. That is physically sensible for a large
regulated river, and it also explains the result above: a model leaning this heavily on
yesterday's level cannot pull far ahead of a baseline that *is* yesterday's level.

For the LSTM, importance rises steadily toward the present. Days -1, -2 and -3 carry about
47% of the attribution between them. The model learned to weight recent history most heavily
without being told to.

## Governance

- No alert is dispatched automatically. A human operator confirms it in two steps.
- Every alert is a link in a hash chained, append only log in Supabase. A verify endpoint
  detects an edited value or a removed link, and both cases are covered by tests.
- Alert records are retained in line with the disposal schedules approved under the
  State Records Act 1997 (SA).
- Aligned with ISO/IEC 23894:2023 and Australia's AI Ethics Principles.

## Running it locally

```bash
# backend
pip install -r backend/requirements.txt
python -m uvicorn main:app --app-dir backend --port 8077

# frontend
cd frontend
npm install
cp .env.example .env.local     # set VITE_API_URL=http://127.0.0.1:8077
npm run dev

# regenerate the metrics table
python notebooks/eval_metrics.py --write

# tests
cd backend && pytest -q
```

`/alerts` needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` on the backend. Without them it
returns 503 and the interface says "Authorised locally", which is the honest state rather
than a silent failure.

## Changes from the project plan

| Planned | Delivered | Why |
|---|---|---|
| Five data sources behind an Airflow pipeline | Two river level sources on a scheduled batch | A robust orchestration layer did not fit the twelve week cycle |
| Benchmark against the WAPABA conceptual model | Persistence baseline | WAPABA needs hydrological calibration well outside the prototype's scope. The persistence baseline turned out to be a harder bar than expected |
| PostgreSQL and PostGIS schema for catchments, scores and alerts | Static GeoJSON for boundaries, hosted Postgres for the alert log | Three council areas do not need a spatial database |

Each change was raised at a sprint review and recorded in the backlog.

## Limitations

- Only Murray Bridge has a trained model. The other council areas on the map are context,
  and the interface greys them out rather than implying coverage that does not exist.
- The improvement over the persistence baseline is small, as reported above.
- Feature importance is global. It describes the model overall, not the specific day on
  screen. Per prediction SHAP would need live computation, which was traded away for a
  responsive interface.
- The LSTM's probabilities are poorly calibrated and would need calibration before any
  operational use.
- Ingestion is scheduled batch, not real time.

## Future work

Richer features are the clearest next step: rainfall, soil moisture and upstream gauges are
what would let a model beat persistence convincingly. Alongside that, extend
`eval_metrics.py` to score the LSTM in the same run as the tabular models, so its row lands
in `docs/metrics.json` and on `/models` like the rest. Then trained models for the remaining
council areas, SMS dispatch through a funded gateway, and a field pilot with a council or
SES unit to validate the alert workflow in practice.

## References

Bureau of Meteorology. (2026). *Climate data online*. Australian Government.
https://www.bom.gov.au/climate/data/

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*,
321–357. https://doi.org/10.1613/jair.953

Department of Industry, Science and Resources. (2025). *Australia's AI Ethics Principles*.
Australian Government.

International Organization for Standardization. (2023). *ISO/IEC 23894:2023 Information
technology, artificial intelligence, AI risk management*.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
*Advances in Neural Information Processing Systems, 30*, 4765–4774.

South Australian Department for Environment and Water. (2026). *River Murray information*.
Government of South Australia. https://www.environment.sa.gov.au/

## Licence

MIT for the code. Data remains subject to its original licences. This is an academic
prototype for educational purposes. No proprietary or confidential data was used.

