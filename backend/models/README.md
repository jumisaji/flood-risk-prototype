# Models

`logistic_regression_real.joblib` — the trained Logistic Regression model exported
from `notebooks/FloodRiskPrediction_LogisticRegression.ipynb`.

Features (order matters): `level_lag1`, `level_lag2`, `level_roll7`, `level_change3`.
Trained with scikit-learn 1.6.1 (pinned in `backend/requirements.txt`).

To update the model, re-run the notebook's export cell and replace this file.

Note: `_writetest.txt` in this folder is a leftover and can be deleted.
