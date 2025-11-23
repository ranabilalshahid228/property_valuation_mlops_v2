# model_predictor.py
# Robust single & batch prediction utilities for the Streamlit app.

from __future__ import annotations
from typing import Union, Dict, Any

import os
import time
import joblib
import numpy as np
import pandas as pd

# SHAP is optional; we keep it best-effort
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from data_processor import validate_and_clean
from config import CONFIG


# --------------------------- helpers ---------------------------

def _load_model():
    """Load the currently active model pipeline (preprocessor + regressor)."""
    path = "models/active_model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _numeric_confidence(df: pd.DataFrame) -> float:
    """
    Lightweight confidence proxy based on dispersion of numeric inputs.
    Uses only numeric columns to avoid str+int errors.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return 0.90  # safe default
    # replace infs before std calc
    arr = num_df.replace([np.inf, -np.inf], np.nan).values.astype(float)
    std_val = float(np.nanstd(arr))
    # map dispersion to [0.85, 0.99]
    return max(0.85, min(0.99, 1.0 - (std_val / 1e5)))


def _log_prediction(row: Dict[str, Any]) -> None:
    os.makedirs("logs", exist_ok=True)
    path = "logs/recent_predictions.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _safe_shap_explanation(model, df: pd.DataFrame):
    """
    Try to compute a small per-prediction explanation.
    Works only if SHAP is available AND we can access the model step.
    Returns dict[str, float] | None
    """
    if not HAS_SHAP:
        return None
    try:
        # We expect a sklearn Pipeline with steps "pre" and "model"
        if not hasattr(model, "named_steps"):
            return None
        pre = model.named_steps.get("pre", None)
        est = model.named_steps.get("model", None)
        if pre is None or est is None:
            return None

        # Transform features, then explain the estimator output on that single row
        X_proc = pre.transform(df)
        explainer = shap.Explainer(est)
        sv = explainer(X_proc)
        vals = np.array(sv.values[0]).ravel()
        names = getattr(explainer, "feature_names", None)
        if names is None or len(names) != len(vals):
            names = [f"f{i}" for i in range(len(vals))]

        # Top 5 absolute impacts
        order = np.argsort(np.abs(vals))[-5:][::-1]
        return {names[i]: float(vals[i]) for i in order}
    except Exception:
        return None


# --------------------------- API ---------------------------

def predict_single(payload: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Accepts a dict (single property) or a 1-row DataFrame.
    Returns: {"prediction": int, "confidence": float, "latency_ms": int, "explanation": dict|None}
    """
    # Convert input to DataFrame
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, pd.DataFrame):
        if len(payload) != 1:
            # If they pass multiple rows, just take the first for "single"
            df = payload.iloc[[0]].copy()
        else:
            df = payload.copy()
    else:
        raise TypeError("predict_single expects a dict or a pandas DataFrame")

    # Clean + engineer; ensures required columns exist and basic types are numeric
    df, _ = validate_and_clean(df)

    # Replace infinities that can arise from ratios; imputer in pipeline will handle NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    model = _load_model()
    if model is None:
        return {"error": "Model not trained yet."}

    start = time.time()
    y_hat = float(model.predict(df)[0])
    latency_ms = int((time.time() - start) * 1000)

    # Confidence proxy uses numeric columns only
    conf = _numeric_confidence(df)

    # Optional explanation (best-effort)
    explanation = _safe_shap_explanation(model, df)

    # Log the request
    log_row = {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "property_id": payload.get("property_id", f"PROP_{np.random.randint(1000, 9999)}") if isinstance(payload, dict) else "PROP_BATCH",
        "predicted_value": int(round(y_hat)),
        "confidence": round(conf, 3),
        "location": df.get("location", pd.Series(["Unknown"])).iloc[0] if "location" in df.columns else "Unknown",
        "latency_ms": latency_ms,
    }
    _log_prediction(log_row)

    return {
        "prediction": int(round(y_hat)),
        "confidence": round(conf, 3),
        "latency_ms": latency_ms,
        "explanation": explanation,
    }


def predict_batch(features: pd.DataFrame) -> pd.DataFrame:
    """
    Batch predictions for N rows DataFrame of features.
    Returns original columns + 'predicted_value'.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("predict_batch expects a pandas DataFrame")

    df, _ = validate_and_clean(features.copy())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    model = _load_model()
    if model is None:
        raise RuntimeError("Model not trained yet.")

    preds = model.predict(df).astype(int)

    out = features.copy()
    out["predicted_value"] = preds
    return out
