# workflow/steps/inference.py
"""
Inference (SKLearn container) for SageMaker Inference Pipeline.

Role of this container:
- Input : application/json  (user request)
- Output: text/csv          (numeric row(s) for the next XGBoost container)

Important:
- This module must NOT require xgboost.
- featurizer.joblib is saved by preprocess.py as a *payload dict*:
    {
      "ct": sklearn.compose.ColumnTransformer,
      "feature_names": [...],
      "ordinal_map": {...}
    }
"""

import io
import json
import os
from typing import Any, Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir: str) -> Dict[str, Any]:
    fz_path = os.path.join(model_dir, "featurizer.joblib")
    if not os.path.exists(fz_path):
        raise FileNotFoundError(f"featurizer.joblib not found: {fz_path}")

    payload = joblib.load(fz_path)
    if not isinstance(payload, dict) or "ct" not in payload:
        raise ValueError(
            "featurizer.joblib must be a dict payload with key 'ct'. "
            "Check preprocess.py saving logic."
        )

    ct = payload["ct"]
    feature_names = payload.get("feature_names")

    tt_path = os.path.join(model_dir, "target_transform.json")
    target_transform = None
    if os.path.exists(tt_path):
        with open(tt_path, "r", encoding="utf-8") as f:
            target_transform = json.load(f)

    return {"ct": ct, "feature_names": feature_names, "target_transform": target_transform}


def input_fn(request_body: bytes, content_type: str) -> Any:
    if content_type not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError(f"Unsupported content_type: {content_type}. Expected application/json")

    if isinstance(request_body, (bytes, bytearray)):
        payload = json.loads(request_body.decode("utf-8"))
    else:
        payload = json.loads(request_body)

    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        records = [payload]

    return pd.DataFrame.from_records(records)


def _validate_required_columns(df: pd.DataFrame, ct) -> None:
    required: List[str] = []
    if hasattr(ct, "feature_names_in_"):
        required = list(getattr(ct, "feature_names_in_", []))

    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing required feature columns for featurizer. "
                f"missing={missing} available={list(df.columns)}"
            )


def predict_fn(input_data: pd.DataFrame, model: Dict[str, Any]) -> str:
    ct = model["ct"]
    _validate_required_columns(input_data, ct)

    X = ct.transform(input_data)

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    out = io.StringIO()
    for row in X:
        out.write(",".join(str(float(v)) for v in row))
        out.write("\n")
    return out.getvalue()


def output_fn(prediction: str, accept: str) -> Tuple[bytes, str]:
    return prediction.encode("utf-8"), "text/csv"
