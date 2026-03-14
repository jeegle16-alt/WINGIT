import os
import io
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import Any, Optional

# optional: sparse 안전처리
try:
    import scipy.sparse as sp
except Exception:
    sp = None

"""
workflow/steps/inference.py

SKLearn inference script used as the *preprocess container* in a multi-container
SageMaker endpoint:  [SKLearn (this script) -> XGBoost built-in]

Fix:
1) Missing required columns -> derive from minimal user inputs when possible,
   and create any remaining required columns with safe defaults.
2) Unsupported content type: text/csv -> accept BOTH application/json and text/csv (CSV must have header).

This container returns:
- numeric feature matrix CSV (no header) for downstream XGBoost container.
"""

HOLIDAY_MONTHS = [1, 3, 4, 5, 6, 8, 10, 11]  # must match preprocess.py

def _parse_dt(x: Any) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return None

def _parse_total_time_to_min(x: Any) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)

    s = str(x).lower().strip()
    if not s:
        return 0

    try:
        h, m = 0, 0
        if "h" in s:
            parts = s.split("h")
            hh = parts[0].strip()
            if hh:
                h = int(float(hh))
            if len(parts) > 1 and "m" in parts[1]:
                mm = parts[1].replace("m", "").strip()
                if mm:
                    m = int(float(mm))
            return h * 60 + m

        if "m" in s:
            mm = s.replace("m", "").strip()
            return int(float(mm)) if mm else 0

        if ":" in s:
            hh, mm = s.split(":", 1)
            return int(hh) * 60 + int(mm)

        return int(float(s))
    except Exception:
        return 0

def _get_time_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "dawn"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"

def _get_days_until_bucket(days: int) -> str:
    if days < 7:
        return "very_close"
    if days < 14:
        return "close"
    if days < 30:
        return "medium"
    return "far"

def _get_duration_bucket(minutes: int) -> str:
    if minutes < 120:
        return "short"
    if minutes < 360:
        return "medium"
    return "long"

def _route_hash_to_int(route: str) -> int:
    return int(hashlib.md5(route.encode("utf-8")).hexdigest()[:8], 16)

def _safe_div_scalar(a: Any, b: Any, default: float = 1.0) -> float:
    try:
        a2 = float(a)
        b2 = float(b)
        if b2 == 0.0 or np.isnan(a2) or np.isnan(b2):
            return float(default)
        out = a2 / b2
        if np.isfinite(out):
            return float(out)
        return float(default)
    except Exception:
        return float(default)

_HISTORY_BASE_COLS = [
    "prev_fare",
    "min_fare_last_7d", "mean_fare_last_7d",
    "min_fare_last_14d", "mean_fare_last_14d",
    "min_fare_last_30d", "mean_fare_last_30d",
]
_HISTORY_RATIO_COLS = [
    "prev_fare_vs_min_7d", "prev_fare_vs_mean_7d",
    "prev_fare_vs_min_14d", "prev_fare_vs_mean_14d",
    "prev_fare_vs_min_30d", "prev_fare_vs_mean_30d",
]

def _get_route_defaults(route_stats: dict, route_hash: int) -> dict:
    by_route = (route_stats or {}).get("by_route_hash", {}) if isinstance(route_stats, dict) else {}
    fb = (route_stats or {}).get("global_fallback", {}) if isinstance(route_stats, dict) else {}
    r = by_route.get(str(int(route_hash))) if by_route else None
    if isinstance(r, dict):
        return r
    return fb if isinstance(fb, dict) else {}

def _apply_route_stats(df: pd.DataFrame, route_stats: Any) -> pd.DataFrame:
    if route_stats is None or "route_hash" not in df.columns:
        return df

    out = df.copy()
    for c in _HISTORY_BASE_COLS + _HISTORY_RATIO_COLS:
        if c not in out.columns:
            out[c] = np.nan

    for idx in out.index:
        rh = out.at[idx, "route_hash"]
        try:
            rh_int = int(rh)
        except Exception:
            rh_int = None

        defaults = _get_route_defaults(route_stats, rh_int) if rh_int is not None else {}
        if not isinstance(defaults, dict):
            defaults = {}

        for c in _HISTORY_BASE_COLS:
            v = out.at[idx, c]
            if pd.isna(v) or float(v) == 0.0:
                dv = defaults.get(c)
                if dv is not None:
                    out.at[idx, c] = dv

        prev = out.at[idx, "prev_fare"]
        for win in ("7d", "14d", "30d"):
            mn = out.at[idx, f"min_fare_last_{win}"]
            me = out.at[idx, f"mean_fare_last_{win}"]

            col_min = f"prev_fare_vs_min_{win}"
            col_mean = f"prev_fare_vs_mean_{win}"

            v1 = out.at[idx, col_min]
            if pd.isna(v1) or float(v1) == 0.0:
                out.at[idx, col_min] = _safe_div_scalar(prev, mn, default=1.0)

            v2 = out.at[idx, col_mean]
            if pd.isna(v2) or float(v2) == 0.0:
                out.at[idx, col_mean] = _safe_div_scalar(prev, me, default=1.0)

    return out

def _derive_features_from_minimal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Source" in out.columns and "origin" not in out.columns:
        out["origin"] = out["Source"]
    if "Destination" in out.columns and "destination" not in out.columns:
        out["destination"] = out["Destination"]

    if "Number Of Stops" in out.columns and "stops_count" not in out.columns:
        out["stops_count"] = out["Number Of Stops"]
    if "number_of_stops" in out.columns and "stops_count" not in out.columns:
        out["stops_count"] = out["number_of_stops"]
    if "stops_count" not in out.columns:
        out["stops_count"] = 0

    if "purchase_datetime" in out.columns:
        crawl_dt = out["purchase_datetime"].apply(_parse_dt)
    elif "Crawl Timestamp" in out.columns:
        crawl_dt = out["Crawl Timestamp"].apply(_parse_dt)
    else:
        crawl_dt = pd.Series([pd.NaT] * len(out), index=out.index)

    if "departure_datetime" in out.columns:
        dep_dt = out["departure_datetime"].apply(_parse_dt)
    elif "departure_date" in out.columns:
        dep_dt = out["departure_date"].apply(_parse_dt)
    elif "Departure Date" in out.columns and "Departure Time" in out.columns:
        dep_dt = (out["Departure Date"].astype(str) + " " + out["Departure Time"].astype(str)).apply(_parse_dt)
    elif "Departure Date" in out.columns:
        dep_dt = out["Departure Date"].apply(_parse_dt)
    else:
        dep_dt = pd.Series([pd.NaT] * len(out), index=out.index)

    if "purchase_day_of_week" not in out.columns:
        out["purchase_day_of_week"] = crawl_dt.dt.dayofweek.fillna(0).astype(int)

    if "purchase_time_bucket" not in out.columns:
        out["purchase_time_bucket"] = crawl_dt.dt.hour.fillna(0).astype(int).apply(_get_time_bucket)

    if "days_until_departure" not in out.columns:
        days = (dep_dt.dt.normalize() - crawl_dt.dt.normalize()).dt.days
        out["days_until_departure"] = days.fillna(0).astype(int)

    if "days_until_departure_bucket" not in out.columns:
        out["days_until_departure_bucket"] = out["days_until_departure"].astype(int).apply(_get_days_until_bucket)

    if "is_weekend_departure" not in out.columns:
        out["is_weekend_departure"] = (dep_dt.dt.dayofweek >= 5).fillna(False).astype(int)

    if "is_holiday_season" not in out.columns:
        out["is_holiday_season"] = dep_dt.dt.month.isin(HOLIDAY_MONTHS).fillna(False).astype(int)

    if "total_time_min" not in out.columns:
        if "total_time" in out.columns:
            out["total_time_min"] = out["total_time"].apply(_parse_total_time_to_min)
        elif "Total Time" in out.columns:
            out["total_time_min"] = out["Total Time"].apply(_parse_total_time_to_min)
        else:
            out["total_time_min"] = 0

    if "flight_duration_bucket" not in out.columns:
        out["flight_duration_bucket"] = out["total_time_min"].astype(int).apply(_get_duration_bucket)

    if "route" not in out.columns:
        if "origin" in out.columns and "destination" in out.columns:
            out["route"] = out["origin"].astype(str) + "_" + out["destination"].astype(str)
        else:
            out["route"] = "UNK_UNK"

    if "route_hash" not in out.columns:
        out["route_hash"] = out["route"].astype(str).apply(_route_hash_to_int).astype(np.int64)

    return out

def _apply_ordinal_map(df: pd.DataFrame, ordinal_map: Any) -> pd.DataFrame:
    if not isinstance(ordinal_map, dict):
        return df
    if "days_until_departure_bucket" not in df.columns:
        return df
    out = df.copy()
    s = out["days_until_departure_bucket"]
    if pd.api.types.is_numeric_dtype(s):
        return out
    out["days_until_departure_bucket"] = s.map(ordinal_map).fillna(0).astype(int)
    return out

def _ensure_required_columns(df: pd.DataFrame, ct) -> pd.DataFrame:
    required = list(getattr(ct, "feature_names_in_", []))
    out = df.copy()

    for c in required:
        if c in out.columns:
            continue
        if c == "purchase_time_bucket":
            out[c] = "dawn"
        elif c == "flight_duration_bucket":
            out[c] = "short"
        elif c == "days_until_departure_bucket":
            out[c] = 0
        else:
            out[c] = 0

    return out

# ---------------- SageMaker hooks ----------------
def model_fn(model_dir: str):
    payload_path = os.path.join(model_dir, "featurizer.joblib")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Missing featurizer payload: {payload_path}")

    payload = joblib.load(payload_path)
    ct = payload["ct"]
    feature_names = payload["feature_names"]
    ordinal_map = payload.get("ordinal_map", {})

    target_transform = None
    tt_path = os.path.join(model_dir, "target_transform.json")
    if os.path.exists(tt_path):
        with open(tt_path, "r", encoding="utf-8") as f:
            target_transform = json.load(f)

    route_stats = None
    rs_path = os.path.join(model_dir, "route_stats.json")
    if os.path.exists(rs_path):
        with open(rs_path, "r", encoding="utf-8") as f:
            route_stats = json.load(f)

    return {
        "ct": ct,
        "feature_names": feature_names,
        "ordinal_map": ordinal_map,
        "target_transform": target_transform,
        "route_stats": route_stats,
    }

def input_fn(request_body, content_type: str = "application/json"):
    # FIX 1: robust decode
    if isinstance(request_body, (bytes, bytearray)):
        body_str = request_body.decode("utf-8")
    else:
        body_str = str(request_body)

    if content_type == "application/json":
        data = json.loads(body_str)
        if isinstance(data, dict) and "records" in data:
            records = data["records"]
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise ValueError("Invalid JSON payload: must be dict/list")
        return pd.DataFrame(records)

    if content_type == "text/csv":
        # CSV MUST have header
        return pd.read_csv(io.StringIO(body_str))

    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: pd.DataFrame, model):
    ct = model["ct"]

    input_data = _derive_features_from_minimal(input_data)
    input_data = _apply_ordinal_map(input_data, model.get("ordinal_map"))
    input_data = _ensure_required_columns(input_data, ct)
    input_data = _apply_route_stats(input_data, model.get("route_stats"))

    required = list(ct.feature_names_in_)
    X = input_data[required].copy()
    arr = ct.transform(X)

    # FIX 2: sparse -> dense for safe CSV downstream
    if sp is not None and sp.issparse(arr):
        arr = arr.toarray()
    elif hasattr(arr, "toarray"):  # just in case
        try:
            arr = arr.toarray()
        except Exception:
            pass

    return pd.DataFrame(arr)

def output_fn(prediction, accept: str = "text/csv"):
    if accept not in ("text/csv", "application/json", "text/plain"):
        raise ValueError(f"Unsupported accept: {accept}")

    if isinstance(prediction, pd.DataFrame):
        csv_str = prediction.to_csv(index=False, header=False)
    else:
        csv_str = pd.DataFrame(prediction).to_csv(index=False, header=False)

    return csv_str, "text/csv"
