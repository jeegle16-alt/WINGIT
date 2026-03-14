"""
workflow/steps/preprocess.py (FINAL + MLFLOW)

- Target column: 'fare' (LOG SCALE) = log1p(original_fare)
- Split: route-wise time split (7:1:2)
- Leakage-free rolling features (shift(1) + time-based rolling windows)
- Outputs to --output-dir:
    train_numeric.csv
    validation_numeric.csv
    test_numeric.csv
    final_dataset.csv
    target_transform.json
    featurizer.joblib
    route_stats.json   <-- (NEW)
"""

import joblib
import argparse
import hashlib
import json
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# Optional MLflow
# =========================================================
mlflow = None
try:
    import mlflow as _mlflow  # type: ignore
    mlflow = _mlflow
except Exception:
    mlflow = None


def _mlflow_setup(stage_name: str) -> bool:
    if mlflow is None:
        return False

    tracking = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("MLFLOW_TRACKING_ARN")
        or os.getenv("MLFLOW_TRACKING_SERVER_ARN")
    )
    if not tracking:
        return False

    mlflow.set_tracking_uri(tracking)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "FlightPriceXGB")
    mlflow.set_experiment(exp_name)

    run_name = os.getenv("MLFLOW_RUN_NAME", stage_name)
    mlflow.start_run(run_name=run_name, nested=bool(os.getenv("MLFLOW_NESTED", "0") == "1"))

    mlflow.set_tag("stage", stage_name)
    if os.getenv("SM_PIPELINE_EXECUTION_ARN"):
        mlflow.set_tag("sm_pipeline_execution_arn", os.getenv("SM_PIPELINE_EXECUTION_ARN"))
    return True


def _mlflow_end(enabled: bool):
    if enabled and mlflow is not None:
        try:
            mlflow.end_run()
        except Exception:
            pass


# =========================================================
# Utils
# =========================================================
def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return float(lower), float(upper)


def _route_hash_to_int(route: str) -> int:
    return int(hashlib.md5(route.encode("utf-8")).hexdigest()[:8], 16)


def _safe_div(a: pd.Series, b: pd.Series, default: float = 1.0) -> pd.Series:
    b2 = b.replace(0, np.nan)
    out = a / b2
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def _route_time_split(g: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1) -> pd.DataFrame:
    g = g.sort_values("crawl_datetime").copy()
    n = len(g)

    if n < 5:
        g["split"] = "train"
        return g

    if n < 10:
        train_end = max(n - 2, 1)
        val_end = min(train_end + 1, n - 1)
    else:
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n - 1)

    g["split"] = "test"
    g.iloc[:train_end, g.columns.get_loc("split")] = "train"
    g.iloc[train_end:val_end, g.columns.get_loc("split")] = "validation"
    return g


# =========================================================
# Feature Engineering
# =========================================================
class FlightFeatureEngineer:
    def __init__(self):
        # 너가 쓰던 로직 유지
        self.holiday_months = [1, 3, 4, 5, 6, 8, 10, 11]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["crawl_datetime"] = pd.to_datetime(df["Crawl Timestamp"], utc=True).dt.tz_localize(None)
        df["departure_datetime"] = pd.to_datetime(
            df["Departure Date"].astype(str) + " " + df["Departure Time"].astype(str),
            errors="coerce",
        )

        df["route"] = df["Source"].astype(str) + "_" + df["Destination"].astype(str)

        features = pd.DataFrame(index=df.index)
        features["crawl_datetime"] = df["crawl_datetime"]
        features["route"] = df["route"]

        features["purchase_day_of_week"] = df["crawl_datetime"].dt.dayofweek.astype(int)

        # IMPORTANT: time bucket을 Lambda/환경변수랑 맞추기 위해 "evening" 사용
        features["purchase_time_bucket"] = df["crawl_datetime"].dt.hour.apply(self._get_time_bucket)

        days_until = (df["departure_datetime"] - df["crawl_datetime"]).dt.days
        features["days_until_departure"] = pd.to_numeric(days_until, errors="coerce").fillna(0).astype(int)
        features["days_until_departure_bucket"] = features["days_until_departure"].apply(self._get_days_until_bucket)

        features["is_weekend_departure"] = (df["departure_datetime"].dt.dayofweek >= 5).fillna(False).astype(int)
        features["is_holiday_season"] = df["departure_datetime"].dt.month.isin(self.holiday_months).fillna(False).astype(int)

        features["stops_count"] = pd.to_numeric(df["Number Of Stops"], errors="coerce").fillna(0).astype(int)
        total_minutes = df["Total Time"].apply(self._parse_duration).astype(int)
        features["flight_duration_bucket"] = total_minutes.apply(self._get_duration_bucket)

        features["route_hash"] = df["route"].apply(_route_hash_to_int).astype(np.int64)

        fare_raw = pd.to_numeric(df["Fare"], errors="coerce")
        features["fare_raw"] = fare_raw

        tmp = pd.DataFrame(
            {
                "route": df["route"].values,
                "crawl_datetime": df["crawl_datetime"].values,
                "fare_raw": fare_raw.values,
            },
            index=df.index,
        )

        def _add_rolling(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("crawl_datetime").copy()
            s_prev = g.set_index("crawl_datetime")["fare_raw"].shift(1)

            g["prev_fare"] = s_prev.to_numpy()
            for win in ["7D", "14D", "30D"]:
                g[f"min_fare_last_{win.lower()}"] = s_prev.rolling(win, min_periods=3).min().to_numpy()
                g[f"mean_fare_last_{win.lower()}"] = s_prev.rolling(win, min_periods=3).mean().to_numpy()
            return g

        tmp = tmp.groupby("route", group_keys=False).apply(_add_rolling)

        features["prev_fare"] = tmp["prev_fare"]
        for win in ["7d", "14d", "30d"]:
            features[f"min_fare_last_{win}"] = tmp[f"min_fare_last_{win}"]
            features[f"mean_fare_last_{win}"] = tmp[f"mean_fare_last_{win}"]

        global_median = float(fare_raw.median())
        features["prev_fare"] = pd.to_numeric(features["prev_fare"], errors="coerce").fillna(global_median)
        for win in ["7d", "14d", "30d"]:
            features[f"min_fare_last_{win}"] = pd.to_numeric(features[f"min_fare_last_{win}"], errors="coerce").fillna(global_median)
            features[f"mean_fare_last_{win}"] = pd.to_numeric(features[f"mean_fare_last_{win}"], errors="coerce").fillna(global_median)

        for win in ["7d", "14d", "30d"]:
            features[f"prev_fare_vs_min_{win}"] = _safe_div(features["prev_fare"], features[f"min_fare_last_{win}"], default=1.0)
            features[f"prev_fare_vs_mean_{win}"] = _safe_div(features["prev_fare"], features[f"mean_fare_last_{win}"], default=1.0)

        features["fare"] = np.log1p(pd.to_numeric(features["fare_raw"], errors="coerce"))
        return features

    def _get_time_bucket(self, hour: int) -> str:
        if 0 <= hour < 6:
            return "dawn"
        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        return "evening"

    def _get_days_until_bucket(self, days: int) -> str:
        if days < 7:
            return "very_close"
        if days < 14:
            return "close"
        if days < 30:
            return "medium"
        return "far"

    def _parse_duration(self, duration_str: str) -> int:
        try:
            if pd.isna(duration_str):
                return 0
            s = str(duration_str).lower().strip()
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
            elif "m" in s:
                mm = s.replace("m", "").strip()
                if mm:
                    m = int(float(mm))
            return h * 60 + m
        except Exception:
            return 0

    def _get_duration_bucket(self, minutes: int) -> str:
        if minutes < 120:
            return "short"
        if minutes < 360:
            return "medium"
        return "long"


# =========================================================
# Numeric Transformer
# =========================================================
class NumericFeaturizer:
    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.ordinal_map = {"very_close": 0, "close": 1, "medium": 2, "far": 3}

        self.cat_cols = ["purchase_time_bucket", "flight_duration_bucket"]

        self.num_cols = [
            "purchase_day_of_week",
            "days_until_departure",
            "days_until_departure_bucket",
            "stops_count",
            "route_hash",
            "prev_fare",
            "min_fare_last_7d",
            "mean_fare_last_7d",
            "min_fare_last_14d",
            "mean_fare_last_14d",
            "min_fare_last_30d",
            "mean_fare_last_30d",
            "prev_fare_vs_min_7d",
            "prev_fare_vs_mean_7d",
            "prev_fare_vs_min_14d",
            "prev_fare_vs_mean_14d",
            "prev_fare_vs_min_30d",
            "prev_fare_vs_mean_30d",
        ]

        self.bool_cols = ["is_weekend_departure", "is_holiday_season"]

        self.ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler() if scale_numeric else None
        num_transformer = self.scaler if self.scaler is not None else "passthrough"

        self.ct = ColumnTransformer(
            transformers=[
                ("cat", self.ohe, self.cat_cols),
                ("num", num_transformer, self.num_cols),
                ("bool", "passthrough", self.bool_cols),
            ],
            remainder="drop",
        )
        self._feature_names: List[str] = []

    def _encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["days_until_departure_bucket"] = df["days_until_departure_bucket"].map(self.ordinal_map)
        return df

    def fit(self, X: pd.DataFrame):
        Xp = self._encode_ordinal(X)
        self.ct.fit(Xp)

        cat_names = list(self.ct.named_transformers_["cat"].get_feature_names_out(self.cat_cols))
        self._feature_names = cat_names + self.num_cols + self.bool_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = self._encode_ordinal(X)
        arr = self.ct.transform(Xp)
        return pd.DataFrame(arr, columns=self._feature_names)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--outlier-method", type=str, default="clip", choices=["clip", "none"])
    p.add_argument("--iqr-multiplier", type=float, default=1.5)
    p.add_argument("--scale-numeric", action="store_true")

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mlflow_enabled = _mlflow_setup("preprocess")
    try:
        df_raw = pd.read_csv(args.input_file).drop_duplicates().reset_index(drop=True)

        # Outlier handling on ORIGINAL Fare (before log)
        df_raw["Fare"] = pd.to_numeric(df_raw["Fare"], errors="coerce")
        if args.outlier_method == "clip":
            lower, upper = detect_outliers_iqr(df_raw["Fare"], multiplier=args.iqr_multiplier)
            df_raw["Fare"] = df_raw["Fare"].clip(lower=lower, upper=upper)

        engineer = FlightFeatureEngineer()
        df_feat = engineer.transform(df_raw)

        df_feat = df_feat.dropna(subset=["fare", "fare_raw", "crawl_datetime", "route"]).reset_index(drop=True)

        # Route-wise time split
        split_base = df_feat[["route", "crawl_datetime"]].copy()
        split_base["split"] = "train"
        split_base = split_base.groupby("route", group_keys=False).apply(
            lambda g: _route_time_split(g, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        )
        df_feat = df_feat.join(split_base["split"])

        train_df = df_feat[df_feat["split"] == "train"].copy()
        val_df = df_feat[df_feat["split"] == "validation"].copy()
        test_df = df_feat[df_feat["split"] == "test"].copy()

        if len(val_df) == 0 or len(test_df) == 0:
            raise RuntimeError("validation/test became empty. Check split rules.")

        # Save audit dataset
        df_feat.to_csv(os.path.join(args.output_dir, "final_dataset.csv"), index=False)

        # target transform metadata
        tt = {
            "target_col": "fare",
            "target_scale": "log1p",
            "inverse_transform": "expm1",
            "note": "Model trains on log1p(original_fare). Convert predictions back with expm1.",
        }
        with open(os.path.join(args.output_dir, "target_transform.json"), "w", encoding="utf-8") as f:
            json.dump(tt, f, ensure_ascii=False, indent=2)

        # ------------------------------------------------------------
        # NEW: route_stats.json (route_hash별 history-derived 기본값)
        # ------------------------------------------------------------
        stats_cols = [
            "route_hash",
            "prev_fare",
            "min_fare_last_7d", "mean_fare_last_7d",
            "min_fare_last_14d", "mean_fare_last_14d",
            "min_fare_last_30d", "mean_fare_last_30d",
        ]
        missing = [c for c in stats_cols if c not in df_feat.columns]
        if missing:
            print(f"⚠️ route_stats skip. Missing columns: {missing}")
        else:
            tmp = df_feat.sort_values("crawl_datetime").copy()
            route_stats_df = tmp.groupby("route_hash", as_index=False).tail(1)[stats_cols]

            global_fallback = {c: float(df_feat[c].median()) for c in stats_cols if c != "route_hash"}

            route_stats_obj = {
                "by_route_hash": {
                    str(int(r["route_hash"])): {k: float(r[k]) for k in stats_cols if k != "route_hash"}
                    for _, r in route_stats_df.iterrows()
                },
                "global_fallback": global_fallback,
                "meta": {
                    "generated_from": "preprocess",
                    "snapshot_rule": "latest_crawl_datetime_per_route_hash",
                }
            }

            route_stats_path = os.path.join(args.output_dir, "route_stats.json")
            with open(route_stats_path, "w", encoding="utf-8") as f:
                json.dump(route_stats_obj, f, ensure_ascii=False, indent=2)

            print("✅ Saved route_stats:", route_stats_path)

        # Prepare X/y
        target_col = "fare"
        drop_cols = ["fare", "fare_raw", "crawl_datetime", "route", "split"]

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col].astype(float)

        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[target_col].astype(float)

        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[target_col].astype(float)

        featurizer = NumericFeaturizer(scale_numeric=bool(args.scale_numeric))

        train_num = featurizer.fit_transform(X_train)
        val_num = featurizer.transform(X_val)
        test_num = featurizer.transform(X_test)

        # Save featurizer payload (safe)
        payload = {"ct": featurizer.ct, "feature_names": featurizer._feature_names, "ordinal_map": featurizer.ordinal_map}
        joblib.dump(payload, os.path.join(args.output_dir, "featurizer.joblib"))

        # Attach target
        train_num[target_col] = y_train.values
        val_num[target_col] = y_val.values
        test_num[target_col] = y_test.values

        train_num.to_csv(os.path.join(args.output_dir, "train_numeric.csv"), index=False)
        val_num.to_csv(os.path.join(args.output_dir, "validation_numeric.csv"), index=False)
        test_num.to_csv(os.path.join(args.output_dir, "test_numeric.csv"), index=False)

        if mlflow_enabled and mlflow is not None:
            try:
                mlflow.log_artifacts(args.output_dir, artifact_path="preprocess_output")
            except Exception:
                pass

        print("✅ Preprocess done")

    finally:
        _mlflow_end(mlflow_enabled)


if __name__ == "__main__":
    main()
