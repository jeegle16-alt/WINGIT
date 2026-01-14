"""
workflow/steps/preprocess.py (FINAL + MLFLOW)

- Target column: 'fare' (LOG SCALE) = log1p(original_fare)
- Split: route-wise time split (7:1:2) to avoid biased tiny validation window
- Leakage-free rolling features (shift(1) + time-based rolling windows)
- Outputs to --output-dir:
    train_numeric.csv
    validation_numeric.csv
    test_numeric.csv
    final_dataset.csv
    target_transform.json

CLI examples (local):
  python steps/preprocess.py --input-file ./raw_data.csv --output-dir ./preprocess_output --outlier-method clip --scale-numeric
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
    """
    Uses Managed MLflow Tracking Server ARN via env:
      - MLFLOW_TRACKING_URI (recommended) OR
      - MLFLOW_TRACKING_ARN / MLFLOW_TRACKING_SERVER_ARN
    """
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

    # run name: pipeline이면 Execution ARN 일부가 들어오도록 pipeline.py에서 env로 주입 가능
    run_name = os.getenv("MLFLOW_RUN_NAME", stage_name)
    mlflow.start_run(run_name=run_name, nested=bool(os.getenv("MLFLOW_NESTED", "0") == "1"))

    # basic tags
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
    """
    Route-wise time split.
    Adds column 'split' in {'train','validation','test'}.
    Handles short routes safely.

    Rule:
      - sort by crawl_datetime
      - n < 5: all train
      - 5 <= n < 10: train=n-2, val=1, test=1
      - else: 7:1:2
    """
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
        return "night"

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
# Numeric Transformer (for XGBoost)
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
        print(f"📥 Loading raw data: {args.input_file}")
        df_raw = pd.read_csv(args.input_file)
        print(f"✅ Loaded: {df_raw.shape}")

        df_raw = df_raw.drop_duplicates().reset_index(drop=True)

        if mlflow_enabled and mlflow is not None:
            mlflow.log_param("outlier_method", args.outlier_method)
            mlflow.log_param("iqr_multiplier", args.iqr_multiplier)
            mlflow.log_param("scale_numeric", bool(args.scale_numeric))
            mlflow.log_param("train_ratio", float(args.train_ratio))
            mlflow.log_param("val_ratio", float(args.val_ratio))
            mlflow.log_metric("n_rows_raw", int(df_raw.shape[0]))

        # Outlier handling on ORIGINAL Fare (before log)
        df_raw["Fare"] = pd.to_numeric(df_raw["Fare"], errors="coerce")
        if args.outlier_method == "clip":
            lower, upper = detect_outliers_iqr(df_raw["Fare"], multiplier=args.iqr_multiplier)
            outlier_mask = (df_raw["Fare"] < lower) | (df_raw["Fare"] > upper)
            print("🔍 Outlier (IQR)")
            print(f"  - lower: {lower:,.0f}, upper: {upper:,.0f}")
            print(f"  - outliers: {int(outlier_mask.sum()):,} ({float(outlier_mask.mean()*100):.2f}%)")
            df_raw["Fare"] = df_raw["Fare"].clip(lower=lower, upper=upper)

            if mlflow_enabled and mlflow is not None:
                mlflow.log_metric("outliers_ratio", float(outlier_mask.mean()))
                mlflow.log_param("iqr_lower", lower)
                mlflow.log_param("iqr_upper", upper)

        print("⚙️ Feature Engineering...")
        engineer = FlightFeatureEngineer()
        df_feat = engineer.transform(df_raw)

        # Drop missing target
        df_feat = df_feat.dropna(subset=["fare", "fare_raw", "crawl_datetime", "route"]).reset_index(drop=True)

        # Route-wise time split
        split_base = df_feat[["route", "crawl_datetime"]].copy()
        split_base["split"] = "train"
        split_base = split_base.groupby("route", group_keys=False).apply(
            lambda g: _route_time_split(g, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        )

        df_feat = df_feat.join(split_base["split"])
        if "split" not in df_feat.columns:
            raise RuntimeError("split column missing after route-wise split")

        train_df = df_feat[df_feat["split"] == "train"].copy()
        val_df = df_feat[df_feat["split"] == "validation"].copy()
        test_df = df_feat[df_feat["split"] == "test"].copy()

        print("📊 Route-wise time split (7:1:2) done")
        print(f"  - Train: {len(train_df):,} (routes={train_df['route'].nunique():,})")
        print(f"  - Val  : {len(val_df):,} (routes={val_df['route'].nunique():,})")
        print(f"  - Test : {len(test_df):,} (routes={test_df['route'].nunique():,})")
        if len(val_df) == 0 or len(test_df) == 0:
            raise RuntimeError("validation/test became empty. Check route counts / split rules.")

        if mlflow_enabled and mlflow is not None:
            mlflow.log_metric("n_train_rows", int(len(train_df)))
            mlflow.log_metric("n_val_rows", int(len(val_df)))
            mlflow.log_metric("n_test_rows", int(len(test_df)))
            mlflow.log_metric("n_routes_train", int(train_df["route"].nunique()))
            mlflow.log_metric("n_routes_val", int(val_df["route"].nunique()))
            mlflow.log_metric("n_routes_test", int(test_df["route"].nunique()))

        # Save audit dataset
        final_path = os.path.join(args.output_dir, "final_dataset.csv")
        df_feat.to_csv(final_path, index=False)

        # Write target transform metadata
        tt = {
            "target_col": "fare",
            "target_scale": "log1p",
            "inverse_transform": "expm1",
            "note": "Model trains on log1p(original_fare). Convert predictions back with expm1 for rupee-scale metrics/output.",
        }
        tt_path = os.path.join(args.output_dir, "target_transform.json")
        with open(tt_path, "w", encoding="utf-8") as f:
            json.dump(tt, f, ensure_ascii=False, indent=2)

        # Prepare X/y (drop debug columns)
        target_col = "fare"
        drop_cols = ["fare", "fare_raw", "crawl_datetime", "route", "split"]

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col].astype(float)

        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[target_col].astype(float)

        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[target_col].astype(float)

        featurizer = NumericFeaturizer(scale_numeric=bool(args.scale_numeric))

        print("🔧 Fit/transform (train), transform (val/test)...")
        train_num = featurizer.fit_transform(X_train)
        val_num = featurizer.transform(X_val)
        test_num = featurizer.transform(X_test)

        # ✅ 커스텀 클래스 전체를 저장하지 말고, sklearn 객체만 저장(추론 로딩 안정화)
        payload = {
            "ct": featurizer.ct,
            "feature_names": featurizer._feature_names,
            "ordinal_map": featurizer.ordinal_map,
        }
        joblib.dump(payload, os.path.join(args.output_dir, "featurizer.joblib"))
        print("✅ Saved featurizer payload:", os.path.join(args.output_dir, "featurizer.joblib"))

        
        # Attach target
        train_num[target_col] = y_train.values
        val_num[target_col] = y_val.values
        test_num[target_col] = y_test.values

        # Sanity checks
        for name, d in [("train", train_num), ("validation", val_num), ("test", test_num)]:
            X_only = d.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
            if X_only.isna().any().any():
                bad_cols = X_only.columns[X_only.isna().any()].tolist()
                raise ValueError(f"{name} has NaN after numeric coercion. bad_cols={bad_cols[:10]}")
            arr = X_only.to_numpy(dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} has Inf/-Inf values after coercion")
            if not np.isfinite(d[target_col].to_numpy(dtype=np.float32)).all():
                raise ValueError(f"{name} target has NaN/Inf")

        # Save numeric sets
        train_path = os.path.join(args.output_dir, "train_numeric.csv")
        val_path = os.path.join(args.output_dir, "validation_numeric.csv")
        test_path = os.path.join(args.output_dir, "test_numeric.csv")

        train_num.to_csv(train_path, index=False)
        val_num.to_csv(val_path, index=False)
        test_num.to_csv(test_path, index=False)

        print("✅ Saved numeric datasets (target is log1p fare):")
        print(f"  - {train_path}  (X={train_num.shape[1]-1} + y=1)")
        print(f"  - {val_path}    (X={val_num.shape[1]-1} + y=1)")
        print(f"  - {test_path}   (X={test_num.shape[1]-1} + y=1)")
        print("✅ Preprocessing complete")

        if mlflow_enabled and mlflow is not None:
            mlflow.log_metric("n_features_numeric", int(train_num.shape[1] - 1))
            # artifacts: output dir 전체를 preprocess_output 아래로 올림
            try:
                mlflow.log_artifacts(args.output_dir, artifact_path="preprocess_output")
            except Exception:
                pass

    finally:
        _mlflow_end(mlflow_enabled)


if __name__ == "__main__":
    main()
