import argparse
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")


# =============================================================================
# Utils
# =============================================================================
def detect_outliers_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    return lower_bound, upper_bound, outlier_mask


def _drop_target_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["price", "price_original"] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def add_past_route_stats(
    df_current: pd.DataFrame,
    df_history: Optional[pd.DataFrame],
    time_col: str = "crawl_datetime",
    route_col: str = "route",
    fare_col: str = "Fare",
) -> pd.DataFrame:
    cur = df_current.copy()
    cur["_is_current"] = 1

    if df_history is None or len(df_history) == 0:
        hist = cur.iloc[0:0].copy()
    else:
        hist = df_history.copy()
    hist["_is_current"] = 0

    combo = pd.concat([hist, cur], axis=0, ignore_index=True)
    combo = combo.sort_values(time_col).reset_index(drop=True)

    past_mean = (
        combo.groupby(route_col)[fare_col]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    global_mean = float(hist[fare_col].mean()) if len(hist) > 0 else float(combo[fare_col].mean())
    past_mean = past_mean.fillna(global_mean).fillna(combo[fare_col])

    combo["current_vs_historical_avg"] = combo[fare_col] / past_mean
    combo["price_trend_7d"] = (combo[fare_col] - past_mean) / past_mean

    out = combo[combo["_is_current"] == 1].drop(columns=["_is_current"]).reset_index(drop=True)
    return out


# =============================================================================
# Feature Engineering
# =============================================================================
class FlightFeatureEngineer:
    def __init__(self, apply_log_to_target=False):
        self.holiday_months = [1, 3, 4, 5, 6, 8, 10, 11]
        self.apply_log_to_target = apply_log_to_target

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["crawl_datetime"] = pd.to_datetime(df["Crawl Timestamp"], utc=True).dt.tz_localize(None)
        df["departure_datetime"] = pd.to_datetime(df["Departure Date"] + " " + df["Departure Time"])

        df["route"] = df["Source"].astype(str) + "_" + df["Destination"].astype(str)

        features = pd.DataFrame(index=df.index)

        features["purchase_day_of_week"] = df["crawl_datetime"].dt.dayofweek
        features["purchase_time_bucket"] = df["crawl_datetime"].dt.hour.apply(self._get_time_bucket)

        days_until = (df["departure_datetime"] - df["crawl_datetime"]).dt.days
        features["days_until_departure_bucket"] = days_until.apply(self._get_days_until_bucket)

        features["is_weekend_departure"] = (df["departure_datetime"].dt.dayofweek >= 5).astype(int)
        features["is_holiday_season"] = (df["departure_datetime"].dt.month.isin(self.holiday_months)).astype(int)

        # placeholder (split 후 leak-safe로 재계산)
        features["price_trend_7d"] = 0.0
        features["current_vs_historical_avg"] = 1.0

        features["route_hash"] = df.apply(lambda r: self._hash_route(r["Source"], r["Destination"]), axis=1)

        features["stops_count"] = df["Number Of Stops"]

        total_minutes = df["Total Time"].apply(self._parse_duration)
        features["flight_duration_bucket"] = total_minutes.apply(self._get_duration_bucket)

        # ✅ 변경: source/destination categorical
        features["source"] = df["Source"].astype(str)
        features["destination"] = df["Destination"].astype(str)

        if self.apply_log_to_target:
            features["price"] = np.log1p(df["Fare"])
            features["price_original"] = df["Fare"]
        else:
            features["price"] = df["Fare"]

        # meta columns
        features["_crawl_datetime"] = df["crawl_datetime"]
        features["_route"] = df["route"]
        features["_fare"] = df["Fare"]

        return features

    def _get_time_bucket(self, hour: int) -> str:
        if 0 <= hour < 6:
            return "dawn"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "night"

    def _get_days_until_bucket(self, days: int) -> str:
        if days < 7:
            return "very_close"
        elif days < 14:
            return "close"
        elif days < 30:
            return "medium"
        else:
            return "far"

    def _hash_route(self, source: str, destination: str) -> str:
        route_str = f"{source}_{destination}"
        return hashlib.md5(route_str.encode()).hexdigest()[:8]

    def _parse_duration(self, duration_str: str) -> int:
        try:
            if pd.isna(duration_str):
                return 0
            s = str(duration_str)
            hours = 0
            minutes = 0
            if "h" in s:
                parts = s.split("h")
                hours = int(parts[0].strip())
                if len(parts) > 1 and "m" in parts[1]:
                    minutes = int(parts[1].replace("m", "").strip())
            elif "m" in s:
                minutes = int(s.replace("m", "").strip())
            return hours * 60 + minutes
        except Exception:
            return 0

    def _get_duration_bucket(self, minutes: int) -> str:
        if minutes < 120:
            return "short"
        elif minutes < 360:
            return "medium"
        else:
            return "long"


# =============================================================================
# Main (Pipeline entry)
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    p.add_argument("--split-train", type=float, default=0.7)
    p.add_argument("--split-val", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional MLflow (있으면 쓰고, 없으면 스킵)
    mlflow = None
    try:
        import mlflow as _mlflow  # noqa
        mlflow = _mlflow
    except Exception:
        pass

    df_raw = pd.read_csv(args.input_file)
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)

    lower, upper, _ = detect_outliers_iqr(df_raw, "Fare", multiplier=1.5)
    df_raw["Fare"] = df_raw["Fare"].clip(lower=lower, upper=upper)

    engineer = FlightFeatureEngineer(apply_log_to_target=False)
    df_features = engineer.transform(df_raw)

    df_meta = pd.DataFrame(
        {
            "crawl_datetime": df_features["_crawl_datetime"],
            "route": df_features["_route"],
            "Fare": df_features["_fare"],
        }
    )
    df_features = df_features.drop(columns=["_crawl_datetime", "_route", "_fare"])

    # time order split
    order = df_meta["crawl_datetime"].sort_values().index
    df_meta = df_meta.loc[order].reset_index(drop=True)
    df_features = df_features.loc[order].reset_index(drop=True)

    n = len(df_features)
    train_end = int(n * args.split_train)
    val_end = int(n * (args.split_train + args.split_val))

    feat_train = df_features.iloc[:train_end].reset_index(drop=True)
    feat_val = df_features.iloc[train_end:val_end].reset_index(drop=True)
    feat_test = df_features.iloc[val_end:].reset_index(drop=True)

    meta_train = df_meta.iloc[:train_end].reset_index(drop=True)
    meta_val = df_meta.iloc[train_end:val_end].reset_index(drop=True)
    meta_test = df_meta.iloc[val_end:].reset_index(drop=True)

    # leak-safe route stats
    train_stats = add_past_route_stats(meta_train, None)
    val_stats = add_past_route_stats(meta_val, meta_train)
    test_stats = add_past_route_stats(meta_test, pd.concat([meta_train, meta_val], ignore_index=True))

    feat_train["price_trend_7d"] = train_stats["price_trend_7d"].values
    feat_train["current_vs_historical_avg"] = train_stats["current_vs_historical_avg"].values
    feat_val["price_trend_7d"] = val_stats["price_trend_7d"].values
    feat_val["current_vs_historical_avg"] = val_stats["current_vs_historical_avg"].values
    feat_test["price_trend_7d"] = test_stats["price_trend_7d"].values
    feat_test["current_vs_historical_avg"] = test_stats["current_vs_historical_avg"].values

    # save CSVs (unprocessed features + target)
    target_col = "price"

    X_train = _drop_target_cols(feat_train); y_train = feat_train[target_col].values
    X_val = _drop_target_cols(feat_val); y_val = feat_val[target_col].values
    X_test = _drop_target_cols(feat_test); y_test = feat_test[target_col].values

    train_df = X_train.copy(); train_df["price"] = y_train
    val_df = X_val.copy(); val_df["price"] = y_val
    test_df = X_test.copy(); test_df["price"] = y_test

    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv(os.path.join(args.output_dir, "final_dataset.csv"), index=False)

    if mlflow:
        try:
            mlflow.log_metric("n_train", len(train_df))
            mlflow.log_metric("n_val", len(val_df))
            mlflow.log_metric("n_test", len(test_df))
        except Exception:
            pass

    print(f"[preprocess] saved to: {args.output_dir}")
    print(f" - train: {train_df.shape}")
    print(f" - val  : {val_df.shape}")
    print(f" - test : {test_df.shape}")


if __name__ == "__main__":
    main()
