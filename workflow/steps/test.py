# workflow/steps/test.py
"""
Flight Price Test Module (Pipeline-friendly) - NUMERIC INPUT VERSION

Input:
  --model-tar: /opt/ml/processing/model/model.tar.gz
  --test-dir : /opt/ml/processing/test (contains test_numeric.csv)
Output:
  --output-dir: /opt/ml/processing/output
    - test_metrics.json
    - test_predictions_sample.csv
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tar", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--test-file", type=str, default="test_numeric.csv")
    p.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    return p.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    extract_dir = "/opt/ml/processing/model_extracted"
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    # Extract model.tar.gz
    with tarfile.open(args.model_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Built-in XGBoost artifact name
    model_path = os.path.join(extract_dir, "xgboost-model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find 'xgboost-model' in extracted dir: {extract_dir}")

    booster = xgb.Booster()
    booster.load_model(model_path)

    test_path = os.path.join(args.test_dir, args.test_file)
    df_test = pd.read_csv(test_path)

    if "price" not in df_test.columns:
        raise ValueError(f"'price' column required in test file: {test_path}")

    y_test = df_test["price"].to_numpy(dtype=np.float32)
    X_test = df_test.drop(columns=["price"]).to_numpy(dtype=np.float32)

    dtest = xgb.DMatrix(X_test)
    y_pred = booster.predict(dtest)

    metrics = {
        "test_rmse": rmse(y_test, y_pred),
        "test_mae": mae(y_test, y_pred),
        "n_test": int(len(y_test)),
        "n_features": int(X_test.shape[1]),
    }

    (Path(args.output_dir) / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sample_n = min(200, len(y_test))
    sample_df = pd.DataFrame(
        {
            "y_true": y_test[:sample_n],
            "y_pred": y_pred[:sample_n],
            "abs_error": np.abs(y_pred[:sample_n] - y_test[:sample_n]),
        }
    )
    sample_df.to_csv(Path(args.output_dir) / "test_predictions_sample.csv", index=False)

    print("[test] metrics:", metrics)


if __name__ == "__main__":
    main()
