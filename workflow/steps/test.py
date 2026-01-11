# workflow/steps/test.py
"""
Flight Fare Test Module (Pipeline-friendly) - NUMERIC INPUT VERSION + MLFLOW

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tar", type=str, default="/opt/ml/processing/model/model.tar.gz")
    p.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    p.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    p.add_argument("--extract-dir", type=str, default="/opt/ml/processing/model_extracted")

    p.add_argument("--test-file", type=str, default="test_numeric.csv")
    p.add_argument("--target-col", type=str, default="fare")
    p.add_argument("--target-transform", type=str, default="log1p", choices=["none", "log1p"])
    p.add_argument("--sample-n", type=int, default=200)
    return p.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _load_xy(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column is required in numeric dataset: {csv_path}")

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)

    X_df = df.drop(columns=[target_col]).copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X = X_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(
            f"X contains NaN/Inf after numeric coercion: {csv_path} "
            f"(example index={bad[0][0]}, col={bad[1][0]})"
        )
    if not np.isfinite(y).all():
        raise ValueError(f"y contains NaN/Inf after numeric coercion: {csv_path}")

    return X, y, X_df.columns.tolist()


def _extract_model(model_tar: str, extract_dir: str) -> str:
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_tar):
        raise FileNotFoundError(f"model tar not found: {model_tar}")

    with tarfile.open(model_tar, "r:gz") as tar:
        # 일반 환경에서는 그대로 extract 가능
        tar.extractall(path=extract_dir)

    candidate = Path(extract_dir) / "xgboost-model"
    if candidate.exists():
        return str(candidate)

    for p in Path(extract_dir).rglob("xgboost-model"):
        return str(p)

    raise FileNotFoundError(f"xgboost-model not found after extracting: {model_tar}")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mlflow_enabled = _mlflow_setup("test")
    try:
        print(f"[test] extracting model tar: {args.model_tar} -> {args.extract_dir}")
        model_path = _extract_model(args.model_tar, args.extract_dir)
        print(f"[test] model file: {model_path}")

        booster = xgb.Booster()
        booster.load_model(model_path)

        test_path = os.path.join(args.test_dir, args.test_file)
        print(f"[test] loading test: {test_path}")
        X_test, y_test, feat_cols = _load_xy(test_path, target_col=args.target_col)
        print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")

        dtest = xgb.DMatrix(X_test, feature_names=feat_cols)
        y_pred = booster.predict(dtest)

        if args.target_transform == "log1p":
            y_true_eval = np.expm1(y_test)
            y_pred_eval = np.expm1(y_pred)
        else:
            y_true_eval = y_test
            y_pred_eval = y_pred

        metrics = {
            "target_col": args.target_col,
            "target_transform": args.target_transform,
            "test_rmse": rmse(y_true_eval, y_pred_eval),
            "test_mae": mae(y_true_eval, y_pred_eval),
            "n_test": int(len(y_test)),
            "n_features": int(X_test.shape[1]),
        }

        metrics_path = Path(args.output_dir) / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        sample_n = min(args.sample_n, len(y_true_eval))
        sample_df = pd.DataFrame(
            {
                "y_true": y_true_eval[:sample_n],
                "y_pred": y_pred_eval[:sample_n],
                "abs_error": np.abs(y_pred_eval[:sample_n] - y_true_eval[:sample_n]),
            }
        )
        sample_df.to_csv(Path(args.output_dir) / "test_predictions_sample.csv", index=False)

        print("[test] metrics:\n", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("[test] done")

        if mlflow_enabled and mlflow is not None:
            mlflow.log_param("target_transform", args.target_transform)
            mlflow.log_metric("test_rmse_rupee", float(metrics["test_rmse"]))
            mlflow.log_metric("test_mae_rupee", float(metrics["test_mae"]))
            mlflow.log_metric("n_test", int(metrics["n_test"]))
            mlflow.log_metric("n_features", int(metrics["n_features"]))

            try:
                mlflow.log_artifacts(str(args.output_dir), artifact_path="test_output")
            except Exception:
                pass

    finally:
        _mlflow_end(mlflow_enabled)


if __name__ == "__main__":
    main()
