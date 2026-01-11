# workflow/steps/train.py
"""
Flight Fare Training Module (XGBoost Regressor) - NUMERIC INPUT VERSION + MLFLOW

Input (channels):
  - SM_CHANNEL_TRAIN: contains train_numeric.csv
  - SM_CHANNEL_VALIDATION: contains validation_numeric.csv

Output:
  /opt/ml/model/xgboost-model
  /opt/ml/model/train_metrics.json
  /opt/ml/output/data/train_metrics.json
"""

import argparse
import json
import os
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
    p.add_argument("--target-transform", type=str, default="log1p", choices=["none", "log1p"])

    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    p.add_argument("--train-file", type=str, default="train_numeric.csv")
    p.add_argument("--validation-file", type=str, default="validation_numeric.csv")
    p.add_argument("--target-col", type=str, default="fare")

    p.add_argument("--objective", type=str, default="reg:squarederror")
    p.add_argument("--eval-metric", type=str, default="rmse")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--tree-method", type=str, default="hist")
    return p.parse_args()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def main():
    args = parse_args()
    mlflow_enabled = _mlflow_setup("train")

    try:
        train_path = os.path.join(args.train, args.train_file)
        val_path = os.path.join(args.validation, args.validation_file)

        print(f"[train] loading train: {train_path}")
        X_train, y_train, feat_cols = _load_xy(train_path, target_col=args.target_col)
        print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")

        print(f"[train] loading validation: {val_path}")
        X_val, y_val, _ = _load_xy(val_path, target_col=args.target_col)
        print(f"  - X_val: {X_val.shape}, y_val: {y_val.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)

        params = {
            "objective": args.objective,
            "eval_metric": args.eval_metric,
            "eta": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "tree_method": args.tree_method,
            "max_bin": args.max_bin,
            "seed": args.seed,
            "verbosity": 1,
        }

        if mlflow_enabled and mlflow is not None:
            mlflow.log_param("target_transform", args.target_transform)
            # log hyperparams
            for k, v in {
                "objective": args.objective,
                "eval_metric": args.eval_metric,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "early_stopping_rounds": args.early_stopping_rounds,
                "tree_method": args.tree_method,
                "max_bin": args.max_bin,
                "seed": args.seed,
            }.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("n_train", int(X_train.shape[0]))
            mlflow.log_metric("n_validation", int(X_val.shape[0]))
            mlflow.log_metric("n_features", int(X_train.shape[1]))

        print("[train] start xgboost training (early stopping)")
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=50,
        )

        best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
        best_score = float(getattr(booster, "best_score", float("nan")))

        yhat_train = booster.predict(dtrain, iteration_range=(0, best_iteration + 1))
        yhat_val = booster.predict(dval, iteration_range=(0, best_iteration + 1))

        if args.target_transform == "log1p":
            y_train_eval = np.expm1(y_train)
            yhat_train_eval = np.expm1(yhat_train)
            y_val_eval = np.expm1(y_val)
            yhat_val_eval = np.expm1(yhat_val)
        else:
            y_train_eval, yhat_train_eval = y_train, yhat_train
            y_val_eval, yhat_val_eval = y_val, yhat_val

        metrics = {
            "train_rmse_rupee": _rmse(y_train_eval, yhat_train_eval),
            "validation_rmse_rupee": _rmse(y_val_eval, yhat_val_eval),
            "train_mae_rupee": _mae(y_train_eval, yhat_train_eval),
            "validation_mae_rupee": _mae(y_val_eval, yhat_val_eval),
            "target_transform": args.target_transform,
            "target_col": args.target_col,
            "best_iteration": best_iteration,
            "best_score_rmse_from_xgb": best_score,
            "train_rmse": _rmse(y_train, yhat_train),
            "validation_rmse": _rmse(y_val, yhat_val),
            "train_mae": _mae(y_train, yhat_train),
            "validation_mae": _mae(y_val, yhat_val),
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_validation": int(X_val.shape[0]),
            "feature_columns_count": int(len(feat_cols)),
            "params": {
                "objective": args.objective,
                "eval_metric": args.eval_metric,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "early_stopping_rounds": args.early_stopping_rounds,
                "tree_method": args.tree_method,
                "max_bin": args.max_bin,
                "seed": args.seed,
            },
        }

        print("[train] metrics:\n", json.dumps(metrics, indent=2, ensure_ascii=False))

        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "xgboost-model"
        booster.save_model(str(model_path))
        print(f"[train] saved model: {model_path}")

        (model_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        out_dir = Path(args.output_data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        if mlflow_enabled and mlflow is not None:
            # 핵심 metric 기록
            mlflow.log_metric("train_rmse_rupee", float(metrics["train_rmse_rupee"]))
            mlflow.log_metric("validation_rmse_rupee", float(metrics["validation_rmse_rupee"]))
            mlflow.log_metric("train_mae_rupee", float(metrics["train_mae_rupee"]))
            mlflow.log_metric("validation_mae_rupee", float(metrics["validation_mae_rupee"]))
            mlflow.log_metric("best_iteration", int(best_iteration))

            # artifacts: model dir / output dir
            try:
                mlflow.log_artifacts(str(model_dir), artifact_path="model_artifacts")
            except Exception:
                pass
            try:
                mlflow.log_artifacts(str(out_dir), artifact_path="train_output")
            except Exception:
                pass

        print("[train] done")
    finally:
        _mlflow_end(mlflow_enabled)


if __name__ == "__main__":
    main()
