# workflow/steps/train.py
"""
Flight Price Training Module (XGBoost Regressor) - NUMERIC INPUT VERSION

Input (channels):
  - SM_CHANNEL_TRAIN: contains train_numeric.csv
  - SM_CHANNEL_VALIDATION: contains validation_numeric.csv

Output:
  /opt/ml/model/xgboost-model           (REQUIRED for SageMaker XGBoost built-in inference)
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


def parse_args():
    p = argparse.ArgumentParser()

    # SageMaker default paths
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # file names
    p.add_argument("--train-file", type=str, default="train_numeric.csv")
    p.add_argument("--validation-file", type=str, default="validation_numeric.csv")

    # hyperparams
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


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)

    if "price" not in df.columns:
        raise ValueError(f"'price' column is required in numeric dataset: {csv_path}")

    # y
    y = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=np.float32)

    # X (all numeric)
    X_df = df.drop(columns=["price"]).copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    X = X_df.to_numpy(dtype=np.float32)

    # NaN/Inf guard
    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(f"X contains NaN/Inf after numeric coercion: {csv_path} (example index={bad[0][0]}, col={bad[1][0]})")
    if not np.isfinite(y).all():
        raise ValueError(f"y contains NaN/Inf after numeric coercion: {csv_path}")

    return X, y, X_df.columns.tolist()


def main():
    args = parse_args()

    train_path = os.path.join(args.train, args.train_file)
    val_path = os.path.join(args.validation, args.validation_file)

    print(f"[train] loading train: {train_path}")
    X_train, y_train, feat_cols = _load_xy(train_path)
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")

    print(f"[train] loading validation: {val_path}")
    X_val, y_val, _ = _load_xy(val_path)
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

    metrics = {
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

    # REQUIRED for XGBoost built-in inference container
    model_path = model_dir / "xgboost-model"
    booster.save_model(str(model_path))
    print(f"[train] saved model: {model_path}")

    # save metrics in model dir
    (model_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # save metrics in output-data-dir (optional but useful)
    out_dir = Path(args.output_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[train] done")


if __name__ == "__main__":
    main()
