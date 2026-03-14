"""
workflow/steps/train.py (Pipeline-friendly)

Input:
  --train-dir: contains train_numeric.csv
  --validation-dir: contains validation_numeric.csv
  --preprocess-dir: contains featurizer.joblib, target_transform.json, route_stats.json
Output:
  /opt/ml/model/xgboost-model            (REQUIRED for XGBoost built-in inference container)
  /opt/ml/model/featurizer.joblib
  /opt/ml/model/target_transform.json
  /opt/ml/model/route_stats.json         <-- (NEW, required for realistic predictions)
  /opt/ml/model/code/inference.py
  /opt/ml/model/code/requirements.txt
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--validation-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    p.add_argument("--preprocess-dir", type=str, default=os.environ.get("SM_CHANNEL_PREPROCESS", "/opt/ml/input/data/preprocess"))

    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # hyperparams
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--num-boost-round", type=int, default=300)
    p.add_argument("--early-stopping-rounds", type=int, default=30)

    return p.parse_args()


def _load_csv(dir_path: str, filename: str) -> pd.DataFrame:
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {filename}: {path}")
    return pd.read_csv(path)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # --- Load train/val numeric datasets
    train_df = _load_csv(args.train_dir, "train_numeric.csv")
    val_df = _load_csv(args.validation_dir, "validation_numeric.csv")

    if "fare" not in train_df.columns:
        raise RuntimeError("train_numeric.csv must contain target column 'fare' (log1p).")

    X_train = train_df.drop(columns=["fare"]).astype(np.float32)
    y_train = train_df["fare"].astype(np.float32)

    X_val = val_df.drop(columns=["fare"]).astype(np.float32)
    y_val = val_df["fare"].astype(np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "eval_metric": "rmse",
    }

    evals = [(dtrain, "train"), (dval, "validation")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    # --- Save XGBoost model for built-in inference
    xgb_model_path = os.path.join(args.model_dir, "xgboost-model")
    booster.save_model(xgb_model_path)

    # --- Metrics (log-scale)
    val_pred = booster.predict(dval)
    rmse = float(mean_squared_error(y_val, val_pred, squared=False))
    mae = float(mean_absolute_error(y_val, val_pred))
    metrics = {"rmse_log": rmse, "mae_log": mae, "n_val": int(len(y_val))}
    with open(os.path.join(args.output_data_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Copy preprocess artifacts into model.tar.gz
    # required for inference/preprocess container
    required_files = ["featurizer.joblib", "target_transform.json", "route_stats.json"]
    for fn in required_files:
        src = os.path.join(args.preprocess_dir, fn)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Missing {fn} in preprocess output. Expected: {src}\n"
                f"-> preprocess.py must generate it in --output-dir"
            )
        shutil.copy(src, os.path.join(args.model_dir, fn))

    # --- Copy inference code into /opt/ml/model/code/
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # These two files must be present in the same directory as train.py in source_dir.
    # pipeline.py에서 source_dir로 steps/를 지정하면 steps/inference.py, steps/requirements_inference.txt를 같이 가져갈 수 있음.
    local_infer_py = os.path.join(os.path.dirname(__file__), "inference.py")
    local_req = os.path.join(os.path.dirname(__file__), "requirements_inference.txt")

    if not os.path.exists(local_infer_py):
        raise FileNotFoundError(f"Missing steps/inference.py next to train.py: {local_infer_py}")
    if not os.path.exists(local_req):
        raise FileNotFoundError(f"Missing steps/requirements_inference.txt next to train.py: {local_req}")

    shutil.copy(local_infer_py, os.path.join(code_dir, "inference.py"))
    shutil.copy(local_req, os.path.join(code_dir, "requirements.txt"))

    print("✅ Train done")
    print(" - saved:", xgb_model_path)
    print(" - bundled:", [*required_files, "code/inference.py", "code/requirements.txt"])


if __name__ == "__main__":
    main()
