#!/usr/bin/env python3
"""Train an XGBoost precipitation model from prepared HDF5 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_h5_sample(
    path: Path, max_rows: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as h5:
        n = h5["X"].shape[0]
        feature_names = json.loads(h5.attrs["feature_names_json"])
        if max_rows is None or max_rows >= n:
            idx = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_rows, replace=False))

        X = h5["X"][idx, :].astype(np.float32)
        y = h5["y_log"][idx].astype(np.float32)

    return X, y, feature_names


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--train-file", default="data/train.h5")
    parser.add_argument("--val-file", default="data/val.h5")
    parser.add_argument("--model-out", default="models/xgb_precip.json")
    parser.add_argument("--importance-out", default="models/feature_importance.csv")
    parser.add_argument("--max-train-rows", type=int, default=5_000_000)
    parser.add_argument("--max-val-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist"])
    args = parser.parse_args()

    train_file = Path(args.train_file)
    val_file = Path(args.val_file)

    print(f"Loading training data from {train_file}")
    X_train, y_train, feature_names = load_h5_sample(
        train_file,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    print(f"Loaded train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    print(f"Loading validation data from {val_file}")
    X_val, y_val, _ = load_h5_sample(
        val_file,
        max_rows=args.max_val_rows,
        seed=args.seed + 1,
    )
    print(f"Loaded val: {X_val.shape[0]:,} rows")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": args.tree_method,
        "seed": args.seed,
    }

    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_out)
    print(f"Saved model to {model_out}")

    yhat_val_log = booster.predict(dval)
    val_rmse_log = rmse(y_val, yhat_val_log)
    val_mae_log = float(mean_absolute_error(y_val, yhat_val_log))

    y_val_raw = np.expm1(y_val)
    yhat_val_raw = np.expm1(yhat_val_log)
    val_rmse_raw = rmse(y_val_raw, yhat_val_raw)
    val_mae_raw = float(mean_absolute_error(y_val_raw, yhat_val_raw))

    print("Validation metrics:")
    print(f"  log1p(PRATE): RMSE={val_rmse_log:.6f}, MAE={val_mae_log:.6f}")
    print(f"  PRATE       : RMSE={val_rmse_raw:.6f}, MAE={val_mae_raw:.6f}")

    score = booster.get_score(importance_type="gain")
    rows = []
    for name in feature_names:
        rows.append({"feature": name, "gain": float(score.get(name, 0.0))})

    fi = pd.DataFrame(rows).sort_values("gain", ascending=False)
    importance_out = Path(args.importance_out)
    importance_out.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(importance_out, index=False)
    print(f"Saved feature importance to {importance_out}")


if __name__ == "__main__":
    main()
