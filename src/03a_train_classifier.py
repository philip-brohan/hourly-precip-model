#!/usr/bin/env python3
"""Train an XGBoost binary classifier (rain/no-rain) from prepared HDF5 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_h5_sample(
    path: Path, max_rows: int | None, seed: int, label_key: str = "y_wet"
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
        y = h5[label_key][idx].astype(np.float32)

    return X, y, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost binary classifier")
    parser.add_argument("--train-file", default="data/train.h5")
    parser.add_argument("--val-file", default="data/val.h5")
    parser.add_argument("--model-out", default="models/xgb_classifier.json")
    parser.add_argument("--importance-out", default="models/classifier_importance.csv")
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
        label_key="y_wet",
    )
    print(f"Loaded train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(
        f"  Rain class distribution: {np.sum(y_train == 0):,} dry, {np.sum(y_train == 1):,} wet"
    )

    print(f"Loading validation data from {val_file}")
    X_val, y_val, _ = load_h5_sample(
        val_file,
        max_rows=args.max_val_rows,
        seed=args.seed + 1,
        label_key="y_wet",
    )
    print(f"Loaded val: {X_val.shape[0]:,} rows")
    print(
        f"  Rain class distribution: {np.sum(y_val == 0):,} dry, {np.sum(y_val == 1):,} wet"
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # Calculate scale_pos_weight to address class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class balance: {n_neg:,} neg, {n_pos:,} pos. scale_pos_weight={scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": args.tree_method,
        "seed": args.seed,
        "scale_pos_weight": scale_pos_weight,
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
    print(f"Saved classifier to {model_out}")

    yhat_val_prob = booster.predict(dval)
    yhat_val = (yhat_val_prob > 0.5).astype(int)

    val_acc = accuracy_score(y_val, yhat_val)
    val_prec = precision_score(y_val, yhat_val)
    val_rec = recall_score(y_val, yhat_val)
    val_f1 = f1_score(y_val, yhat_val)
    val_auc = roc_auc_score(y_val, yhat_val_prob)

    print("Validation metrics:")
    print(f"  Accuracy  : {val_acc:.4f}")
    print(f"  Precision : {val_prec:.4f}")
    print(f"  Recall    : {val_rec:.4f}")
    print(f"  F1 Score  : {val_f1:.4f}")
    print(f"  ROC AUC   : {val_auc:.4f}")

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
