"""
train.py
========
Trains a Random Forest classifier on the generated dataset.

Why Random Forest?
------------------
- Handles the mix of binary flags, histograms, and continuous stats
  without needing feature scaling.
- Feature importances reveal which structural signals matter most.
- Fast training, deterministic with a fixed seed.
- Outperforms gradient boosting and MLP on this dataset in practice
  because the most discriminating features (exact length flags for
  ML-KEM, overhead patterns for AES-GCM) are simple threshold rules
  that decision trees capture perfectly.

Usage
-----
    python train.py                          # expects dataset.npz in cwd
    python train.py --data path/dataset.npz
    python train.py --data dataset.npz --trees 500
"""

import os
import sys
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_DIM


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PQC cipher classifier")
    p.add_argument("--data",   default="dataset.npz", help="Path to dataset.npz")
    p.add_argument("--trees",  type=int, default=500,  help="Number of trees (default 500)")
    p.add_argument("--test",   type=float, default=0.20, help="Test split fraction")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--model",  default="model.pkl",    help="Where to save the model")
    return p.parse_args()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── Load dataset ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("PQC Cipher Classifier  —  Training")
    print("=" * 65)

    if not os.path.exists(args.data):
        print(f"ERROR: dataset not found at '{args.data}'")
        print("Run:  python generate_dataset.py  first.")
        sys.exit(1)

    data   = np.load(args.data, allow_pickle=True)
    X      = data["X"].astype(np.float32)
    y      = data["y"].astype(np.int32)
    labels = list(data["labels"])

    print(f"\nDataset       : {args.data}")
    print(f"Total samples : {len(X)}")
    print(f"Feature dim   : {X.shape[1]}")
    print(f"Classes       : {len(labels)}")
    for i, lbl in enumerate(labels):
        print(f"  [{i}] {lbl}  ({int(np.sum(y == i))} samples)")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test,
        random_state=args.seed,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")

    # ── Fit Random Forest ─────────────────────────────────────────────────────
    print(f"\nTraining Random Forest ({args.trees} trees) …")
    clf = RandomForestClassifier(
        n_estimators   = args.trees,
        max_features   = "sqrt",
        min_samples_leaf = 1,
        n_jobs         = -1,          # use all CPU cores
        random_state   = args.seed,
        verbose        = 0,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*65}")
    print(f"TEST ACCURACY : {acc * 100:.2f}%")
    print(f"{'='*65}\n")
    print(classification_report(y_test, y_pred, target_names=labels, digits=3))

    # Confusion matrix (text)
    cm = confusion_matrix(y_test, y_pred)
    col_w = 10
    header = " " * 14 + "  ".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    print("Confusion Matrix  (rows = true, columns = predicted):")
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:{col_w}d}" for v in row)
        print(f"{labels[i][:12]:>12}  {row_str}")

    # ── Feature importances ───────────────────────────────────────────────────
    fi      = clf.feature_importances_
    top_idx = np.argsort(fi)[::-1][:15]
    print(f"\nTop-15 most important features:")
    feat_names = _feature_names()
    for rank, idx in enumerate(top_idx, 1):
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"  {rank:2d}.  {name:35s}  {fi[idx]:.4f}")

    # ── 5-fold cross-validation (on full dataset) ─────────────────────────────
    print(f"\nRunning 5-fold cross-validation on full dataset …")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── Save model ────────────────────────────────────────────────────────────
    payload = {
        "model"      : clf,
        "labels"     : labels,
        "accuracy"   : acc,
        "cv_mean"    : float(cv_scores.mean()),
        "cv_std"     : float(cv_scores.std()),
        "feature_dim": int(X.shape[1]),
    }
    with open(args.model, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"\nModel saved → {args.model}")
    print("Training complete.")
    return acc


def _feature_names():
    """Human-readable names for the first ~30 features."""
    names = ["length"]
    for mod in [8, 16, 32, 64, 128]:
        names.append(f"len_mod_{mod}")
    names += [
        "is_block16_aligned", "is_block8_aligned",
        "has_iv16_prefix", "gcm_body_length_valid",
        "is_mlkem512_len", "is_mlkem768_len", "is_mlkem1024_len",
        "is_pqc_length", "is_short_ct",
    ]
    for i in range(256):
        names.append(f"byte_hist_{i}")
    names += ["mean", "std", "min", "max", "median", "p25", "p75", "p10", "p90"]
    names += ["entropy", "bigram_entropy", "chi2"]
    for lag in [1, 2, 4, 8, 16, 32, 64]:
        names.append(f"autocorr_lag{lag}")
    return names


if __name__ == "__main__":
    train(parse_args())
