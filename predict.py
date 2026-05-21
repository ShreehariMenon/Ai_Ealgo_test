"""
predict.py
==========
Load a trained model and predict which encryption algorithm produced
a given ciphertext.

Usage
-----
    # Run demo: encrypt with each algorithm, then predict
    python predict.py --demo

    # Predict from a hex string (paste ciphertext as hex)
    python predict.py --hex  a3f09c2b1d7e...

    # Predict from a binary file
    python predict.py --file  /path/to/ciphertext.bin

    # Batch predict from a folder of binary files
    python predict.py --dir  /path/to/ciphertexts/

Options
-------
    --model   Path to model.pkl  (default: model.pkl in same directory)
    --top     Show top-N predictions (default: 3)
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import extract


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at '{model_path}'")
        print("Run:  python train.py  first.")
        sys.exit(1)
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["model"], payload["labels"], payload


# ── Predict single ciphertext ─────────────────────────────────────────────────

def predict_one(ct_bytes: bytes, model, labels: list, top_n: int = 3):
    feat  = extract(ct_bytes).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    ranked = sorted(enumerate(proba), key=lambda x: -x[1])

    pred_idx  = ranked[0][0]
    pred_name = labels[pred_idx]
    pred_conf = ranked[0][1]
    return pred_name, pred_conf, ranked


def print_result(ct_bytes: bytes, model, labels: list,
                 top_n: int = 3, true_label: str = None):
    pred_name, pred_conf, ranked = predict_one(ct_bytes, model, labels, top_n)

    print(f"\n{'─'*55}")
    print(f"  Ciphertext length  : {len(ct_bytes)} bytes")
    if true_label:
        correct = "✓  CORRECT" if pred_name == true_label else "✗  WRONG"
        print(f"  True algorithm     : {true_label}")
        print(f"  Predicted          : {pred_name}  [{correct}]")
    else:
        print(f"  Predicted algorithm: {pred_name}  (confidence: {pred_conf*100:.1f}%)")

    print(f"\n  Top-{top_n} probabilities:")
    for rank, (idx, p) in enumerate(ranked[:top_n], 1):
        bar  = "█" * int(p * 32)
        mark = " ←" if rank == 1 else ""
        print(f"    {rank}. {labels[idx]:15s}  {p*100:6.2f}%  {bar}{mark}")
    print()


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo(model, labels, top_n):
    from encryptors import ENC_FUNCS, LABELS
    from Crypto.Random import get_random_bytes

    print("=" * 60)
    print("DEMO — encrypting with each algorithm, then predicting")
    print("=" * 60)

    correct = 0
    total   = len(LABELS)
    results = []

    for true_name, enc_fn in zip(LABELS, ENC_FUNCS):
        pt = get_random_bytes(128)
        ct = enc_fn(pt)
        pred_name, pred_conf, ranked = predict_one(ct, model, labels, top_n)
        ok = (pred_name == true_name)
        if ok:
            correct += 1
        results.append((true_name, pred_name, pred_conf, ok, len(ct), ranked))
        print_result(ct, model, labels, top_n=top_n, true_label=true_name)

    # Summary
    print("=" * 60)
    print(f"DEMO SUMMARY  —  {correct}/{total} correct  ({correct/total*100:.0f}%)")
    print("=" * 60)
    print(f"\n  {'Algorithm':<16}  {'Predicted':<16}  {'Conf':>6}  {'CT len':>7}  Result")
    print(f"  {'─'*16}  {'─'*16}  {'─'*6}  {'─'*7}  ──────")
    for true_name, pred_name, conf, ok, ct_len, _ in results:
        mark = "✓" if ok else "✗"
        print(f"  {true_name:<16}  {pred_name:<16}  {conf*100:5.1f}%  {ct_len:7d}  {mark}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    default_model = os.path.join(os.path.dirname(__file__), "model.pkl")

    p = argparse.ArgumentParser(description="PQC cipher algorithm predictor")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo",  action="store_true", help="Run demo on all algorithms")
    g.add_argument("--hex",   help="Hex-encoded ciphertext string")
    g.add_argument("--file",  help="Path to binary ciphertext file")
    g.add_argument("--dir",   help="Directory of binary ciphertext files")
    p.add_argument("--model", default=default_model, help="Path to model.pkl")
    p.add_argument("--top",   type=int, default=3,   help="Show top-N predictions")
    args = p.parse_args()

    model, labels, meta = load_model(args.model)
    print(f"Model loaded  : {args.model}")
    print(f"Trained on    : {len(labels)} algorithms")
    print(f"Test accuracy : {meta.get('accuracy', '?') * 100:.2f}%"
          if isinstance(meta.get('accuracy'), float) else "")
    print(f"CV accuracy   : {meta.get('cv_mean', 0)*100:.2f}% "
          f"± {meta.get('cv_std', 0)*100:.2f}%")

    if args.demo:
        run_demo(model, labels, args.top)

    elif args.hex:
        try:
            ct = bytes.fromhex(args.hex.replace(" ", "").replace("\n", ""))
        except ValueError as e:
            print(f"ERROR: invalid hex string — {e}")
            sys.exit(1)
        print_result(ct, model, labels, top_n=args.top)

    elif args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "rb") as fh:
            ct = fh.read()
        print_result(ct, model, labels, top_n=args.top)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERROR: not a directory: {args.dir}")
            sys.exit(1)
        files = sorted(f for f in os.listdir(args.dir)
                       if os.path.isfile(os.path.join(args.dir, f)))
        if not files:
            print("No files found in directory.")
            sys.exit(1)
        for fname in files:
            fpath = os.path.join(args.dir, fname)
            with open(fpath, "rb") as fh:
                ct = fh.read()
            pred_name, pred_conf, ranked = predict_one(ct, model, labels, args.top)
            print(f"  {fname:<30s}  →  {pred_name:<16s}  ({pred_conf*100:.1f}%)")


if __name__ == "__main__":
    main()
