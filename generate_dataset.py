"""
generate_dataset.py
===================
Generates the training / test dataset.

Usage
-----
    python generate_dataset.py                  # default 2000 samples/class
    python generate_dataset.py --samples 3000   # custom count
    python generate_dataset.py --out data/      # custom output directory

Output
------
    dataset.npz  — compressed numpy archive with keys:
                     X       float32 (N, FEATURE_DIM)
                     y       int32   (N,)
                     labels  object  list of class names

Plaintext strategy
------------------
For ML-KEM: plaintext is ignored (KEM ciphertext is the shared-key blob).
For AES:    we use a 50/50 mix of:
              - fully random bytes          → tests statistical features
              - structured repeating blocks → exposes ECB block repetition
"""

import os
import sys
import argparse
import numpy as np
from Crypto.Random import get_random_bytes

# local modules
sys.path.insert(0, os.path.dirname(__file__))
from encryptors import ENC_FUNCS, LABELS, NUM_CLASSES
from features   import extract, FEATURE_DIM


# ── Plaintext generation ──────────────────────────────────────────────────────

def make_plaintext() -> bytes:
    """
    Returns bytes of random length [64, 256].
    50% chance: purely random bytes.
    50% chance: structured (repeating 16-byte blocks) to stress-test ECB detection.
    """
    import random
    length = random.randint(64, 256)
    if random.random() < 0.5:
        return get_random_bytes(length)
    else:
        # Build structured plaintext with repeating blocks
        block    = get_random_bytes(16)
        repeats  = random.randint(2, length // 16 + 1)
        raw      = (block * repeats)[:length]
        # Pad to exact length with random tail
        tail_len = length - len(raw)
        return raw + (get_random_bytes(tail_len) if tail_len > 0 else b'')


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(samples_per_class: int = 2000, verbose: bool = True) -> tuple:
    """
    Returns (X, y) numpy arrays.
    """
    X_list, y_list = [], []

    for label_idx, (name, enc_fn) in enumerate(zip(LABELS, ENC_FUNCS)):
        if verbose:
            print(f"  [{label_idx+1}/{NUM_CLASSES}]  {name:15s} — generating {samples_per_class} samples …",
                  flush=True)
        for _ in range(samples_per_class):
            pt = make_plaintext()
            ct = enc_fn(pt)
            X_list.append(extract(ct))
            y_list.append(label_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Generate cipher classification dataset")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Samples per class (default: 2000)")
    parser.add_argument("--out", default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "dataset.npz")

    print("=" * 60)
    print("PQC Cipher Classifier  —  Dataset Generation")
    print("=" * 60)
    print(f"  Algorithms      : {NUM_CLASSES}")
    print(f"  Samples/class   : {args.samples}")
    print(f"  Total samples   : {NUM_CLASSES * args.samples}")
    print(f"  Feature dim     : {FEATURE_DIM}")
    print(f"  Output          : {out_path}")
    print()

    X, y = generate(samples_per_class=args.samples)

    np.savez_compressed(out_path, X=X, y=y, labels=np.array(LABELS, dtype=object))
    print(f"\nDataset saved → {out_path}")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {dict(zip(LABELS, [int(np.sum(y == i)) for i in range(NUM_CLASSES)]))}")


if __name__ == "__main__":
    main()
