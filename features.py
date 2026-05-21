"""
features.py
===========
Converts a raw ciphertext blob into a fixed-length numeric feature vector
for the classifier.

Design rationale
----------------
Ciphertext-only classification is fundamentally limited by the fact that
every secure cipher produces output that is statistically close to uniform
random bytes.  The discriminating signals come from STRUCTURE, not content:

  1. Length & length-derived signals
       - ML-KEM ciphertexts are fixed-length (768 / 1088 / 1568 bytes).
       - AES-GCM always adds exactly 28 bytes overhead (12 nonce + 16 tag).
       - AES-CBC always prepends a 16-byte IV → length is always a multiple
         of 16, and always ≥ 16 bytes longer than the unpadded plaintext.
       - AES-ECB has no IV → length is a multiple of 16 with no extra prefix.
       - AES-GCM body length == pt_len (no block padding) → ct_len - 28
         can be any value, NOT necessarily a multiple of 16.

  2. Block-repetition (ECB weakness)
       When plaintext contains repeating 16-byte blocks, AES-ECB will
       produce repeated ciphertext blocks.  All other modes will not.

  3. Byte-distribution statistics
       All ciphers produce near-uniform bytes, but compressed polynomial
       coefficients in ML-KEM produce a subtly different distribution
       compared to AES keystream output.

  4. Regional statistics
       The first N bytes (IV / nonce region), the last 16 bytes (GCM tag
       region), and the body carry different statistical fingerprints.

Feature vector layout (total: 339 features)
--------------------------------------------
  [0]        raw length
  [1–5]      length modulo 8, 16, 32, 64, 128
  [6–12]     binary flags: block-aligned?, IV-16 offset?, GCM-28 overhead?,
             ML-KEM-512/768/1024 exact length?, etc.
  [13–268]   256-bin byte histogram (normalised)
  [269–277]  statistical moments (mean, std, min, max, median, p25, p75, p10, p90)
  [278]      Shannon entropy
  [279]      bigram entropy
  [280]      chi-squared vs uniform
  [281–287]  autocorrelation at lags 1,2,4,8,16,32,64
  [288–303]  block repetition features (8-byte and 16-byte blocks)
  [304–317]  regional statistics (prefix-8, prefix-12, prefix-16, tail-16)
  [318–321]  byte-difference stats
  [322–325]  byte quartile populations
  [326]      serial (adjacent-byte) correlation
  [327–330]  run-length stats
  [331–338]  nibble-level entropy, MSB balance, diff-direction balance
"""

import numpy as np
from collections import Counter


def extract(ct_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(ct_bytes, dtype=np.uint8)
    n   = len(arr)
    f   = []                        # feature list — extended with .append / .extend

    # ── 1. Length & structural flags ─────────────────────────────────────────
    f.append(float(n))
    for mod in [8, 16, 32, 64, 128]:
        f.append(float(n % mod))

    # Binary alignment / format flags
    f.append(float(n % 16 == 0))               # AES block-aligned (ECB or CBC body)
    f.append(float(n % 8  == 0))               # 8-byte block aligned (3DES / Blowfish)
    f.append(float((n - 16) % 16 == 0))        # CBC format: 16-byte IV + padded body
    f.append(float((n - 28) >= 0 and
                   (n - 28) % 1 == 0))         # GCM format: any body length is valid

    # ML-KEM exact-length flags (fixed, deterministic)
    f.append(float(n == 768))                  # ML-KEM-512
    f.append(float(n == 1088))                 # ML-KEM-768
    f.append(float(n == 1568))                 # ML-KEM-1024
    f.append(float(n >= 768))                  # PQC territory
    f.append(float(n < 512))                   # definitely not ML-KEM

    # ── 2. Byte histogram (256 bins, normalised) ──────────────────────────────
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist_f  = hist.astype(np.float64) / n
    f.extend(hist_f.tolist())                  # 256 features

    # ── 3. Statistical moments ───────────────────────────────────────────────
    mean_v = float(np.mean(arr))
    std_v  = float(np.std(arr))
    f += [
        mean_v, std_v,
        float(np.min(arr)), float(np.max(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 90)),
    ]

    # ── 4. Entropy measures ──────────────────────────────────────────────────
    p = hist_f[hist_f > 0]
    f.append(float(-np.sum(p * np.log2(p))))   # Shannon entropy (bits)

    # Bigram entropy
    if n > 1:
        pairs = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
        ph    = np.bincount(pairs, minlength=65536).astype(np.float64) / (n - 1)
        pp    = ph[ph > 0]
        f.append(float(-np.sum(pp * np.log2(pp))))
    else:
        f.append(0.0)

    # Chi-squared vs uniform
    expected = n / 256.0
    f.append(float(np.sum((hist - expected) ** 2 / expected)))

    # ── 5. Autocorrelation at multiple lags ──────────────────────────────────
    var_a = float(np.var(arr)) + 1e-9
    for lag in [1, 2, 4, 8, 16, 32, 64]:
        if n > lag:
            ac = float(
                np.mean((arr[lag:].astype(np.float64) - mean_v) *
                        (arr[:-lag].astype(np.float64) - mean_v)) / var_a
            )
        else:
            ac = 0.0
        f.append(ac)

    # ── 6. Block repetition (ECB detection) ──────────────────────────────────
    for bsz in [8, 16]:
        blocks = [bytes(ct_bytes[i: i + bsz]) for i in range(0, n - bsz + 1, bsz)]
        if blocks:
            nb    = len(blocks)
            cnts  = Counter(blocks)
            uniq  = len(cnts)
            f.append(float(uniq / nb))                          # unique-block ratio
            f.append(float(nb))                                 # total blocks
            f.append(float(max(cnts.values())))                 # max repetitions
            f.append(float(sum(1 for v in cnts.values() if v > 1) / nb))  # dup fraction
            # inter-block mean / std
            blk_means = [float(np.mean(arr[i: i + bsz])) for i in range(0, n - bsz + 1, bsz)]
            blk_stds  = [float(np.std( arr[i: i + bsz])) for i in range(0, n - bsz + 1, bsz)]
            f.append(float(np.mean(blk_means)))
            f.append(float(np.std(blk_means)))
            f.append(float(np.mean(blk_stds)))
            f.append(float(np.std(blk_stds)))
        else:
            f.extend([0.0] * 8)

    # ── 7. Regional statistics ────────────────────────────────────────────────
    def region_stats(region: np.ndarray) -> list:
        if len(region) == 0:
            return [0.0, 0.0, 0.0]
        rh, _ = np.histogram(region, bins=16, range=(0, 256))
        rhf   = rh.astype(np.float64) / len(region)
        rhp   = rhf[rhf > 0]
        ent   = float(-np.sum(rhp * np.log2(rhp))) if len(rhp) > 0 else 0.0
        return [float(np.mean(region)), float(np.std(region)), ent]

    # prefix-8  (3DES / Blowfish IV candidate)
    f.extend(region_stats(arr[:min(8, n)]))
    # prefix-12 (GCM nonce / ChaCha20 nonce candidate)
    f.extend(region_stats(arr[:min(12, n)]))
    # prefix-16 (AES-CBC IV candidate / AES-ECB first block)
    f.extend(region_stats(arr[:min(16, n)]))
    # tail-16   (AES-GCM authentication tag candidate)
    f.extend(region_stats(arr[max(0, n - 16):]))

    # ── 8. Byte-difference statistics ────────────────────────────────────────
    diff = np.diff(arr.astype(np.int16))
    f.append(float(np.mean(np.abs(diff))))
    f.append(float(np.std(diff)))
    f.append(float(np.mean(diff > 0)))     # fraction of rising transitions
    f.append(float(np.sum(diff == 0) / max(1, len(diff))))  # fraction identical

    # ── 9. Byte quartile populations ─────────────────────────────────────────
    for lo_b, hi_b in [(0, 64), (64, 128), (128, 192), (192, 256)]:
        f.append(float(np.sum((arr >= lo_b) & (arr < hi_b)) / n))

    # ── 10. Serial (adjacent-byte) correlation ────────────────────────────────
    if n > 2:
        sc = np.corrcoef(arr[:-1].astype(np.float64),
                         arr[1:].astype(np.float64))[0, 1]
        f.append(float(sc) if np.isfinite(sc) else 0.0)
    else:
        f.append(0.0)

    # ── 11. Run-length encoding stats ─────────────────────────────────────────
    runs = []
    cur  = 1
    for i in range(1, min(n, 1024)):
        if arr[i] == arr[i - 1]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    f.append(float(np.mean(runs)))
    f.append(float(np.max(runs)))
    f.append(float(np.std(runs)))
    f.append(float(sum(1 for r in runs if r > 1) / max(1, len(runs))))

    # ── 12. Nibble entropy & misc ─────────────────────────────────────────────
    for nibble in [(arr & 0x0F), (arr >> 4)]:
        nh, _ = np.histogram(nibble, bins=16, range=(0, 16))
        nhf   = nh.astype(np.float64) / n
        nhp   = nhf[nhf > 0]
        f.append(float(-np.sum(nhp * np.log2(nhp))) if len(nhp) > 0 else 0.0)

    f.append(float(np.mean(arr > 127)))     # MSB balance
    f.append(float(np.mean(diff > 0)) if len(diff) > 0 else 0.0)  # rising-byte fraction

    return np.array(f, dtype=np.float32)


# Compute dimension once at import time
FEATURE_DIM = len(extract(b'\x00' * 256))


if __name__ == "__main__":
    import os, sys
    ct = os.urandom(768)
    vec = extract(ct)
    print(f"Feature vector dimension: {FEATURE_DIM}")
    print(f"Sample vector (first 10): {vec[:10]}")
