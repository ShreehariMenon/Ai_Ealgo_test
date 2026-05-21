# PQC Cipher Algorithm Classifier

A machine learning model that identifies which encryption algorithm was used to produce a given ciphertext — trained on real ML-KEM (post-quantum) and AES ciphertext samples.

Part of a larger **Post-Quantum Cryptography (PQC)** research project.

---

## Overview

Given only a raw ciphertext blob, this classifier predicts which of 8 encryption algorithms produced it:

| # | Algorithm | Type | Key Size |
|---|-----------|------|----------|
| 0 | ML-KEM-512 | Post-Quantum KEM (NIST PQC standard) | Lattice-based |
| 1 | ML-KEM-768 | Post-Quantum KEM | Lattice-based |
| 2 | ML-KEM-1024 | Post-Quantum KEM | Lattice-based |
| 3 | AES-128-CBC | Symmetric block cipher | 128-bit |
| 4 | AES-256-CBC | Symmetric block cipher | 256-bit |
| 5 | AES-128-GCM | Authenticated encryption | 128-bit |
| 6 | AES-256-GCM | Authenticated encryption | 256-bit |
| 7 | AES-128-ECB | Symmetric block cipher | 128-bit |

---

## Results

Trained on 16,000 samples (2,000 per class). Tested on 3,200 held-out samples.

```
Overall test accuracy  : 68.94%
5-fold CV accuracy     : 68.31% ± 0.56%
```

### Per-class breakdown

| Algorithm | Precision | Recall | F1 | Notes |
|-----------|-----------|--------|----|-------|
| ML-KEM-512 | **1.000** | **1.000** | **1.000** | Perfect — never misclassified |
| ML-KEM-768 | **1.000** | **1.000** | **1.000** | Perfect — never misclassified |
| ML-KEM-1024 | **1.000** | **1.000** | **1.000** | Perfect — never misclassified |
| AES-128-CBC | 0.392 | 0.460 | 0.423 | Confused with AES-256-CBC |
| AES-256-CBC | 0.408 | 0.560 | 0.472 | Confused with AES-128-CBC |
| AES-128-GCM | 0.487 | 0.497 | 0.492 | Confused with AES-256-GCM |
| AES-256-GCM | 0.493 | 0.422 | 0.455 | Confused with AES-128-GCM |
| AES-128-ECB | **1.000** | 0.575 | 0.730 | Partially confused with CBC variants |

### Confusion matrix

```
              ML-KEM-512  ML-KEM-768  ML-KEM-1024  AES-128-CBC  AES-256-CBC  AES-128-GCM  AES-256-GCM  AES-128-ECB
  ML-KEM-512         400           0            0            0            0            0            0            0
  ML-KEM-768           0         400            0            0            0            0            0            0
 ML-KEM-1024           0           0          400            0            0            0            0            0
 AES-128-CBC           0           0            0          184          216            0            0            0
 AES-256-CBC           0           0            0          176          224            0            0            0
 AES-128-GCM           0           0            0           15           12          199          174            0
 AES-256-GCM           0           0            0           12            9          210          169            0
 AES-128-ECB           0           0            0           82           88            0            0          230
```

### Key observation

The model **never confuses ML-KEM with any AES variant** — zero cross-group errors across all 1,200 ML-KEM test samples. The only confusion is within the AES family, between variants that share identical ciphertext structure (AES-128-CBC vs AES-256-CBC differ only in key size, which leaves no trace in the ciphertext bytes — a fundamental property of AES security).

---

## Why AES variants are hard to distinguish

AES-128-CBC and AES-256-CBC produce output with **identical format and identical statistical properties**. The key size (128 vs 256 bits) never appears in the ciphertext. Distinguishing them from ciphertext alone is computationally equivalent to breaking AES — which is by design. The same applies to the GCM pair. This is not a model limitation; it is the security guarantee of the cipher working as intended.

---

## Project structure

```
.
├── encryptors.py          # All 8 encryption functions (uses real liboqs for ML-KEM)
├── features.py            # Converts raw ciphertext bytes → 335-feature vector
├── generate_dataset.py    # Generates labeled ciphertext dataset → dataset.npz
├── train.py               # Trains Random Forest classifier → model.pkl
├── predict.py             # CLI inference tool
├── INSTALL.md             # Full setup guide for Ubuntu
└── README.md              # This file
```

---

## Installation

### Requirements

- Ubuntu 20.04+ (or any Linux with Python 3.8+)
- Python packages: `pycryptodome`, `scikit-learn`, `numpy`, `liboqs-python`
- liboqs C library (for real ML-KEM — must be compiled from source)

### Quick install

```bash
# System dependencies
sudo apt install -y cmake libssl-dev git build-essential

# Python packages
pip3 install pycryptodome scikit-learn numpy liboqs-python

# Build and install liboqs C library
git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs_src
cmake -S /tmp/liboqs_src -B /tmp/liboqs_build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON
cmake --build /tmp/liboqs_build --parallel $(nproc)
sudo cmake --install /tmp/liboqs_build && sudo ldconfig
```

See `INSTALL.md` for detailed step-by-step instructions and troubleshooting.

---

## Usage

### 1. Generate dataset

```bash
python3 generate_dataset.py --samples 2000
```

Creates `dataset.npz` with 16,000 labeled ciphertext samples. Takes ~3–5 minutes.

### 2. Train the model

```bash
python3 train.py --data dataset.npz --model model.pkl
```

Trains a 500-tree Random Forest. Takes ~1–2 minutes. Prints accuracy and confusion matrix.

### 3. Predict

```bash
# Demo — encrypts with each algorithm and predicts
python3 predict.py --demo

# From a hex-encoded ciphertext string
python3 predict.py --hex a3f09c2b1d7e...

# From a binary ciphertext file
python3 predict.py --file ciphertext.bin

# Batch predict a folder of ciphertext files
python3 predict.py --dir ./ciphertexts/
```

### Example output

```
  Ciphertext length  : 768 bytes
  True algorithm     : ML-KEM-512
  Predicted          : ML-KEM-512  [✓  CORRECT]

  Top-3 probabilities:
    1. ML-KEM-512        99.20%  ███████████████████████████████ ←
    2. ML-KEM-768         0.40%
    3. AES-256-CBC        0.40%
```

---

## How it works

### Feature extraction (335 features per ciphertext)

Each ciphertext is converted into a 335-dimensional feature vector before classification:

| Feature group | Count | What it captures |
|--------------|-------|-----------------|
| Length & structural flags | 13 | Exact-length flags for ML-KEM, block-alignment for AES modes |
| Byte histogram | 256 | Full 256-bin normalised byte frequency distribution |
| Statistical moments | 9 | Mean, std, min, max, median, percentiles |
| Entropy measures | 3 | Shannon entropy, bigram entropy, chi-squared vs uniform |
| Autocorrelation | 7 | At lags 1, 2, 4, 8, 16, 32, 64 |
| Block repetition | 16 | Repeated 8-byte and 16-byte blocks (AES-ECB detection) |
| Regional statistics | 12 | Prefix bytes (IV/nonce), tail bytes (GCM auth tag) |
| Byte-difference & run-length | 11 | Transition smoothness, consecutive byte patterns |
| Serial correlation & misc | 8 | Adjacent-byte correlation, MSB balance |

### Model

Random Forest with 500 decision trees. Chosen because the most discriminating features (exact length flags for ML-KEM, structural offset patterns for AES modes) are simple threshold rules that decision trees capture exactly, without needing feature scaling or normalisation.

---

## Limitations

- **AES-128 vs AES-256 variants cannot be distinguished from ciphertext alone.** Key size leaves no trace in the ciphertext. This is not fixable without additional side-channel or metadata information.
- The liboqs version warning (`liboqs version 0.15.0 differs from liboqs-python 0.14.1`) is cosmetic and harmless.
- The `RuntimeWarning: invalid value encountered in divide` from numpy is a known cosmetic warning in the serial correlation feature for near-zero-variance inputs. It does not affect results.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pycryptodome` | AES-CBC, AES-GCM, AES-ECB encryption |
| `liboqs-python` | ML-KEM-512/768/1024 via NIST liboqs |
| `scikit-learn` | Random Forest classifier |
| `numpy` | Feature computation and dataset handling |

---

## License

MIT
