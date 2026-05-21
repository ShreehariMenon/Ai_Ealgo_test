# PQC Cipher Classifier — Installation & Usage Guide
## Ubuntu PC Setup (Step-by-Step)

---

## Step 1 — Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip cmake ninja-build libssl-dev git build-essential
```

---

## Step 2 — Install Python packages

```bash
pip3 install pycryptodome scikit-learn numpy liboqs-python --break-system-packages
```

> If `--break-system-packages` causes an error, omit it:
> `pip3 install pycryptodome scikit-learn numpy liboqs-python`

---

## Step 3 — Build liboqs (the C library for ML-KEM)

`liboqs-python` is only a wrapper. The actual ML-KEM implementation is in
the liboqs C library, which must be compiled from source.

```bash
# Clone liboqs
git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs_src

# Configure (shared library, no tests)
cmake -S /tmp/liboqs_src -B /tmp/liboqs_build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DOQS_BUILD_ONLY_LIB=ON

# Build  (takes 2–5 minutes)
cmake --build /tmp/liboqs_build --parallel $(nproc)

# Install
sudo cmake --install /tmp/liboqs_build
sudo ldconfig
```

### Verify liboqs works

```bash
python3 -c "
import oqs
kem = oqs.KeyEncapsulation('ML-KEM-512')
pk  = kem.generate_keypair()
ct, ss = kem.encap_secret(pk)
print('ML-KEM-512  ct_len:', len(ct))   # should print 768
kem.free()
"
```

Expected output:
```
ML-KEM-512  ct_len: 768
```

---

## Step 4 — Put project files in one folder

```
~/pqc_classifier/
├── encryptors.py
├── features.py
├── generate_dataset.py
├── train.py
└── predict.py
```

```bash
mkdir ~/pqc_classifier
cd ~/pqc_classifier
# Copy the five .py files here
```

---

## Step 5 — Generate the dataset

```bash
cd ~/pqc_classifier
python3 generate_dataset.py --samples 2000
```

This creates `dataset.npz` with 16 000 labeled ciphertext samples
(2000 × 8 algorithms).  Takes about **3–5 minutes**.

Increase `--samples` for higher accuracy (3000–5000 recommended).

---

## Step 6 — Train the model

```bash
python3 train.py --data dataset.npz --model model.pkl
```

Takes about **1–2 minutes**.  Prints per-class accuracy and confusion matrix.
Saves the trained model to `model.pkl`.

---

## Step 7 — Run predictions

### Demo (tests all 8 algorithms automatically)

```bash
python3 predict.py --demo
```

### Predict from a hex-encoded ciphertext

```bash
python3 predict.py --hex a3f09c2b1d7e...
```

### Predict from a binary ciphertext file

```bash
python3 predict.py --file /path/to/ciphertext.bin
```

### Batch predict a folder of ciphertext files

```bash
python3 predict.py --dir /path/to/folder/
```

---

## Expected Results

| Algorithm     | Expected accuracy |
|---------------|-------------------|
| ML-KEM-512    | ~100%  (fixed 768-byte output)   |
| ML-KEM-768    | ~100%  (fixed 1088-byte output)  |
| ML-KEM-1024   | ~100%  (fixed 1568-byte output)  |
| AES-128-GCM   | ~95%+  (fixed 28-byte overhead)  |
| AES-256-GCM   | ~95%+  (fixed 28-byte overhead)  |
| AES-128-CBC   | ~90%+  (16-byte IV prefix)       |
| AES-256-CBC   | ~90%+  (16-byte IV prefix)       |
| AES-128-ECB   | ~70%   (confused with CBC)       |

**Overall: ~90–93% accuracy (ciphertext-only)**

---

## Troubleshooting

**`No oqs shared libraries found`**
→ liboqs C library not built/installed. Redo Step 3.

**`cmake: not found`**
→ Run `sudo apt install cmake`

**`ModuleNotFoundError: No module named 'Crypto'`**
→ Run `pip3 install pycryptodome`

**`sudo ldconfig` complains**
→ Try `sudo ldconfig /usr/local/lib` or check that liboqs.so was installed to `/usr/local/lib/`
