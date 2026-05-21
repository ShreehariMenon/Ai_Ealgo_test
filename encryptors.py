"""
encryptors.py
=============
Implements all 8 target algorithms.
ML-KEM uses real liboqs when available; raises a clear error if not installed.

Algorithms
----------
  0  ML-KEM-512
  1  ML-KEM-768
  2  ML-KEM-1024
  3  AES-128-CBC
  4  AES-256-CBC
  5  AES-128-GCM
  6  AES-256-GCM
  7  AES-128-ECB

Each encrypt_*(plaintext: bytes) -> bytes function returns the raw ciphertext
blob that will be fed to the classifier (no key, no label).

ML-KEM note
-----------
ML-KEM is a Key Encapsulation Mechanism (KEM), not a bulk cipher.
  - A fresh keypair is generated per sample (as in a real handshake).
  - encap_secret(public_key) returns (ciphertext, shared_secret).
  - The ciphertext is what we classify; the shared_secret is discarded.
  - plaintext bytes are NOT directly encrypted by ML-KEM here; the
    KEM ciphertext carries the encapsulated key, which is the artifact
    we want the model to recognise.
"""

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

# ── ML-KEM via liboqs ─────────────────────────────────────────────────────────

def _get_oqs():
    try:
        import oqs
        return oqs
    except ImportError:
        raise ImportError(
            "liboqs-python is not installed.\n"
            "Run:  pip install liboqs-python\n"
            "liboqs (the C library) must also be built — see INSTALL.md."
        )

def encrypt_mlkem(variant: str) -> bytes:
    """
    variant: 'ML-KEM-512' | 'ML-KEM-768' | 'ML-KEM-1024'
    Returns the KEM ciphertext (encapsulated key blob).
    """
    oqs = _get_oqs()
    kem = oqs.KeyEncapsulation(variant)
    public_key = kem.generate_keypair()
    ciphertext, _shared_secret = kem.encap_secret(public_key)
    kem.free()
    return bytes(ciphertext)

def encrypt_mlkem512(_pt=None)  -> bytes: return encrypt_mlkem('ML-KEM-512')
def encrypt_mlkem768(_pt=None)  -> bytes: return encrypt_mlkem('ML-KEM-768')
def encrypt_mlkem1024(_pt=None) -> bytes: return encrypt_mlkem('ML-KEM-1024')

# ── AES variants ──────────────────────────────────────────────────────────────

def encrypt_aes128_cbc(pt: bytes) -> bytes:
    key = get_random_bytes(16)
    iv  = get_random_bytes(16)
    ct  = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(pt, 16))
    return iv + ct                          # 16-byte IV prepended

def encrypt_aes256_cbc(pt: bytes) -> bytes:
    key = get_random_bytes(32)
    iv  = get_random_bytes(16)
    ct  = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(pt, 16))
    return iv + ct

def encrypt_aes128_gcm(pt: bytes) -> bytes:
    key   = get_random_bytes(16)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(pt)
    return nonce + ct + tag                 # 12-byte nonce | ct | 16-byte tag

def encrypt_aes256_gcm(pt: bytes) -> bytes:
    key   = get_random_bytes(32)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(pt)
    return nonce + ct + tag

def encrypt_aes128_ecb(pt: bytes) -> bytes:
    key = get_random_bytes(16)
    return AES.new(key, AES.MODE_ECB).encrypt(pad(pt, 16))

# ── Registry ──────────────────────────────────────────────────────────────────

ALGORITHMS = [
    ("ML-KEM-512",   encrypt_mlkem512),
    ("ML-KEM-768",   encrypt_mlkem768),
    ("ML-KEM-1024",  encrypt_mlkem1024),
    ("AES-128-CBC",  encrypt_aes128_cbc),
    ("AES-256-CBC",  encrypt_aes256_cbc),
    ("AES-128-GCM",  encrypt_aes128_gcm),
    ("AES-256-GCM",  encrypt_aes256_gcm),
    ("AES-128-ECB",  encrypt_aes128_ecb),
]

LABELS    = [name for name, _ in ALGORITHMS]
ENC_FUNCS = [fn   for _, fn  in ALGORITHMS]
NUM_CLASSES = len(ALGORITHMS)

# Expected ciphertext lengths for ML-KEM (fixed, deterministic)
MLKEM_CT_LENGTHS = {
    "ML-KEM-512":  768,
    "ML-KEM-768":  1088,
    "ML-KEM-1024": 1568,
}


if __name__ == "__main__":
    from Crypto.Random import get_random_bytes as grb
    pt = grb(128)
    for name, fn in ALGORITHMS:
        ct = fn(pt)
        print(f"  {name:15s}  ct_len={len(ct)}")
