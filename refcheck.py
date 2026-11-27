"""Fingerprint of a run's archive, for exact before/after comparisons: python refcheck.py <run>"""
import hashlib, pickle, sys, zlib
from pathlib import Path

saved = pickle.loads(zlib.decompress((Path("runs") / sys.argv[1] / "checkpoint.pkl.z").read_bytes()))
digest = hashlib.sha256()
for state, place in zip(saved["states"], saved["places"]):
    digest.update(state)
    digest.update(repr(place).encode())
print(sys.argv[1], len(saved["states"]), "saves", digest.hexdigest()[:16])
