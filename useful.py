import pickle, sys, zlib
from pathlib import Path
import numpy as np
if __name__ == "__main__":
    for run in sys.argv[1:]:
        saved = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))
        productive = np.sort(np.array(saved["productive"], float))[::-1]
        chosen = np.array(saved["times_chosen"], float)
        total = productive.sum()
        share = lambda k: productive[:k].sum() / total
        n = len(productive)
        print(f"{run}: {n} saves, {int(total)} productive tries, {int(chosen.sum())} tries in all; "
              f"saves that ever produced anything: {(productive > 0).sum()} ({(productive > 0).mean():.0%})")
        print("   share of productive tries from the top k saves:",
              ", ".join(f"top {k} ({k / n:.2%}): {share(k):.0%}" for k in (10, 100, 1000, 10000) if k < n))
        print("   saves needed for 50% / 90% / 99%:", [int(np.searchsorted(np.cumsum(productive) / total, q)) + 1 for q in (0.5, 0.9, 0.99)])
