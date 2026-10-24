"""Monte Carlo floor of the final-layer truth for networks of Keenan Pepper's corpus (N = 1e8 samples per network).

The corpus stores raw moments m1..m4 of every post-ReLU neuron at every layer. The truth's own expected squared error
at the last layer is mean_j (m2_j - m1_j^2) / N; an estimator's measured MSE against that truth is its true MSE plus
this floor, so paired.py can subtract it.

    python kp_floor.py --corpus <dir of shardNN.npz / shard_extNN.npz> --ids 3408,3409,... --out floors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def where(idx: int):
    if idx < 2048:
        return f"shard{idx % 16:02d}", idx // 16
    return f"shard_ext{(idx - 2048) % 8:02d}", (idx - 2048) // 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = [int(x) for x in a.ids.split(",")]
    out, cache = {}, {}
    for idx in want:
        name, row = where(idx)
        if name not in cache:
            cache[name] = np.load(Path(a.corpus) / f"{name}.npz")
        z = cache[name]
        assert int(z["idx"][row]) == idx, (idx, name, row)
        m1 = z["moments"][row, 0, 15].astype(np.float64)
        m2 = z["moments"][row, 1, 15].astype(np.float64)
        out[str(idx)] = float(np.maximum(m2 - m1 * m1, 0.0).mean() / float(z["n_samples"]))
    Path(a.out).write_text(json.dumps({"n": len(out), "mean_floor": float(np.mean(list(out.values()))), "floor": out}, indent=1),
                           encoding="utf-8")
    print(json.dumps({"n": len(out), "mean_floor": float(np.mean(list(out.values())))}))


if __name__ == "__main__":
    main()
