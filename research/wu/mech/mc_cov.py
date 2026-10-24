"""Monte Carlo truth of the activation covariance C_t[i, j] = Cov(h_t,i, h_t,j) (h_t = relu(z_t), the engine's C' of
layer t) at layers 2..14 of public mini networks (training pool only; held-out rows 60-99 are refused), for the band
study's local test (research/wu/mech): which setting of a layer's dropped 3 T^T entries brings that layer's C' closest
to the truth when everything upstream is the same.

Sampling as mc_truth.py (the organizers' x ~ N(0, I_n), float32 forward, float64 accumulators): per batch of B rows the
unbiased covariance with the batch's own means; estimate = mean over batches, standard error = batch-means one.

    .venv-fs0123/bin/python research/wu/mech/mc_cov.py --id 0 --n 10000000 --batch 50000 --out research/wu/mech/mc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # research/wu: ablate.load_rows
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"
LAYERS = list(range(2, 15))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert 0 <= a.id < 60, "held-out mini rows 60-99 are not read"
    assert a.batch <= 100_000 and a.n % a.batch == 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (mid, seed, name, W, truth), = list(AB.load_rows(SNAP, "mini", [a.id]))
    L, n = W.shape[0], W.shape[1]
    B, nb = a.batch, a.n // a.batch
    rng = np.random.default_rng([a.seed, mid])
    S1 = {t: np.zeros((n, n)) for t in LAYERS}
    S2 = {t: np.zeros((n, n)) for t in LAYERS}
    hm = np.zeros((L, n))
    t0 = time.time()
    for b in range(nb):
        h = rng.standard_normal((B, n), dtype=np.float32)
        for t in range(L):
            h = np.maximum(h @ W[t], np.float32(0.0))
            m = h.mean(axis=0, dtype=np.float64)
            hm[t] += m
            if t in S1:
                hc = h - m.astype(np.float32)
                cb = (hc.T @ hc).astype(np.float64) / (B - 1.0)
                S1[t] += cb
                S2[t] += cb * cb
                del hc
        if (b + 1) % 10 == 0 or b == 0:
            print(json.dumps({"id": mid, "batch": b + 1, "of": nb, "sec": round(time.time() - t0, 1)}), flush=True)
    arr = {"act_mean": hm / nb, "layers": np.array(LAYERS)}
    for t in LAYERS:
        mu = S1[t] / nb
        arr[f"C_{t}"] = mu.astype(np.float32)
        arr[f"CSE_{t}"] = np.sqrt(np.maximum(S2[t] / nb - mu * mu, 0.0) / (nb - 1)).astype(np.float32)
    np.savez(out / f"mccov_mini{mid}.npz", **arr)
    tr = truth.astype(np.float64)
    rec = {"id": mid, "name": name, "seed_rng": [a.seed, mid], "n": a.n, "batch": B, "batches": nb,
           "sec": round(time.time() - t0, 1),
           "act_mean_vs_dataset_mse": [float(((arr["act_mean"][t] - tr[t]) ** 2).mean()) for t in range(L)]}
    (out / f"mccov_mini{mid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("id", "sec")}), flush=True)


if __name__ == "__main__":
    main()
