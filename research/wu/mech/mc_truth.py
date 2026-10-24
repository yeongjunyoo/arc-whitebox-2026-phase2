"""Monte Carlo truth of the preactivation third-cumulant slice that the engine's 3 T^T product carries, on public mini
networks (training pool only; held-out rows 60-99 are refused).

Inputs are drawn as the organizers' sampler draws them (whestbench/simulation.py sample_layer_statistics: x ~ N(0, I_n),
float32, h_t = relu(h_{t-1} @ W_t)). The forward pass is float32; every accumulator is float64.

At every layer t the preactivation is z_t = h_{t-1} @ W_t (h_{-1} = x). For layers 2..14 this script estimates

    TT[i, j] = kappa3(z_i, z_j, z_j)       (row i: the single index, column j: the doubled index)

which is the engine's M1 / 3 off the diagonal (M1 = 3 T^T, T[i, j] = kappa3(z_i, z_i, z_j)), and at every layer the
diagonal t3_i = kappa3(z_i, z_i, z_i), the preactivation mean and variance, and the activation mean.

Estimator: the samples are split into batches of B rows; each batch gives the unbiased k-statistic
k_b[i, j] = B / ((B - 1)(B - 2)) sum_s (z_si - zbar_i)(z_sj - zbar_j)^2 with the batch's own means; the estimate is the
mean over batches and its standard error the batch-means one, std_b(k_b) / sqrt(nb).

    .venv-fs0123/bin/python research/wu/mech/mc_truth.py --id 0 --n 10000000 --batch 50000 --out research/wu/mech/mc
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
LAYERS = list(range(2, 15))  # the layers whose 3 T^T the band can restrict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert 0 <= a.id < 60, "held-out mini rows 60-99 are not read"
    assert a.batch <= 100_000 and a.n % a.batch == 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (mid, seed, name, W, truth), = list(AB.load_rows(SNAP, "mini", [a.id]))
    L, n = W.shape[0], W.shape[1]
    B, nb = a.batch, a.n // a.batch
    kf3 = B / ((B - 1.0) * (B - 2.0))  # k-statistic factor (third order)
    rng = np.random.default_rng([a.seed, mid])
    S1 = {t: np.zeros((n, n)) for t in LAYERS}  # sum over batches of k_b
    S2 = {t: np.zeros((n, n)) for t in LAYERS}  # sum over batches of k_b^2
    d1 = np.zeros((L, n)); d2 = np.zeros((L, n))  # t3: sum and sum of squares of the batch k-statistics
    zm = np.zeros((L, n)); zq = np.zeros((L, n))  # sum of z, sum of z^2 (float64)
    hm = np.zeros((L, n))  # sum of relu(z)
    sq = np.empty((B, n), dtype=np.float32)  # work buffer: squares of the centered z
    cu = np.empty((B, n), dtype=np.float32)  # work buffer: cubes of the centered z
    t0 = time.time()
    for b in range(nb):
        h = rng.standard_normal((B, n), dtype=np.float32)
        for t in range(L):
            z = h @ W[t]  # float32, as the organizers' forward pass
            h = np.maximum(z, np.float32(0.0))
            hm[t] += h.sum(axis=0, dtype=np.float64)
            zm[t] += z.sum(axis=0, dtype=np.float64)
            m = z.mean(axis=0, dtype=np.float64)
            z -= m.astype(np.float32)  # centered on the batch mean (float32 data, float64 mean)
            np.multiply(z, z, out=sq)
            zq[t] += sq.sum(axis=0, dtype=np.float64) + B * m * m  # sum z^2 = sum zc^2 + B m^2
            np.multiply(sq, z, out=cu)
            c3 = cu.sum(axis=0, dtype=np.float64) * kf3
            d1[t] += c3
            d2[t] += c3 * c3
            if t in S1:
                kb = (z.T @ sq).astype(np.float64)  # [i, j] = sum_s zc_si zc_sj^2 (float32 product of one batch)
                kb *= kf3
                S1[t] += kb
                S2[t] += kb * kb
            del z
        if (b + 1) % 10 == 0 or b == 0:
            print(json.dumps({"id": mid, "batch": b + 1, "of": nb, "sec": round(time.time() - t0, 1)}), flush=True)
    N = float(a.n)
    mean_z = zm / N
    var_z = zq / N - mean_z * mean_z
    act = hm / N
    t3 = d1 / nb
    t3_se = np.sqrt(np.maximum(d2 / nb - t3 * t3, 0.0) / (nb - 1))
    arr = {"mean_z": mean_z, "var_z": var_z, "act_mean": act, "t3": t3, "t3_se": t3_se, "layers": np.array(LAYERS)}
    for t in LAYERS:
        mu = S1[t] / nb
        arr[f"TT_{t}"] = mu.astype(np.float32)
        arr[f"SE_{t}"] = np.sqrt(np.maximum(S2[t] / nb - mu * mu, 0.0) / (nb - 1)).astype(np.float32)
    np.savez(out / f"mc_mini{mid}.npz", **arr)
    tr = truth.astype(np.float64)
    rec = {"id": mid, "name": name, "seed_rng": [a.seed, mid], "n": a.n, "batch": B, "batches": nb,
           "sec": round(time.time() - t0, 1),
           "act_mean_vs_dataset_mse": [float(((act[t] - tr[t]) ** 2).mean()) for t in range(L)],
           "dataset_mean_sq": [float((tr[t] ** 2).mean()) for t in range(L)]}
    (out / f"mc_mini{mid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("id", "sec")}), flush=True)


if __name__ == "__main__":
    main()
