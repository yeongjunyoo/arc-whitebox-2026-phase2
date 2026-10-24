"""Pair two ablate.py outputs network by network and print the write-up's ratios.

Ratios are of means over the paired networks: adjusted score (MSE x max(0.1, C/B)), final-layer MSE and FLOPs, with a
paired bootstrap 95% interval for each (10,000 resamples of networks, seed 12345), the number of networks whose
adjusted score is lower than the base's, and how many predictions are bit-identical to the base's.

    python paired.py runs/variant_*.jsonl --base runs/submitted_*.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(paths):
    out = {}
    for p in paths:
        for x in Path(p).read_text(encoding="utf-8").splitlines():
            x = x.strip()
            if x.startswith("{") and '"mlp_id"' in x:
                r = json.loads(x)
                out[(r["set"], r["mlp_id"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", nargs="+")
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--floor", default=None, help="kp_floor.py output: subtract each network's Monte Carlo floor from "
                    "both arms' MSE before the ratios (kp networks only)")
    a = ap.parse_args()
    v, b = rows(a.variant), rows(a.base)
    keys = sorted(set(v) & set(b))
    assert keys, "no network in common"
    get = lambda d, f: np.array([d[k][f] for k in keys], dtype=np.float64)  # noqa: E731
    fl = np.zeros(len(keys))
    if a.floor:
        f_ = json.loads(Path(a.floor).read_text(encoding="utf-8"))["floor"]
        fl = np.array([f_[str(k[1])] if k[0] == "kp" else 0.0 for k in keys])
    C1, C0 = get(v, "C_over_B"), get(b, "C_over_B")
    M1, M0 = get(v, "final_mse") - fl, get(b, "final_mse") - fl
    A1, A0 = M1 * np.maximum(0.1, C1), M0 * np.maximum(0.1, C0)
    F1, F0 = get(v, "flops"), get(b, "flops")
    rng = np.random.default_rng(12345)
    J = rng.integers(0, len(keys), (10000, len(keys)))

    def ratio(x1, x0):
        bs = x1[J].mean(1) / x0[J].mean(1)
        return float(x1.mean() / x0.mean()), [float(q) for q in np.quantile(bs, [0.025, 0.975])]

    res = {"n": len(keys), "sets": sorted({k[0] for k in keys}), "floor_subtracted": bool(a.floor), "mean_floor": float(fl.mean()),
           "adjusted_ratio": ratio(A1, A0), "mse_ratio": ratio(M1, M0), "flops_ratio": ratio(F1, F0),
           "better": int((A1 < A0).sum()), "bit_identical": int(sum(v[k]["pred_sha16"] == b[k]["pred_sha16"] for k in keys)),
           "variant_mean_adjusted": float(A1.mean()), "base_mean_adjusted": float(A0.mean()),
           "variant_mean_c_over_b": float(np.mean(get(v, "C_over_B"))), "max_wall_s": float(get(v, "wall_s").max()),
           "all_finite": all(v[k]["finite"] for k in keys)}
    print(json.dumps(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
