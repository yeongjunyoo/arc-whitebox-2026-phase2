"""Refit the per-layer law table lam'_t of estimators/k3_acc_aug.py around a base table from base-relative
finite-difference Jacobian runs of research/acc_feed_scan.py ({"jacbase": base, "rel": r}: setting 0 = base,
setting t+1 = base with lam'_t scaled by 1 + r, t = 0..14). Linearized least squares over all neurons of the fit MLPs,
optional ridge toward the base table, leave-one-MLP-out check, writes the refitted table.

    whest-starterkit/.venv/Scripts/python.exe research/acc_aug_fit.py --tag aug_jac --rel 0.3
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="aug_jac")
    ap.add_argument("--rel", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=0.0, help="penalty on (delta lam / base lam)^2, per neuron-sum units")
    ap.add_argument("--nlam", type=int, default=15)
    ap.add_argument("--groups", default="", help='JSON list of layer groups, one per perturbed setting, e.g. "[[0,1,2],[3,4,5]]"')
    args = ap.parse_args()
    groups = json.loads(args.groups) if args.groups else [[t] for t in range(args.nlam)]
    args.nlam = len(groups)
    z = dict(np.load(HERE / f"acc_feedscan_{args.tag}.npz"))
    base = np.array(z["lam_0"], dtype=np.float64)
    mlps = sorted({int(m.group(1)) for k in z for m in [re.match(r"pred_(\d+)_0$", k)] if m})
    data = {}
    for i in mlps:
        keys = [f"pred_{i}_{k}" for k in range(args.nlam + 1)]
        if not all(k in z for k in keys):
            continue
        p0 = z[keys[0]]
        # derivative w.r.t. the relative change x_t = dlam_t / base_t
        J = np.stack([(z[keys[t + 1]] - p0) / args.rel for t in range(args.nlam)], axis=1)
        data[i] = (p0, J, z[f"gt_{i}"])
    print("fit MLPs:", list(data))

    def fit(ids):
        A = np.concatenate([data[i][1] for i in ids])
        b = np.concatenate([data[i][2] - data[i][0] for i in ids])
        if args.ridge:
            A = np.vstack([A, np.sqrt(args.ridge) * np.eye(A.shape[1])])
            b = np.concatenate([b, np.zeros(A.shape[1])])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    x = fit(list(data))
    scale = np.ones(len(base))
    for g, xv in zip(groups, x):
        for t in g:
            scale[t] = 1.0 + xv
    out = {"tag": args.tag, "rel": args.rel, "ridge": args.ridge, "groups": groups, "base": base.tolist(),
           "rel_change": x.tolist(), "new_table": (base * scale).tolist(), "per_mlp": []}
    for i, (p0, J, gt) in data.items():
        b0 = float(((p0 - gt) ** 2).mean())
        rec = {"mlp": i, "base_mse": b0, "in_sample_lin": float(((p0 + J @ x - gt) ** 2).mean()) / b0}
        if len(data) > 1:
            xl = fit([k for k in data if k != i])
            rec["loo_lin"] = float(((p0 + J @ xl - gt) ** 2).mean()) / b0
        out["per_mlp"].append(rec)
        print(json.dumps(rec))
    print("relative change per layer:", " ".join(f"{v:+.2f}" for v in x))
    print("new table:", " ".join(f"{v:.5f}" for v in out["new_table"]))
    (HERE / f"acc_aug_fit_{args.tag}_r{args.ridge:g}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
