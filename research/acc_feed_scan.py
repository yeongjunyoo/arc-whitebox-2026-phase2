"""Run estimators/k3_acc_feed.py (or another estimators/ module with the same propagate(lam=...) signature) on public
mini MLPs for a list of per-layer lam settings, metered like the harness (float32, BudgetContext(2^41, 120 s)), and
save every final-layer prediction so per-layer lam can be fitted offline (research/acc_feed_fit.py).

Settings (--lams, JSON): a list whose items are a scalar (lam_t for every t >= 1), a length-16 list, or
{"jac": delta} which expands to the 15 settings 0 and delta * e_t for t = 1..14 (finite-difference Jacobian).

    cd whest-starterkit
    python <vault>/scripts/memcap.py --gb 4 -- <abs>\\.venv\\Scripts\\python.exe ..\\research\\acc_feed_scan.py --mlps 50,51 --lams "[0.01, 0.03]" --threads 2 --tag s1
"""
from __future__ import annotations

import argparse
import os
import sys


def _early_threads() -> None:
    if "--threads" in sys.argv:
        k = sys.argv[sys.argv.index("--threads") + 1]
        for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = k


_early_threads()

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "estimators"))
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from acc_dump import ids, load_full  # noqa: E402
from k3_measure import free_gb  # noqa: E402

B = 2**41


def expand(settings, depth=16):
    out = []
    for s in settings:
        if isinstance(s, dict) and "jacbase" in s:  # base table and base + rel * base_t * e_t for t = 0..depth-2
            base = [float(v) for v in s["jacbase"]]
            rel = float(s.get("rel", 0.3))
            out.append(list(base))
            for t in range(depth - 1):
                v = list(base)
                v[t] = base[t] * (1.0 + rel)
                out.append(v)
        elif isinstance(s, dict) and "jac" in s:
            d = float(s["jac"])
            out.append([0.0] * depth)
            for t in range(1, depth - 1):
                v = [0.0] * depth
                v[t] = d
                out.append(v)
        elif isinstance(s, (int, float)):
            out.append([0.0] + [float(s)] * (depth - 1))
        else:
            assert len(s) == depth
            out.append([float(v) for v in s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", default="50")
    ap.add_argument("--lams", default="[0.01]")
    ap.add_argument("--threads", default=None)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--estimator", default="k3_acc_feed")
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget-mult", type=float, default=1.0, help="BudgetContext cap = mult * 2^41 (accuracy studies of costlier variants)")
    args = ap.parse_args()
    mod = __import__(args.estimator)
    kw = json.loads(args.kwargs)
    settings = expand(json.loads(args.lams))
    tag = args.tag or time.strftime("%m%d_%H%M")
    npz = HERE / f"acc_feedscan_{tag}.npz"
    store = dict(np.load(npz)) if npz.exists() else {}
    summary = []
    for mlp_id in ids(args.mlps):
        W, gt, avg_var, seed = load_full(mlp_id)
        weights = [fnp.asarray(W[l]) for l in range(16)]
        store[f"gt_{mlp_id}"] = gt[-1].astype(np.float64)
        for k, lam in enumerate(settings):
            key = f"pred_{mlp_id}_{k}"
            if key in store:
                continue
            fg = free_gb()
            waited = 0
            while fg < args.min_free_gb and waited < 900:  # host rule: wait while free RAM is under the floor
                time.sleep(5)
                waited += 5
                fg = free_gb()
            if fg < args.min_free_gb:
                print(f"[skip] mlp {mlp_id} setting {k}: only {fg:.2f} GB free after {waited}s", flush=True)
                continue
            est = mod.Estimator(lam=lam, **kw)
            t0 = time.time()
            with flops.BudgetContext(flop_budget=int(B * args.budget_mult), wall_time_limit_s=600.0, quiet=True) as ctx:
                pred = est.propagate(weights, dt=fnp.float32)
            pred = np.asarray(pred, dtype=np.float64)
            store[key] = pred[-1]
            store[f"lam_{k}"] = np.array(lam)
            mse = float(((pred[-1] - gt[-1]) ** 2).mean())
            rec = {"mlp_id": mlp_id, "setting": k, "lam_nonzero": {t: v for t, v in enumerate(lam) if v}, "final_mse": mse,
                   "C_over_B": ctx.flops_used / B, "wall_s": round(time.time() - t0, 1),
                   "residual_s": round(ctx.residual_wall_time_s, 3)}
            print(json.dumps(rec), flush=True)
            summary.append(rec)
            np.savez(npz, **store)
        del W, weights
    (HERE / f"acc_feedscan_{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("saved", npz.name)


if __name__ == "__main__":
    main()
