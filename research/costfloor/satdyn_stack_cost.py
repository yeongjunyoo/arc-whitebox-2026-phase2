"""Per-MLP saturation frames on the depth-6 stack (coordinator 10-08 04:5x): what mode dyn would save on C against the
stack's fixed (p99) frames, and what size bins for the setup plan would keep of it. From shapes only, like
sat_cost_engine.py: the stack's own plan calls (_SwPlan.calls of _cf_sp_p2b_d6p: fold layers, depth-6 push and 3 T^T),
the Strassen-Winograd bill (swcost.py) plus the sat gathers and scatters, and each MLP's counts from the 392 engine dumps
(research/lc3_cache/s6_192, alpha per layer). The fold layers keep an unrestricted push in every mode (the fold's frames
need it); dyn binned rounds each call's padded size up to the smallest of k bins (quantiles over the 392 MLPs, the top
bin their max), so a setup plan can make every bin's frames and views once.
    whest-starterkit/.venv/Scripts/python.exe -X utf8 research/costfloor/satdyn_stack_cost.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(HERE))
from swcost import sw_cost  # noqa: E402

B = 2 ** 41
N, DEPTH = 1024, 16
STACK = "k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_p2b_d6p"


def stack_plan():
    from bench import load_estimator
    import flopscope.numpy as fnp
    Est = load_estimator(ROOT / "estimators" / STACK, None)
    g = Est.propagate.__globals__
    c = Est()._cost
    hp = frozenset(int(t) for t in c["sw_hi_push"])
    ht = frozenset(int(t) for t in c["sw_hi_tt"])
    cf = g["_covfold_layers"](N, DEPTH, c["sw_depth"], ht)
    real = g["fnp"].empty

    def fake(shape, dtype=None):
        raise RuntimeError("no arena")
    g["fnp"].empty = fake
    try:
        P = g["_SwPlan"].__new__(g["_SwPlan"])
        try:
            g["_SwPlan"].__init__(P, fnp.float32, N, DEPTH, c["age"], c["rank"], c["sw_depth"], c["sw_depth_batched"], hp, ht,
                                  frozenset(int(t) for t in c["sw_hi_batched"]), bool(c["sw_inplace"]), True, cf)
        except RuntimeError:
            pass
        calls = dict(P.calls)
    finally:
        g["fnp"].empty = real
    return calls, c, cf, ht, g["_SAT"], g["_SAT_FRAC"]


def pad(x, mult):
    k = int(x // mult) + (1 if x % mult > 1e-9 else 0)
    return min(N, k * mult)


def main():
    calls, c, cf, ht, SAT, FRAC = stack_plan()
    d = c["sw_depth"]
    q = SAT.get("q", 32)
    ti, to, ta, tb, band = SAT["in"], SAT["out"], SAT["ta"], SAT["tb"], SAT["tb_band"]
    rb = ("band_" if band else "alive_") + str(tb)
    A = np.array([np.load(f)["a"] for f in sorted(glob.glob(str(ROOT / "research" / "lc3_cache" / "s6_192" / "full_*.npz")))])

    def push_cost(t, Pp):
        dd, _, m, K, Nn = calls[("push", t)]
        if Pp < N:
            return sw_cost(m, Pp, Pp, d) + 4 * m * Pp + 4 * Pp * N + 4 * Pp * Pp + m * N + m * Pp
        return sw_cost(m, N, N, dd)

    def tt_cost(t, Pa, Pb):
        dd, _, m, K, Nn = calls[("tt", t)]
        nth = t * 5
        if Pa < N or Pb < N:
            out = sw_cost(Pa, K, Pb, dd) + (4 * K * Pa if Pa < N else 0) + (4 * K * Pb if Pb < N else 0)
            return out + N * N + (Pa * Pb if (Pa < N and Pb < N) else 0) + Pa * Pb + 2 * K * N + 2 * nth * N + N
        return sw_cost(N, K, N, dd)

    early = [t for t in range(1, DEPTH - 1)]
    # sizes per MLP and layer: fixed (p99 tables) and dyn (own counts), as propagate's sat step computes them
    def fixed_sizes(t):
        dtt = d + 1 if t in ht else d
        Pp = N if (t == 1 or t in cf) else max(pad(FRAC["alive_" + str(ti)][t - 1] * N, max(1 << d, q)),
                                                 pad(FRAC["alive_" + str(to)][t] * N, max(1 << d, q)))
        Pa = pad(FRAC["alive_" + str(ta)][t] * N, max(1 << dtt, q))
        Pb = pad(FRAC[rb][t] * N, max(1 << dtt, q))
        return Pp, Pa, Pb

    def dyn_sizes(a, t):
        dtt = d + 1 if t in ht else d
        al = a[t - 1]
        cin = int((a[t - 2] > -ti).sum()) if t > 1 else N
        cout = int((al > -to).sum())
        ca = int((al > -ta).sum())
        cb = int(((np.abs(al) < tb) if band else (al > -tb)).sum())
        Pp = N if (t == 1 or t in cf) else max(pad(cin, max(1 << d, q)), pad(cout, max(1 << d, q)))
        return Pp, pad(ca, max(1 << dtt, q)), pad(cb, max(1 << dtt, q))

    fixed = {t: fixed_sizes(t) for t in early}
    dyn = np.array([[dyn_sizes(a, t) for t in early] for a in A])  # (mlp, layer, 3)
    base = sum(push_cost(t, N) + tt_cost(t, N, N) for t in early)
    cost_fixed = sum(push_cost(t, fixed[t][0]) + tt_cost(t, fixed[t][1], fixed[t][2]) for t in early)
    res = {"fold_layers": sorted(cf), "fixed_sizes": {t: list(map(int, fixed[t])) for t in early}}

    def mean_cost(sizes):  # sizes (mlp, layer, 3)
        return float(np.mean([sum(push_cost(t, int(s[i][0])) + tt_cost(t, int(s[i][1]), int(s[i][2])) for i, t in enumerate(early))
                              for s in sizes]))

    exact = mean_cost(dyn)
    res["dyn_exact"] = {"dC/B_vs_fixed": (exact - cost_fixed) / B, "ratio_C_vs_fixed_stack": None}
    restricted_calls = {("push", t) for t in early if fixed[t][0] < N} | {("tt", t) for t in early if fixed[t][1] < N or fixed[t][2] < N}
    res["restricted_calls_fixed"] = len(restricted_calls)
    # bins: per call, quantiles of the dyn padded sizes; joint for 3 T^T along one index (both sizes from the same quantile)
    for k in (1, 2, 3, 4):
        qs = [1.0] if k == 1 else list(np.linspace(0.0, 1.0, k + 1)[1:])
        binned = dyn.copy()
        n_shapes = 0
        for i, t in enumerate(early):
            dtt = d + 1 if t in ht else d
            for j in range(3):
                if j == 0 and (t == 1 or t in cf):
                    continue
                col = dyn[:, i, j]
                if np.all(col == N):
                    continue
                mult = max(1 << (d if j == 0 else dtt), q)
                bins = sorted({pad(float(np.quantile(col, qq, method="higher")), mult) for qq in qs})
                binned[:, i, j] = [next(b for b in bins if b >= v) for v in col]
                n_shapes += len(bins) if j != 2 else 0  # 3 T^T bins counted once per layer (joint index)
            # joint 3 T^T: one shape per (Pa, Pb) pair actually used
        shapes = set()
        for s in binned:
            for i, t in enumerate(early):
                if s[i][0] < N:
                    shapes.add(("push", t, int(s[i][0])))
                if s[i][1] < N or s[i][2] < N:
                    shapes.add(("tt", t, int(s[i][1]), int(s[i][2])))
        cst = mean_cost(binned)
        res[f"dyn_bins_{k}"] = {"dC/B_vs_fixed": (cst - cost_fixed) / B, "distinct_restricted_shapes": len(shapes)}
    res["fixed_dC/B_vs_unrestricted"] = (cost_fixed - base) / B
    print(json.dumps(res, indent=1, default=str))
    (HERE / "satdyn_stack_cost.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
