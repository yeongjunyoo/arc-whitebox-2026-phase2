"""Size bins for the per-MLP saturation frames on the depth-6 stack (mode "bins", coordinator 10-08 04:5x). From each
mini MLP's own counts (the 392 engine dumps in research/lc3_cache/s6_192, alpha per layer, as satdyn_stack_cost.py):
per early layer t, the restricted push sizes (layers outside the fold, t >= 2) and the 3 T^T (Pa, Pb) sizes, padded as
the sat step pads them, then k quantile bins per axis (the top bin the max over the 392 MLPs). Push bins are a sorted
list; 3 T^T bins are the distinct (Pa, Pb) pairs the 392 MLPs round up to, plus the per-axis top pair. Written as
fractions of n (n = 1024 here) so other widths scale: research/costfloor/satbins_table.json.
    whest-starterkit/.venv/Scripts/python.exe -X utf8 research/costfloor/satbins_table.py [--k 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from satdyn_stack_cost import N, DEPTH, pad, stack_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    a_ = ap.parse_args()
    calls, c, cf, ht, SAT, FRAC = stack_plan()
    d, q = c["sw_depth"], SAT.get("q", 32)
    ti, to, ta, tb, band = SAT["in"], SAT["out"], SAT["ta"], SAT["tb"], SAT["tb_band"]
    A = np.array([np.load(f)["a"] for f in sorted(glob.glob(str(ROOT / "research" / "lc3_cache" / "s6_192" / "full_*.npz")))])
    qs = [1.0] if a_.k == 1 else list(np.linspace(0.0, 1.0, a_.k + 1)[1:])
    push, tt, skip = {}, {}, []  # skip: calls every one of the 392 MLPs restricts (their plain views are not made)
    for t in range(1, DEPTH - 1):
        dtt = d + 1 if t in ht else d
        mp, mt = max(1 << d, q), max(1 << dtt, q)
        sp, sa, sb = [], [], []
        for a in A:
            al = a[t - 1]
            cin = int((a[t - 2] > -ti).sum()) if t > 1 else N
            sp.append(max(pad(cin, mp), pad(int((al > -to).sum()), mp)))
            sa.append(pad(int((al > -ta).sum()), mt))
            sb.append(pad(int(((np.abs(al) < tb) if band else (al > -tb)).sum()), mt))
        sp, sa, sb = map(np.array, (sp, sa, sb))

        def bins(col, mult):
            return sorted({pad(float(np.quantile(col, qq, method="higher")), mult) for qq in qs})
        if t >= 2 and t not in cf:
            push[t] = [b / N for b in bins(sp, mp) if b < N]
            if sp.max() < N:
                skip.append(["push", t])
        if sa.max() < N or sb.max() < N:
            skip.append(["tt", t])
        ba, bb = bins(sa, mt), bins(sb, mt)
        pairs = set()
        for x, y in zip(sa, sb):
            if x < N or y < N:
                pairs.add((next(b for b in ba if b >= x), next(b for b in bb if b >= y)))
        if pairs:
            pairs.add((ba[-1], bb[-1]))
        tt[t] = [[x / N, y / N] for x, y in sorted(pairs, key=lambda p: (p[0] * p[1], p)) if x < N or y < N]
    out = {"k": a_.k, "n_ref": N, "mlps": len(A), "fold_layers": sorted(cf), "push": push, "tt": tt, "skip_plain": skip,
           "shapes": sum(len(v) for v in push.values()) + sum(len(v) for v in tt.values())}
    (HERE / "satbins_table.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("push", "tt")}))
    print("push", {t: [round(f * N) for f in v] for t, v in push.items()})
    print("tt", {t: [[round(x * N), round(y * N)] for x, y in v] for t, v in tt.items()})


if __name__ == "__main__":
    main()
