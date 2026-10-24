"""R25-B: size bins for the new column restrictions from the 392 public full-MLP alpha dumps (lc3_cache/s6_192; ids
112-575, none of 304-351), k quantile bins per axis like research/costfloor/satbins_table.py.
  s1: fold-layer push inner count alive_<inner> (default 2.5) at t-1, depth-6 multiples of 64 (layers <s1_from>-10,
      default 6)
  s8: layers 11-14 push inner alive_<inner> at t-1 and output alive_<out> at t, multiples of 32 (APA depth 5)
  s2: last layer inner alive_<inner> at 14 and output band |a| < <band> at 15 (multiples of g: 32 APA depth 5, 64 plain 6)
    python research/r25b/bins.py --k 3 --out 3.5 --band 3.5 --g2 32 [--inner 2.5]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_A = None


def alphas():
    global _A
    if _A is None:
        fs = sorted(glob.glob(str(ROOT / 'research' / 'lc3_cache' / 's6_192' / 'full_*.npz')))
        assert fs and not any(304 <= int(Path(f).name[5:9]) <= 351 for f in fs)
        _A = np.array([np.load(f)['a'] for f in fs])
    return _A


def pad(x, m):
    return int(-(-int(x) // m) * m)


def compute(k=3, out=3.5, band=3.5, g2=32, inner=2.5, s1_from=6):
    A = alphas(); M, L, n = A.shape
    qs = list(np.linspace(0, 1, k + 1)[1:])
    def bins(col, g): return sorted({min(n, pad(np.quantile(col, q, method='higher'), g)) for q in qs})
    def up(x, bs): return next((b for b in bs if b >= x), n)
    res = {'k': k, 'mlps': M, 'inner': inner, 'out': out, 'band': band, 's1_bins': {}, 's8_pairs': {}, 'kept': {}}
    for t in range(s1_from, 11):
        c = np.array([pad((A[i, t - 2] > -inner).sum(), 64) for i in range(M)])
        b = [x for x in bins(c, 64) if x < n]
        res['s1_bins'][t] = b
        res['kept'][f's1_{t}'] = float(np.mean([up(x, b + [n]) for x in c]) / n)
    for t in range(11, 15):
        ci = np.array([pad((A[i, t - 2] > -inner).sum(), 32) for i in range(M)])
        co = np.array([pad((A[i, t - 1] > -out).sum(), 32) for i in range(M)])
        bi, bo = bins(ci, 32), bins(co, 32)
        res['s8_pairs'][t] = [list(p) for p in sorted({(up(x, bi), up(y, bo)) for x, y in zip(ci, co)} | {(bi[-1], bo[-1])})]
        res['kept'][f's8_{t}'] = float(np.mean([up(x, bi) * up(y, bo) for x, y in zip(ci, co)]) / n ** 2)
    ci = np.array([pad((A[i, 13] > -inner).sum(), g2) for i in range(M)])
    cb = np.array([pad((np.abs(A[i, 14]) < band).sum(), g2) for i in range(M)])
    bi, bb = bins(ci, g2), bins(cb, g2)
    res['s2_pairs'] = [list(p) for p in sorted({(up(x, bi), up(y, bb)) for x, y in zip(ci, cb)} | {(bi[-1], bb[-1])})]
    res['s2_g'] = g2
    res['kept']['s2'] = float(np.mean([up(x, bi) * up(y, bb) for x, y in zip(ci, cb)]) / n ** 2)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=3); ap.add_argument('--out', type=float, default=3.5)
    ap.add_argument('--band', type=float, default=3.5); ap.add_argument('--g2', type=int, default=32)
    ap.add_argument('--inner', type=float, default=2.5)
    ap.add_argument('--s1-from', type=int, default=6); a_ = ap.parse_args()
    print(json.dumps(compute(a_.k, a_.out, a_.band, a_.g2, a_.inner, a_.s1_from)))
