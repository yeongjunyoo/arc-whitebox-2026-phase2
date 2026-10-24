"""Tabulate research/r38/out/*.jsonl: per config, mean final-layer MSE ratio to base over shared networks."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
base = {r["id"]: r["final"] for r in rows if r["tag"] == "base" and "final" in r}
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)
for tag, rs in by.items():
    ok = [r for r in rs if "final" in r and r["finite"]]
    bad = [r for r in rs if "final" not in r or not r["finite"]]
    ids = [r["id"] for r in ok if r["id"] in base]
    if ids:
        f = np.array([next(r["final"] for r in ok if r["id"] == i) for i in ids])
        b = np.array([base[i] for i in ids])
        ratio = f.mean() / b.mean()
        better = int((f < b).sum())
        geo = float(np.exp(np.log(f / b).mean()))
        print(f"{tag:6s} {str(rs[0]['kw']):58s} n={len(ids):2d} mean {f.mean():.4e} ratio {ratio:.4f} geo {geo:.4f} better {better}/{len(ids)} bad {len(bad)}")
    else:
        for r in ok:
            print(f"{tag:6s} {str(r['kw']):58s} id {r['id']} final {r['final']:.4e} (no base)")
        for r in bad:
            print(f"{tag:6s} {str(r['kw']):58s} id {r['id']} BAD {r.get('error', 'nonfinite')}")
