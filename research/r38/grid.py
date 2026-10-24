"""R38 grid: engine-only final-layer MSE (and per-layer MSE, eps) of research/r38/k3_acc_tc.py configs on public mini
networks, one json line per (config, network). Public mini rows only (held-out rows 60-99 are refused).

    (Mac) .venv-fs0123/bin/python research/r38/grid.py --ids 0 --cfgs "base:,cA:carry=True;tau=1;lam_mult=0" --out research/r38/out/g1.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wu"))
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"


def parse(c):
    tag, _, kv = c.partition(":")
    kw = {}
    for item in filter(None, kv.split(";")):
        k, _, v = item.partition("=")
        try:
            kw[k] = (v == "True") if v in ("True", "False") else float(v)
        except ValueError:
            kw[k] = v
    return tag, kw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0")
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    assert all(0 <= i < 60 for i in ids), "held-out mini rows 60-99 are not read"
    spec = importlib.util.spec_from_file_location("k3_acc_tc", str(HERE / "k3_acc_tc.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import flopscope.numpy as fnp
    cfgs = [parse(c) for c in a.cfgs.split(",")]
    for mid, seed, name, W, truth in AB.load_rows(SNAP, "mini", ids):
        Wf = [fnp.asarray(W[t]) for t in range(W.shape[0])]
        tr = truth.astype(np.float64)
        for tag, kw in cfgs:
            t0 = time.time()
            rec = {}
            try:
                y = np.asarray(m.Estimator(preset="recommended", **kw).propagate(Wf, dt=fnp.float32, record=rec),
                               dtype=np.float64)
                mse = ((y - tr) ** 2).mean(axis=1)
                row = {"tag": tag, "kw": kw, "id": mid, "final": float(mse[-1]), "layers": [float(v) for v in mse],
                       "eps": [float(e) for e in rec.get("eps", [])], "finite": bool(np.isfinite(y).all()),
                       "sec": round(time.time() - t0, 1)}
            except Exception as e:  # a diverging config is a result, not a crash of the grid
                row = {"tag": tag, "kw": kw, "id": mid, "error": repr(e)[:300], "sec": round(time.time() - t0, 1)}
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in ("tag", "id", "final", "sec", "error")}), flush=True)


if __name__ == "__main__":
    main()
