"""Band study (research/wu/mech): where does the next layer's mean move when layer L's 3 T^T is replaced by the Monte Carlo
truth (run_layers.sh tag L<L>t) instead of the clean band (Lbase)? Layer L + 1's mean is pK1 = w01(a, s) + t3 w31 / 6
+ d4 w41 / 24 (engine mode: no per-neuron law); its input mean mu = W^T pK1_L is the same in both runs, so the change
comes from s (through C'_L), t3 (the new K3 source) and d4 (the K4 rows). This splits the change of the prediction
into the s part (phi(a) ds), the t3 part (dt3 w31 / 6) and the rest (d4 and higher order), and reports each part's
share and whether it moves the mean toward or away from the dataset's Monte Carlo mean.

    .venv-fs0123/bin/python research/wu/mech/mech_decomp.py
"""
from __future__ import annotations

import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"


def phi(a):
    return np.exp(-0.5 * a * a) / sqrt(2 * np.pi)


def main() -> None:
    out, md = {}, ["| mini | L | MSE(L+1) base | MSE(L+1) t | rms d(pred) | s part | t3 part | rest | <d, err_base> / |d|^2 (s, t3, rest) |",
                   "|---|---|---|---|---|---|---|---|---|"]
    for mid in (0, 1):
        (_, _, _, W, gt), = list(AB.load_rows(SNAP, "mini", [mid]))
        b = np.load(HERE / "rec" / "loc" / f"Lbase_mini{mid}.npz")
        per = {}
        for L in range(2, 14):
            f = HERE / "rec" / "loc" / f"L{L}t_mini{mid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            N1 = L + 1
            if f"a_{N1}" not in z.files or f"a_{N1}" not in b.files:
                continue
            Wn = W[N1].astype(np.float64)
            vb = np.einsum("ka,kl,la->a", Wn, b[f"Cp_{L}"].astype(np.float64), Wn, optimize=True)
            vt = np.einsum("ka,kl,la->a", Wn, z[f"Cp_{L}"].astype(np.float64), Wn, optimize=True)
            sb, st = np.sqrt(vb), np.sqrt(vt)
            ab = b[f"a_{N1}"].astype(np.float64)
            t3b, t3t = b[f"diag_{N1}"] / 3.0, z[f"diag_{N1}"] / 3.0
            w31 = -(ab * phi(ab)) / vb
            pb, pt = b["pred"][N1], z["pred"][N1]
            tr = gt[N1].astype(np.float64)
            d = pt - pb
            e_b = tr - pb  # the change that would be right
            parts = {"s": phi(ab) * (st - sb), "t3": (t3t - t3b) * w31 / 6.0}
            parts["rest"] = d - parts["s"] - parts["t3"]
            rd = float(np.sqrt((d ** 2).mean()))
            row = {"mse_base": float(((pb - tr) ** 2).mean()), "mse_t": float(((pt - tr) ** 2).mean()), "rms_d": rd,
                   "part_rms_over_d": {k: float(np.sqrt((v ** 2).mean()) / rd) for k, v in parts.items()},
                   "part_gain": {k: float(v @ e_b / max(v @ v, 1e-300)) for k, v in parts.items()},
                   "part_corr_with_d": {k: float(np.corrcoef(v, d)[0, 1]) for k, v in parts.items()}}
            per[str(L)] = row
            md.append(f"| {mid} | {L} | {row['mse_base']:.3e} | {row['mse_t']:.3e} | {rd:.2e} | "
                      + " | ".join("%.2f" % row["part_rms_over_d"][k] for k in ("s", "t3", "rest")) + " | "
                      + ", ".join("%.2f" % row["part_gain"][k] for k in ("s", "t3", "rest")) + " |")
        out[str(mid)] = per
    (HERE / "decomp.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "decomp.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
