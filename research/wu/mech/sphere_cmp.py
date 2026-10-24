"""Which truth does each file's third cumulant follow: the Gaussian-input one (mc_truth.py, the organizers' sampler) or
the one with the input-norm scale mode taken out (mc_sphere.py, inputs on the sphere of radius sqrt(n))? Per layer, the
band-off engine's (Mx - F) / 3 against both, kept and dropped entries, and the diagonal t3 (research/wu/mech,
coordinator 10-22, second round). Public mini networks with both truths only.

    .venv-fs0123/bin/python research/wu/mech/sphere_cmp.py     # writes sphere.json and sphere.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech_report import stats  # noqa: E402

FILES = [("opr30", "rec_opr30"), ("r37", "rec_r37"), ("r38 without the extraction", "rec_r38noext"),
         ("r38 with the law back", "rec_r38lawon"), ("r38", "rec")]
LAYERS = range(2, 15)


def main() -> None:
    ids = [i for i in range(60) if (HERE / "mc" / f"mcs_mini{i}.npz").exists()]
    assert all(i < 60 for i in ids)
    out = {}
    for mid in ids:
        mg = np.load(HERE / "mc" / f"mc_mini{mid}.npz")
        ms = np.load(HERE / "mc" / f"mcs_mini{mid}.npz")
        per_file = {}
        for lab, d in FILES:
            p = HERE / d / f"rec_mech_nott_mini{mid}.npz"
            if not p.exists():
                continue
            rb = np.load(p)
            rows = {}
            for t in LAYERS:
                F = rb[f"F_{t}"].astype(np.float64)
                Mx = rb[f"Mx_{t}"].astype(np.float64)
                te = (Mx - F) / 3.0
                n = te.shape[0]
                off = ~np.eye(n, dtype=bool)
                kr, kc = rb[f"kr_{t}"], rb[f"kc_{t}"]
                keep = np.outer(kr, kc) & off
                drop = ~np.outer(kr, kc) & off
                row = {}
                for tname, mc in (("gauss", mg), ("sphere", ms)):
                    TT, SE = mc[f"TT_{t}"].astype(np.float64), mc[f"SE_{t}"].astype(np.float64)
                    r = {"keep": stats(te[keep], TT[keep], SE[keep])}
                    if drop.any():
                        r["drop"] = stats(te[drop], TT[drop], SE[drop])
                    d3 = np.diagonal(Mx) / 3.0
                    r["t3"] = stats(d3, mc["t3"][t], mc["t3_se"][t])
                    row[tname] = r
                rows[str(t)] = row
            per_file[lab] = rows
        out[str(mid)] = per_file
    (HERE / "sphere.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# Engine third cumulant against the Gaussian-input truth and the sphere-input truth (sphere_cmp.py)\n",
         "Slope of the band-off engine's entries on the truth (noise-corrected): kept / dropped / diagonal t3.\n"]
    for mid, per_file in out.items():
        labs = list(per_file)
        for tname, title in (("gauss", "Gaussian inputs (mc_truth.py)"), ("sphere", "inputs on the sphere (mc_sphere.py)")):
            L += ["", f"mini {mid}, truth: {title}; kept / dropped / t3 slope", "", "| t | " + " | ".join(labs) + " |",
                  "|---|" + "---|" * len(labs)]
            for t in LAYERS:
                cells = []
                for lab in labs:
                    r = per_file[lab][str(t)][tname]
                    cells.append(" / ".join(("%.2f" % r[k]["slope_engine_on_truth"]) if k in r else "-"
                                            for k in ("keep", "drop", "t3")))
                L.append(f"| {t} | " + " | ".join(cells) + " |")
    (HERE / "sphere.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", HERE / "sphere.json", HERE / "sphere.md")


if __name__ == "__main__":
    main()
