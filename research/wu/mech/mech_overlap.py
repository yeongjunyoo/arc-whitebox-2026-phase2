"""Does the band do the extraction's job? Final-layer means of each file's band-off (B) and band-on (P) engine on public
minis 0-3 against the dataset's means (research/wu/mech, coordinator 10-22, second round).

With e = truth - B (band-off error) and d = P - B (what the band changes), MSE(P) - MSE(B) = (|d|^2 - 2 d.e) / n.
For the engine without the extraction (r38 with _X38 off, "noext") e splits exactly into the extraction's own change
X = B(r38) - B(noext) and r38's remaining error e38 = truth - B(r38): e = X + e38. The band's MSE change in noext then
splits into -2 d.X (what the band does along the extraction's change), -2 d.e38 (along the error the extraction leaves)
and |d|^2. The same with R = B(r38) - B(r37) for r37 (all of R38) and with the law-on pair B(lawon) - B(r37).

Inputs (all written by mech_run.py --record): rec/ (r38: rec_mech_nott, rec_mech_prod), rec_opr30/, rec_r37/,
rec_r38noext/, rec_r38lawon/ (band off, round 2c) and pred_r37/, pred_r38noext/, pred_r38lawon/ (band on, round 2e),
pred_r38noxa/ (r38 without the extraction's dA part, band off and on, round 2e).
Truth: the dataset's all_layer_means through research/wu/ablate.py load_rows (public ids < 60 only).

    .venv-fs0123/bin/python research/wu/mech/mech_overlap.py      # writes overlap.json and overlap.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"
IDS = [0, 1, 2, 3]
FILES = {  # name: (band-off recording, band-on recording)
    "opr30": ("rec_opr30/rec_mech_nott", "rec_opr30/rec_mech_prod"),
    "r37": ("rec_r37/rec_mech_nott", "pred_r37/rec_mech_prod"),
    "noext": ("rec_r38noext/rec_mech_nott", "pred_r38noext/rec_mech_prod"),
    "lawon": ("rec_r38lawon/rec_mech_nott", "pred_r38lawon/rec_mech_prod"),
    "noxa": ("pred_r38noxa/rec_mech_nott", "pred_r38noxa/rec_mech_prod"),
    "r38": ("rec/rec_mech_nott", "rec/rec_mech_prod"),
}


def final(stem: str, mid: int):
    p = HERE / f"{stem}_mini{mid}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return np.asarray(z["pred"][-1], dtype=np.float64)


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main() -> None:
    assert all(0 <= i < 60 for i in IDS), "held-out mini rows 60-99 are not read"
    truth = {m: gt[-1].astype(np.float64) for m, _, _, _, gt in AB.load_rows(SNAP, "mini", IDS)}
    out = {}
    for mid in IDS:
        tr = truth[mid]
        n = tr.size
        B = {k: final(v[0], mid) for k, v in FILES.items()}
        P = {k: final(v[1], mid) for k, v in FILES.items()}
        row = {}
        for k in FILES:
            if B[k] is None or P[k] is None:
                continue
            e, d = tr - B[k], P[k] - B[k]
            row[k] = {"mse_off": float(e @ e / n), "mse_on": float((e - d) @ (e - d) / n),
                      "on_over_off": float((e - d) @ (e - d) / (e @ e)), "cos_d_e": cos(d, e),
                      "d_rms_over_e_rms": float(np.sqrt((d @ d) / (e @ e)))}
        # the band's change in each file against the others' (does the band do the same thing everywhere?)
        names = [k for k in FILES if B[k] is not None and P[k] is not None]
        row["cos_between_band_changes"] = {f"{a}~{b}": cos(P[a] - B[a], P[b] - B[b])
                                           for i, a in enumerate(names) for b in names[i + 1:]}
        # transfer: file a's band change put on file b's band-off means, MSE over b's band-off MSE (the diagonal is b's
        # own band on / off): is it the band's change or the error it meets that differs between files?
        row["transfer"] = {f"{a}->{b}": float((tr - B[b] - (P[a] - B[a])) @ (tr - B[b] - (P[a] - B[a])) / ((tr - B[b]) @ (tr - B[b])))
                           for a in names for b in names}
        # splits of the band's MSE change along a change X between two band-off engines and the error left after it
        for base, ref, lab in (("noext", "r38", "extraction (law off)"), ("r37", "lawon", "extraction (law on)"),
                               ("r37", "r38", "R38"), ("opr30", "r38", "opr30 -> r38"), ("noxa", "r38", "the dA part"),
                               ("noext", "noxa", "the extraction without its dA part")):
            if any(x is None for x in (B[base], P[base], B[ref])):
                continue
            e, d = tr - B[base], P[base] - B[base]
            X = B[ref] - B[base]
            eref = tr - B[ref]
            assert np.allclose(e, X + eref)
            tot = (d @ d - 2 * d @ e) / n
            row[f"split {base} by {lab}"] = {
                "band_mse_change": float(tot), "along_change": float(-2 * d @ X / n), "along_left_error": float(-2 * d @ eref / n),
                "d_squared": float(d @ d / n), "cos_d_X": cos(d, X), "cos_d_left_error": cos(d, eref), "cos_X_e": cos(X, e),
                "X_rms_over_e_rms": float(np.sqrt((X @ X) / (e @ e)))}
        out[str(mid)] = row
    (HERE / "overlap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# The band's change of the final means against the extraction's (mech_overlap.py)\n",
         "Per mini: on / off = MSE(P) / MSE(B) from the recorded predictions; cos(d, e): the band's change against the band-off "
         "error; split: the band's MSE change (x 1e-9) = along the other engine's change + along the error that engine leaves "
         "+ |d|^2.\n",
         "| file | on / off (minis " + ", ".join(out) + ") | cos(d, e) | rms d / rms e |", "|---|---|---|---|"]
    for k in FILES:
        rs = [row[k] for row in out.values() if k in row]
        if rs:
            L.append(f"| {k} | " + ", ".join("%.3f" % r["on_over_off"] for r in rs) + " | "
                     + ", ".join("%.2f" % r["cos_d_e"] for r in rs) + " | " + ", ".join("%.3f" % r["d_rms_over_e_rms"] for r in rs) + " |")
    L += ["", "| mini | split | band MSE change | along the change | along the error left | d^2 | cos(d, change) | cos(d, error left) | cos(change, e) | rms change / rms e |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mid, row in out.items():
        for k, r in row.items():
            if k.startswith("split "):
                L.append(f"| {mid} | {k[6:]} | {r['band_mse_change'] * 1e9:+.3f} | {r['along_change'] * 1e9:+.3f} | "
                         f"{r['along_left_error'] * 1e9:+.3f} | {r['d_squared'] * 1e9:.3f} | {r['cos_d_X']:.3f} | "
                         f"{r['cos_d_left_error']:.3f} | {r['cos_X_e']:.3f} | {r['X_rms_over_e_rms']:.3f} |")
    names = [k for k in FILES if all(k in row for row in out.values())]
    L += ["", "Transfer: the band change of the row's file put on the column file's band-off means, MSE over that band-off MSE "
          "(diagonal: the file's own band on / off), geometric mean over minis " + ", ".join(out) + ":", "",
          "| band change of (row), applied to (column) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        cells = []
        for b in names:
            v = [row["transfer"][f"{a}->{b}"] for row in out.values() if f"{a}->{b}" in row.get("transfer", {})]
            cells.append("%.3f" % float(np.exp(np.mean(np.log(v)))) if v else "-")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "Cosine between the band changes of two files (range over minis " + ", ".join(out) + "):", "",
          "| band change of | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        cells = []
        for b in names:
            v = [row["cos_between_band_changes"].get(f"{a}~{b}", row["cos_between_band_changes"].get(f"{b}~{a}"))
                 for row in out.values()]
            v = [x for x in v if x is not None]
            cells.append("1" if a == b else ("%.2f-%.2f" % (min(v), max(v)) if v else "-"))
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    (HERE / "overlap.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", HERE / "overlap.json", HERE / "overlap.md")


if __name__ == "__main__":
    main()
