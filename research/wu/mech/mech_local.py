"""Band study, local test (research/wu/mech): with everything upstream the same (the clean band at every other layer),
how does the choice for layer L's dropped 3 T^T entries move layer L's C' (activation covariance) and layer L + 1's
third cumulant, measured against Monte Carlo truth? Reads rec/loc/*.npz (run_layers.sh), mc/mccov_mini<id>.npz (C'),
mc/mc_mini<id>.npz (T, t3) and runs/layers.jsonl (MSE per layer); writes local.json and local.md.

Variants at layer L (base: dropped entries 0): x = the engine's exact values, m = Monte Carlo truth on the dropped
entries, t = Monte Carlo truth on every off-diagonal entry of layer L.
For each variant v: rel_err = ||C'_v - C_mc|| / ||C_mc|| over off-diagonal entries (Frobenius), and the step quality
of its change d = C'_v - C'_base against the needed change g = C_mc - C'_base: gain = <d, g> / <d, d>, the best multiple of d (1: d has the right size
along its own direction; 0.5: d is twice too large and leaves the error unchanged; below 0.5 it makes C' worse; below
0 it points the wrong way), and cos = <d, g> / (|d| |g|). The squared-error change is <d, d> - 2 <d, g>.

    .venv-fs0123/bin/python research/wu/mech/mech_local.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # research/wu: ablate.load_rows
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"
IDS = [0, 1]
LAYERS = list(range(2, 15))


def frob(x):
    return float(np.sqrt((x * x).sum()))


def main() -> None:
    runs = {}
    for ln in (HERE / "runs" / "layers.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("{"):
            r = json.loads(ln)
            runs[(r["tag"], r["mlp_id"])] = r
    out = {}
    md = ["| mini | L | dropped | C' rel err base | x | m | t | gain x | gain m | gain t | cos x | cos m | "
          "t3(L+1) rel rms err base | x | m | t | var(L+1) mean rel err base | x | m | t | var(L+1) rms rel err base | t | "
          "MSE(L+1) base | x | m | t | final MSE base | x | m | t |",
          "|" + "---|" * 30]
    for mid in IDS:
        (_, _, _, Wall, _), = list(AB.load_rows(SNAP, "mini", [mid]))
        cv = np.load(HERE / "mc" / f"mccov_mini{mid}.npz")
        mc = np.load(HERE / "mc" / f"mc_mini{mid}.npz")
        base = np.load(HERE / "rec" / "loc" / f"Lbase_mini{mid}.npz")
        per = {}
        for L in LAYERS:
            if f"Cp_{L}" not in base.files:
                continue
            C = cv[f"C_{L}"].astype(np.float64)
            n = C.shape[0]
            off = ~np.eye(n, dtype=bool)
            Cb = base[f"Cp_{L}"].astype(np.float64)
            g = (C - Cb)[off]
            row = {"n_drop": int(runs[("Lbase", mid)]["band"].get(str(L), {}).get("n_drop", 0)),
                   "C_rel_err": {"base": frob((Cb - C)[off]) / frob(C[off])},
                   "C_diag_rel_err": {"base": frob(np.diagonal(Cb) - np.diagonal(C)) / frob(np.diagonal(C))},
                   "gain": {}, "cos": {}, "t3_next_rel_rms_err": {}, "T_next_rel_rms_err": {}, "final_mse": {},
                   "next_layer_mse": {}, "var_next_mean_rel_err": {}, "var_next_rms_rel_err": {}}
            Wn = Wall[L + 1].astype(np.float64)
            vtrue = mc["var_z"][L + 1]
            N1 = L + 1
            if N1 <= 14:
                t3m = mc["t3"][N1]
                TT = mc[f"TT_{N1}"].astype(np.float64)
            for v, tag in (("base", "Lbase"), ("x", f"L{L}x"), ("m", f"L{L}m"), ("t", f"L{L}t")):
                r = runs.get((tag, mid))
                if r is None:
                    continue
                row["final_mse"][v] = r["final_mse"]
                row["next_layer_mse"][v] = r["layer_mse"][N1] if N1 < len(r["layer_mse"]) else None
                z = base if v == "base" else np.load(HERE / "rec" / "loc" / f"{tag}_mini{mid}.npz")
                Cz = z[f"Cp_{L}"].astype(np.float64)
                vz = np.einsum("ka,kl,la->a", Wn, Cz, Wn, optimize=True)  # the next layer's variance, diag(W^T C' W)
                row["var_next_mean_rel_err"][v] = float(((vz - vtrue) / vtrue).mean())
                row["var_next_rms_rel_err"][v] = float(np.sqrt((((vz - vtrue) / vtrue) ** 2).mean()))
                if v != "base":
                    Cv = z[f"Cp_{L}"].astype(np.float64)
                    d = (Cv - Cb)[off]
                    row["C_rel_err"][v] = frob((Cv - C)[off]) / frob(C[off])
                    row["C_diag_rel_err"][v] = frob(np.diagonal(Cv) - np.diagonal(C)) / frob(np.diagonal(C))
                    dd = float(d @ d)
                    row["gain"][v] = float(d @ g) / dd if dd > 0 else None
                    row["cos"][v] = float(d @ g) / (np.sqrt(dd) * frob(g)) if dd > 0 else None
                if N1 <= 14 and f"diag_{N1}" in z.files:
                    t3e = z[f"diag_{N1}"] / 3.0
                    row["t3_next_rel_rms_err"][v] = float(np.sqrt(((t3e - t3m) ** 2).mean() / (t3m ** 2).mean()))
                    Te = (z[f"Mx_{N1}"].astype(np.float64) - z[f"F_{N1}"].astype(np.float64)) / 3.0
                    row["T_next_rel_rms_err"][v] = float(np.sqrt(((Te - TT)[off] ** 2).mean() / (TT[off] ** 2).mean()))
            per[str(L)] = row
            f = lambda d_, k: ("-" if d_.get(k) is None else "%.4f" % d_[k])  # noqa: E731
            e = lambda d_, k: ("-" if d_.get(k) is None else "%.4e" % d_[k])  # noqa: E731
            md.append(f"| {mid} | {L} | {row['n_drop']} | " + " | ".join(f(row["C_rel_err"], k) for k in ("base", "x", "m", "t")) + " | "
                      + " | ".join(f(row["gain"], k) for k in ("x", "m", "t")) + " | " + " | ".join(f(row["cos"], k) for k in ("x", "m")) + " | "
                      + " | ".join(f(row["t3_next_rel_rms_err"], k) for k in ("base", "x", "m", "t")) + " | "
                      + " | ".join(f(row["var_next_mean_rel_err"], k) for k in ("base", "x", "m", "t")) + " | "
                      + " | ".join(f(row["var_next_rms_rel_err"], k) for k in ("base", "t")) + " | "
                      + " | ".join(e(row["next_layer_mse"], k) for k in ("base", "x", "m", "t")) + " | "
                      + " | ".join(e(row["final_mse"], k) for k in ("base", "x", "m", "t")) + " |")
        out[str(mid)] = per
    (HERE / "local.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "local.md").write_text("\n".join(md) + "\n" + narrow(out), encoding="utf-8")
    print("\n".join(md))


def narrow(out):
    """The README's two tables: final-layer MSE of one-layer swaps relative to the clean band, and the local effect of
    the full truth at layer L (base -> t) on C'_L, the next layer's variance and t3, and the next layer's mean."""
    r = lambda a, b: "-" if (a is None or b is None) else "%.3f" % (a / b)  # noqa: E731
    L1 = ["\n#### One-layer swaps: final-layer MSE / clean band (base = 1)\n",
          "| L | dropped (mini 0 / 1) | x: engine values, mini 0 | mini 1 | m: truth on dropped, mini 0 | mini 1 | t: truth on all, mini 0 | mini 1 |",
          "|---|---|---|---|---|---|---|---|"]
    L2 = ["\n#### Local effect at layer L of the truth on every off-diagonal entry (clean band -> truth)\n",
          "| mini | L | C'_L rel err | var(L+1) rms rel err | var(L+1) mean rel err | t3(L+1) rel rms err | MSE of layer L+1 means |",
          "|---|---|---|---|---|---|---|"]
    Ls = sorted({int(L) for per in out.values() for L in per})
    for L in Ls:
        cells = []
        for v in ("x", "m", "t"):
            for mid in ("0", "1"):
                row = out.get(mid, {}).get(str(L))
                cells.append(r(row["final_mse"].get(v), row["final_mse"].get("base")) if row else "-")
        nd = " / ".join(str(out.get(m, {}).get(str(L), {}).get("n_drop", "-")) for m in ("0", "1"))
        L1.append(f"| {L} | {nd} | " + " | ".join(cells) + " |")
        for mid in ("0", "1"):
            row = out.get(mid, {}).get(str(L))
            if not row or row["C_rel_err"].get("t") is None:
                continue
            f = lambda d_, fmt_="%.4f": (fmt_ % d_["base"]) + " -> " + (fmt_ % d_["t"]) if d_.get("t") is not None else "-"  # noqa: E731
            L2.append(f"| {mid} | {L} | {f(row['C_rel_err'])} | {f(row['var_next_rms_rel_err'])} | {f(row['var_next_mean_rel_err'], '%+.4f')} | "
                      f"{f(row['t3_next_rel_rms_err'])} | {f(row['next_layer_mse'], '%.3e')} |")
    return "\n".join(L1 + L2) + "\n"


if __name__ == "__main__":
    main()
