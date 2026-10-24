"""Band study report (research/wu/mech): the engine's 3 T^T entries against Monte Carlo truth, kept (in-band) and dropped
(out-of-band) apart, per layer; the diagonal t3 the same way; the oracle-swap MSE tables from runs/*.jsonl.
Writes research/wu/mech/results.json and research/wu/mech/tables.md (the README's tables).

    .venv-fs0123/bin/python research/wu/mech/mech_report.py

Entry groups at layer t (M1 = 3 T^T of the band-off engine, rows i the single index, columns j the doubled index;
kr / kc the rows / columns the submitted rule keeps at that layer's alpha):
  keep           i in Ia and j in Ib (off the diagonal)
  drop           the other off-diagonal entries (the ones the band sets to zero)
  drop_row_only  i not in Ia, j in Ib (the single index below alpha -1.5)
  drop_col_only  i in Ia, j not in Ib (the doubled index outside |alpha| < 3)
  drop_both      both
Engine value of TT[i, j] = kappa3(z_i, z_j, z_j): (M1 - F) / 3, F the K4 star fold's share (r25a k4f); "apa" is M1 as
the engine computes it (one APA level), "exact" the exact product of the same operands. A Monte Carlo entry is
"resolved" when |TT_mc| > 3 SE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAYERS = list(range(2, 15))
def ids_in(recdir):
    """Networks with Monte Carlo truth and a band-off recording in recdir (rec: r38, rec_opr30: opr30, rec_r37, ...)."""
    return [i for i in range(60) if all((HERE / d / f).exists() for d, f in (
        ("mc", f"mc_mini{i}.npz"), (recdir, f"rec_mech_nott_mini{i}.npz")))]
GROUPS = ("keep", "drop", "drop_row_only", "drop_col_only", "drop_both")


def stats(te, tt, se, g=None):
    """te, tt, se: 1-d arrays over one group's entries (engine, Monte Carlo, its standard error); g: 1 / (sigma_i
    sigma_j^2) per entry (standardization)."""
    n = int(te.size)
    out = {"n": n}
    if n == 0:
        return out
    res = np.abs(tt) > 3.0 * se
    out["n_resolved"] = int(res.sum())
    out["frac_resolved"] = float(res.mean())
    s2 = float((se ** 2).mean())
    tt2 = float((tt ** 2).mean())
    out["corr_all"] = float(np.corrcoef(te, tt)[0, 1]) if n > 2 and te.std() > 0 and tt.std() > 0 else None
    # noise-corrected: E[(te - tt_mc)^2] = (te - tt)^2 + se^2, E[tt_mc^2] = tt^2 + se^2
    out["rel_rms_err"] = float(np.sqrt(max(float(((te - tt) ** 2).mean()) - s2, 0.0) / max(tt2 - s2, 1e-300)))
    out["rms_engine_over_rms_truth"] = float(np.sqrt(float((te ** 2).mean()) / max(tt2 - s2, 1e-300)))
    out["slope_engine_on_truth"] = float(float((te * tt).mean()) / max(tt2 - s2, 1e-300))
    out["truth_rms_over_se_rms"] = float(np.sqrt(tt2 / max(s2, 1e-300)))
    if g is not None:
        out["truth_rms_std"] = float(np.sqrt(((tt * g) ** 2).mean()))
        out["err_rms_std"] = float(np.sqrt(max(float((((te - tt) * g) ** 2).mean()) - float(((se * g) ** 2).mean()), 0.0)))
    if res.sum() >= 3:
        r, m = te[res], tt[res]
        rel = np.abs(r - m) / np.abs(m)
        out["rel_err_median"] = float(np.median(rel))
        out["rel_err_mean"] = float(rel.mean())
        out["sign_agree"] = float((np.sign(r) == np.sign(m)).mean())
        out["corr_resolved"] = float(np.corrcoef(r, m)[0, 1]) if r.std() > 0 and m.std() > 0 else None
        out["bias_mean"] = float((r - m).mean())
        out["bias_over_mean_abs_truth"] = float((r - m).mean() / np.abs(m).mean())
        out["mean_truth_over_mean_abs_truth"] = float(m.mean() / np.abs(m).mean())
        if g is not None:
            out["bias_std"] = float(((r - m) * g[res]).mean())
    return out


def entry_tables(recdir="rec"):
    tabs = {}
    for mid in ids_in(recdir):
        mc = np.load(HERE / "mc" / f"mc_mini{mid}.npz")
        rb = np.load(HERE / recdir / f"rec_mech_nott_mini{mid}.npz")
        pp = HERE / recdir / f"rec_mech_prod_mini{mid}.npz"
        rp = np.load(pp) if pp.exists() else None  # band-on recording (rec, rec_opr30 only)
        p7 = HERE / recdir / f"rec_mech_nos7_mini{mid}.npz"
        rs = np.load(p7) if p7.exists() else None
        per = {}
        for t in LAYERS:
            TT = mc[f"TT_{t}"].astype(np.float64)
            SE = mc[f"SE_{t}"].astype(np.float64)
            sig = np.sqrt(mc["var_z"][t])
            n = TT.shape[0]
            off = ~np.eye(n, dtype=bool)
            kr, kc = rb[f"kr_{t}"], rb[f"kc_{t}"]
            Kp = np.outer(kr, kc)
            masks = {"keep": Kp & off, "drop": ~Kp & off, "drop_row_only": np.outer(~kr, kc) & off,
                     "drop_col_only": np.outer(kr, ~kc) & off, "drop_both": np.outer(~kr, ~kc) & off}
            F = rb[f"F_{t}"].astype(np.float64)
            M1 = rb[f"M1_{t}"].astype(np.float64)
            Mx = rb[f"Mx_{t}"].astype(np.float64)
            Te = {"apa": (M1 - F) / 3.0, "exact": (Mx - F) / 3.0}
            G = 1.0 / np.outer(sig, sig * sig)
            a = rb[f"a_{t}"].astype(np.float64)
            amc = mc["mean_z"][t] / sig
            row = {"Pa": int(kr.sum()), "Pb": int(kc.sum()), "n_drop": int(masks["drop"].sum()),
                   "alpha_engine_minus_mc_rms": float(np.sqrt(((a - amc) ** 2).mean())),
                   "mc_alpha_dropped_rows_max": float(amc[~kr].max()) if (~kr).any() else None,
                   "mc_alpha_dropped_cols_abs_min": float(np.abs(amc[~kc]).min()) if (~kc).any() else None,
                   "engine_alpha_dropped_rows_max": float(a[~kr].max()) if (~kr).any() else None,
                   "engine_alpha_dropped_cols_abs_min": float(np.abs(a[~kc]).min()) if (~kc).any() else None}
            # orientation check: the engine's M1 / 3 against TT (kappa3(z_i, z_j, z_j)) and against its transpose
            o = masks["keep"]
            row["orient_corr_TT"] = float(np.corrcoef(Te["exact"][o], TT[o])[0, 1])
            row["orient_corr_TT_transposed"] = float(np.corrcoef(Te["exact"][o], TT.T[o])[0, 1])
            for gname, m in masks.items():
                if not m.any():
                    continue
                gi, gj = np.nonzero(m)
                d = {k: stats(te[m], TT[m], SE[m], G[m]) for k, te in Te.items()}
                d["fold_rms_over_truth_rms"] = float(np.sqrt(((F[m] / 3.0) ** 2).mean() / max((TT[m] ** 2).mean() - (SE[m] ** 2).mean(), 1e-300)))
                d["apa_err_rms_over_exact_rms"] = float(np.sqrt(((M1[m] - Mx[m]) ** 2).mean() / max((Mx[m] ** 2).mean(), 1e-300)))
                row[gname] = d
            # diagonal t3 = kappa3(z_i, z_i, z_i): band-off engine (diagonal of the APA product; the exact sum), band on
            t3m, t3s = mc["t3"][t], mc["t3_se"][t]
            sg3 = 1.0 / sig ** 3
            dg = {"nott_apa": rb[f"diag_{t}"] / 3.0, "nott_exact": np.diagonal(Mx) / 3.0}
            if rp is not None:
                dg["prod"] = rp[f"diag_{t}"] / 3.0
            if rs is not None:
                dg["nos7"] = rs[f"diag_{t}"] / 3.0
            dd = {}
            for k, v in dg.items():
                kk = rp[f"kc_{t}"] if k == "prod" else (rs[f"kc_{t}"] if k == "nos7" else kc)
                dd[k] = {"in_band": stats(v[kk], t3m[kk], t3s[kk], sg3[kk]),
                         "out_band": stats(v[~kk], t3m[~kk], t3s[~kk], sg3[~kk]) if (~kk).any() else None}
            kcp = rp[f"kc_{t}"] if rp is not None else None
            if rp is not None:
                dd["prod_band"] = {"Pa": int(rp[f"kr_{t}"].sum()), "Pb": int(kcp.sum())}
            if rp is not None and (~kcp).any():
                dd["prod_compressed_fold_diag_out_band_rms_std"] = float(np.sqrt(((4.0 * rp[f"DGFc_{t}"][~kcp] / 3.0 * sg3[~kcp]) ** 2).mean()))
            row["diag"] = dd
            per[str(t)] = row
        tabs[str(mid)] = per
    return tabs


def runs_table():
    rows = []
    for f in sorted((HERE / "runs").glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip().startswith("{"):
                continue
            r = json.loads(ln)
            if "meta" in r or "done" in r:
                continue
            r["file"] = f.name
            rows.append(r)
    return rows


def fmt(x, f="%.3f"):
    return "-" if x is None else (f % x)


def tables_md(E, R, runs=True):
    L = []
    L.append("### Band sets (band-off engine's own alpha, the submitted rule)\n")
    L.append("| mini | t | Pa | Pb | dropped entries | max MC alpha of dropped rows | min MC abs alpha of dropped cols |")
    L.append("|---|---|---|---|---|---|---|")
    for mid, per in E.items():
        for t, row in per.items():
            L.append(f"| {mid} | {t} | {row['Pa']} | {row['Pb']} | {row['n_drop']} | {fmt(row['mc_alpha_dropped_rows_max'], '%.2f')} | {fmt(row['mc_alpha_dropped_cols_abs_min'], '%.2f')} |")
    for kind in ("apa", "exact"):
        L.append(f"\n### Engine TT vs Monte Carlo, kept vs dropped (engine values: {kind})\n")
        L.append("| mini | t | group | n | resolved | rel err median | rel err mean | sign agree | corr (res.) | corr (all) | bias / mean abs truth | slope eng~truth | rel rms err | fold rms / truth rms |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for mid, per in E.items():
            for t, row in per.items():
                for g in GROUPS:
                    if g not in row:
                        continue
                    s = row[g][kind]
                    L.append(f"| {mid} | {t} | {g} | {s['n']} | {s.get('n_resolved', 0)} | {fmt(s.get('rel_err_median'))} | {fmt(s.get('rel_err_mean'))} | "
                             f"{fmt(s.get('sign_agree'))} | {fmt(s.get('corr_resolved'))} | {fmt(s.get('corr_all'))} | {fmt(s.get('bias_over_mean_abs_truth'), '%+.3f')} | "
                             f"{fmt(s.get('slope_engine_on_truth'))} | {fmt(s.get('rel_rms_err'))} | {fmt(row[g]['fold_rms_over_truth_rms'])} |")
    L.append("\n### Diagonal t3 vs Monte Carlo (in-band vs out-of-band neurons)\n")
    L.append("| mini | t | engine | group | n | resolved | rel err median | sign agree | corr (all) | slope | rel rms err |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for mid, per in E.items():
        for t, row in per.items():
            for k, v in row["diag"].items():
                if not isinstance(v, dict) or "in_band" not in v:
                    continue
                for g in ("in_band", "out_band"):
                    s = v[g]
                    if s is None:
                        continue
                    L.append(f"| {mid} | {t} | {k} | {g} | {s['n']} | {s.get('n_resolved', 0)} | {fmt(s.get('rel_err_median'))} | {fmt(s.get('sign_agree'))} | "
                             f"{fmt(s.get('corr_all'))} | {fmt(s.get('slope_engine_on_truth'))} | {fmt(s.get('rel_rms_err'))} |")
    if not runs:
        return "\n".join(L) + "\n"
    L.append("\n### Final-layer MSE per run\n")
    L.append("| tag | file | spec | override | mini 0 | mini 1 |")
    L.append("|---|---|---|---|---|---|")
    by = {}
    for r in R:
        tag = r.get("tag") or r["file"].replace(".jsonl", "")
        key = (tag, r["file"], json.dumps(r.get("spec", "")), r.get("override", ""))
        by.setdefault(key, {})[r["mlp_id"]] = r["final_mse"]
    for (tag, f, sp, ov), v in by.items():
        L.append(f"| {tag} | {f} | {sp} | {ov} | {fmt(v.get(0), '%.4e')} | {fmt(v.get(1), '%.4e')} |")
    return "\n".join(L) + "\n"


def compact_md(E, R):
    """The README's tables: per layer, kept vs dropped (exact engine values; APA changes nothing material), the diagonal,
    and the final-layer MSE of every run with its ratio to the band-off engine (B) and to the clean band (dkz)."""
    return compact_entries(E) + compact_runs(R)


def compact_entries(E, tag=""):
    """Per layer, kept vs dropped and the diagonal; tag marks the header's first column (e.g. " (opr30)")."""
    L = ["#### Off-diagonal entries TT[i, j] = kappa3(z_i, z_j, z_j): engine (exact product, fold removed) vs Monte Carlo\n",
         "res: resolved entries (|TT_mc| > 3 SE), share of the group; med: median |eng - mc| / |mc| over resolved; sign: sign "
         "agreement over resolved; corr: Pearson over the group; slope: E[eng mc] / E[mc^2] (noise-corrected), the engine's "
         "scale relative to the truth.\n",
         f"| mini{tag} | t | Pa / Pb | dropped | kept res | kept med | kept sign | kept corr | kept slope | drop res | drop med | drop sign | drop corr | drop slope | rows-only slope / corr | cols-only slope / corr |",
         "|" + "---|" * 16]
    for mid, per in E.items():
        for t, row in per.items():
            k = row["keep"]["exact"]
            d = row.get("drop", {}).get("exact")
            ro = row.get("drop_row_only", {}).get("exact")
            co = row.get("drop_col_only", {}).get("exact")
            cell = lambda s_, key, f="%.2f": fmt(s_.get(key), f) if s_ else "-"  # noqa: E731
            L.append(f"| {mid} | {t} | {row['Pa']} / {row['Pb']} | {row['n_drop']} | {cell(k, 'frac_resolved')} | {cell(k, 'rel_err_median')} | "
                     f"{cell(k, 'sign_agree')} | {cell(k, 'corr_all')} | {cell(k, 'slope_engine_on_truth')} | {cell(d, 'frac_resolved')} | "
                     f"{cell(d, 'rel_err_median')} | {cell(d, 'sign_agree')} | {cell(d, 'corr_all')} | {cell(d, 'slope_engine_on_truth')} | "
                     f"{cell(ro, 'slope_engine_on_truth')} / {cell(ro, 'corr_all')} | {cell(co, 'slope_engine_on_truth')} / {cell(co, 'corr_all')} |")
    L += ["\n#### Diagonal t3 = kappa3(z_i, z_i, z_i): slope (engine on truth) and corr, neurons in the column band vs outside\n",
          f"| mini{tag} | t | band off: in | band off: out | band on (submitted): in | band on: out | band on without s7: out |",
          "|---|---|---|---|---|---|---|"]
    for mid, per in E.items():
        for t, row in per.items():
            dg = row["diag"]
            c = lambda s_: "-" if not s_ else "%.2f / %.2f" % (s_["slope_engine_on_truth"], s_["corr_all"])  # noqa: E731
            pr = dg.get("prod") or {}
            L.append(f"| {mid} | {t} | {c(dg['nott_exact']['in_band'])} | {c(dg['nott_exact']['out_band'])} | {c(pr.get('in_band'))} | "
                     f"{c(pr.get('out_band'))} | {c(dg.get('nos7', {}).get('out_band') if dg.get('nos7') else None)} |")
    return "\n".join(L) + "\n"


def compact_runs(R):
    L = []
    by = {}
    for r in R:
        tag = r.get("tag") or r["file"].replace(".jsonl", "")
        by.setdefault((tag, r["file"]), {})[r["mlp_id"]] = r["final_mse"]
    # ratios within a file's own family: r38 runs to r38's band-off engine (rec.jsonl) and dkz (swap.jsonl), opr30 runs
    # to opr30's (opr30_rec.jsonl); the r37 / r38x band on-off pairs are compared in the README's gain table instead
    refs = {f_: d_ for (t_, f_), d_ in by.items() if t_ == "rec_mech_nott"}
    cbs = {f_: d_ for (t_, f_), d_ in by.items() if t_ == "dkz"}

    def family(f_):
        if "opr30" in f_:
            return refs.get("opr30_rec.jsonl", {}), {}
        if f_.startswith(("r37_", "r38x_")):
            return {}, {}
        return refs.get("rec.jsonl", {}), cbs.get("swap.jsonl", {})
    L += ["\n#### Final-layer MSE (x 1e-8) and ratios\n",
          "| run | file | mini 0 | mini 1 | / B mini 0 | / B mini 1 | / dkz mini 0 | / dkz mini 1 |", "|---|---|---|---|---|---|---|---|"]
    for (tag, f_), d_ in by.items():
        if not ({0, 1} & set(d_)):
            continue
        ref, cb = family(f_)
        g = lambda m, den: fmt(d_[m] / den[m], "%.3f") if (m in d_ and m in den) else "-"  # noqa: E731
        L.append(f"| {tag} | {f_} | {fmt(d_.get(0) and d_[0] * 1e8, '%.4f')} | {fmt(d_.get(1) and d_[1] * 1e8, '%.4f')} | "
                 f"{g(0, ref)} | {g(1, ref)} | {g(0, cb)} | {g(1, cb)} |")
    others = sorted({m for d_ in by.values() for m in d_} - {0, 1})
    if others:
        L += ["\n#### Other networks: final-layer MSE (x 1e-8)\n", "| run | file | " + " | ".join(f"mini {m}" for m in others) + " |",
              "|---|---|" + "---|" * len(others)]
        for (tag, f_), d_ in by.items():
            if set(d_) & set(others):
                L.append(f"| {tag} | {f_} | " + " | ".join(fmt(d_.get(m) and d_[m] * 1e8, "%.4f") for m in others) + " |")
    return "\n".join(L) + "\n"


OTHER = [("opr30", "rec_opr30", "opr30 file (submission 336955; est_opr30 cuts)"),
         ("r37", "rec_r37", "r37 file (opr30 + rank 256 + L4; band-off cut est_r37/mech_nott)"),
         ("r38noext", "rec_r38noext", "r38 without the extraction (law off; est_r38x/mech_nott_noext)"),
         ("r38lawon", "rec_r38lawon", "r38 with the law back (extraction on; est_r38x/mech_nott_lawon)")]


def main() -> None:
    E = entry_tables()
    R = runs_table()
    other = {name: entry_tables(d) for name, d, _ in OTHER if (HERE / d).exists()}
    res = {"entries": E, "runs": R}
    res.update({f"entries_{name}": v for name, v in other.items()})
    if (HERE / "overlap.json").exists():  # mech_overlap.py: the band's change of the final means against the others'
        res["overlap"] = json.loads((HERE / "overlap.json").read_text(encoding="utf-8"))
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    t = tables_md(E, R)
    c = compact_md(E, R)
    for name, d, title in OTHER:
        if other.get(name):
            t += f"\n## {title}\n\n" + tables_md(other[name], R, runs=False)
            c += f"\n## {title}\n\n" + compact_entries(other[name], f" ({name})")
    (HERE / "tables.md").write_text(t, encoding="utf-8")
    (HERE / "tables_compact.md").write_text(c, encoding="utf-8")
    print("wrote", HERE / "results.json", HERE / "tables.md", HERE / "tables_compact.md")


if __name__ == "__main__":
    main()
