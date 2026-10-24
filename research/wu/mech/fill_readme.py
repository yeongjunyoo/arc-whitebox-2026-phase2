"""Fill README.md's table placeholders from the report outputs (tables_compact.md, local.md, results.json) and the text
blocks below. Run on the Mac after mech_report.py and mech_local.py:  .venv-fs0123/bin/python research/wu/mech/fill_readme.py
(reads README.template.md, writes README.md)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def section(md, start, stop=None):
    lines = md.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith(start):
            on = True
            continue
        if on and stop and ln.startswith(stop):
            break
        if on:
            out.append(ln)
    return "\n".join(x for x in out if x.strip() != "" or out).strip()


def table_rows(md, header_start):
    lines = md.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(header_start))
    out = []
    for ln in lines[i:]:
        if not ln.startswith("|"):
            break
        out.append(ln)
    return "\n".join(out)


SWAP_ROWS = [  # (file, tag, label)
    ("ref_v_engine.jsonl", None, "v_engine: band on, as submitted (a)"),
    ("ref_v_engine_nos7.jsonl", None, "v_engine_nos7: band on without s7"),
    ("ref_v_engine_nott.jsonl", None, "v_engine_nott: band off (b) = B"),
    ("swap.jsonl", "z", "z: dropped entries 0 (clean band)"),
    ("swap.jsonl", "dz", "dz: z + diagonal exact"),
    ("swap.jsonl", "dkz", "dkz: z + diagonal and kept entries exact"),
    ("swap.jsonl", "d", "d: band off, diagonal exact"),
    ("swap.jsonl", "dkx", "dkx: band off, exact product (no APA) on band layers"),
    ("swap.jsonl", "xall", "xall: band off, exact product on layers 1-14"),
    ("swap.jsonl", "enf", "enf: dropped = engine minus fold share"),
    ("swap.jsonl", "dkxnf", "dkxnf: dkx, dropped minus fold share"),
    ("swap.jsonl", "m", "m: dropped = Monte Carlo truth + fold (c)"),
    ("swap.jsonl", "mnf", "mnf: dropped = Monte Carlo truth"),
    ("swap.jsonl", "dkm", "dkm: dkz with dropped = truth + fold"),
    ("swap.jsonl", "dkmnf", "dkmnf: dkz with dropped = truth"),
    ("swap.jsonl", "dkmn", "dkmn: dkm with the Monte Carlo noise doubled"),
    ("swap.jsonl", "dmm", "dmm: every off-diagonal entry = truth + fold"),
    ("scale.jsonl", "s125", "s125: band off, off-diagonal x 1.25 (layers 1-14)"),
    ("scale.jsonl", "s090", "s090: band off, off-diagonal x 0.9"),
    ("scale.jsonl", "s075", "s075: band off, off-diagonal x 0.75"),
    ("scale.jsonl", "s050", "s050: band off, off-diagonal x 0.5"),
    ("scale.jsonl", "s000", "s000: band off, off-diagonal x 0"),
    ("scale.jsonl", "k075", "k075: band off, kept entries x 0.75"),
    ("scale.jsonl", "mm075", "mm075: dmm x 0.75"),
    ("scale.jsonl", "mm050", "mm050: dmm x 0.5"),
    ("scale.jsonl", "dd", "dd: band off, diagonal = Monte Carlo 3 t3 (layers 1-14)"),
    ("scale.jsonl", "ddmm", "ddmm: dd + every off-diagonal entry = truth + fold"),
]


def swap_table(runs):
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]
    B = by.get(("ref_v_engine_nott.jsonl", None), {})
    Z = by.get(("swap.jsonl", "dkz"), {})
    out = ["| run | mini 0 (x 1e-8) | mini 1 | / B mini 0 | / B mini 1 | / dkz mini 0 | / dkz mini 1 |",
           "|---|---|---|---|---|---|---|"]
    for f, tag, label in SWAP_ROWS:
        d = by.get((f, tag))
        if not d:
            continue
        g = lambda m, den: "%.3f" % (d[m] / den[m]) if (m in d and m in den) else "-"  # noqa: E731
        v = lambda m: "%.4f" % (d[m] * 1e8) if m in d else "-"  # noqa: E731
        out.append(f"| {label} | {v(0)} | {v(1)} | {g(0, B)} | {g(1, B)} | {g(0, Z)} | {g(1, Z)} |")
    return "\n".join(out)


SWAP23_ROWS = [  # (file, tag, label) on mini 2 and 3
    ("rec23.jsonl", "rec_mech_prod", "v_engine: band on, as submitted"),
    ("rec23.jsonl", "rec_mech_nott", "v_engine_nott: band off = B"),
    ("swap23.jsonl", "z", "z: dropped entries 0"),
    ("swap23.jsonl", "dkz", "dkz: clean band"),
    ("swap23.jsonl", "dkx", "dkx: band off, exact product"),
    ("swap23.jsonl", "m", "m: dropped = truth + fold"),
    ("swap23.jsonl", "dkm", "dkm: dkz with dropped = truth + fold"),
    ("swap23.jsonl", "dkmnf", "dkmnf: dkz with dropped = truth"),
    ("swap23.jsonl", "dmm", "dmm: every off-diagonal entry = truth + fold"),
]


def swap23_table(runs):
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]
    B = by.get(("rec23.jsonl", "rec_mech_nott"), {})
    out = ["| run | mini 2 (x 1e-8) | mini 3 | / B mini 2 | / B mini 3 |", "|---|---|---|---|---|"]
    for f, tag, label in SWAP23_ROWS:
        d = by.get((f, tag))
        if not d:
            continue
        g = lambda m: "%.3f" % (d[m] / B[m]) if (m in d and m in B) else "-"  # noqa: E731
        v = lambda m: "%.4f" % (d[m] * 1e8) if m in d else "-"  # noqa: E731
        out.append(f"| {label} | {v(2)} | {v(3)} | {g(2)} | {g(3)} |")
    return "\n".join(out)


def ext_table(runs):
    """Band on (P) / band off (B) per public mini for the r38 cuts and the opr30 cuts (estimators/wu2), and the clean band."""
    col = {}
    for r in runs:
        f, tag, m = r["file"], r.get("tag"), r["mlp_id"]
        key = {("ref_v_engine.jsonl", None): "rP", ("ext_r38_v_engine.jsonl", None): "rP",
               ("ref_v_engine_nott.jsonl", None): "rB", ("ext_r38_v_engine_nott.jsonl", None): "rB",
               ("swap.jsonl", "dkz"): "rZ", ("ext_dkz.jsonl", "dkz"): "rZ",
               ("ext_opr30_v_engine.jsonl", None): "oP", ("ext_opr30_v_engine_nott.jsonl", None): "oB"}.get((f, tag))
        if key:
            col.setdefault(m, {})[key] = r["final_mse"]
    ms = sorted(m for m, d in col.items() if "rP" in d or "oP" in d)
    out = ["| mini | r38 band on (x 1e-8) | r38 band off | r38 on / off | r38 clean band / off | opr30 band on | opr30 band off | opr30 on / off |",
           "|---|---|---|---|---|---|---|---|"]
    rr, oo, zz = [], [], []
    for m in ms:
        d = col[m]
        f = lambda k: "%.4f" % (d[k] * 1e8) if k in d else "-"  # noqa: E731
        q = lambda a, b: (d[a] / d[b]) if (a in d and b in d) else None  # noqa: E731
        r1, r2, r3 = q("rP", "rB"), q("rZ", "rB"), q("oP", "oB")
        rr += [r1] if r1 else []
        zz += [r2] if r2 else []
        oo += [r3] if r3 else []
        g = lambda x: "%.3f" % x if x else "-"  # noqa: E731
        out.append(f"| {m} | {f('rP')} | {f('rB')} | {g(r1)} | {g(r2)} | {f('oP')} | {f('oB')} | {g(r3)} |")
    import math
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else None  # noqa: E731
    summ = lambda v: ("geometric mean %.3f, band on better on %d of %d" % (gm(v), sum(x < 1 for x in v), len(v))) if v else "-"  # noqa: E731
    out.append(f"| all | | | {summ(rr)} | {summ(zz)} | | | {summ(oo)} |")
    return "\n".join(out)


def ratios(runs):
    """The short answer's computed numbers: clean band and submitted band over band off on minis 0-3, the r38 band on /
    off over minis 0-15, and the scale test's ratios."""
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]
    B = {**by.get(("ref_v_engine_nott.jsonl", None), {}), **by.get(("rec23.jsonl", "rec_mech_nott"), {})}
    Z = {**by.get(("swap.jsonl", "z"), {}), **by.get(("swap23.jsonl", "z"), {})}
    P = {**by.get(("ref_v_engine.jsonl", None), {}), **by.get(("rec23.jsonl", "rec_mech_prod"), {})}
    f = lambda D: ", ".join("x%.3f" % (D[m] / B[m]) for m in sorted(D) if m in B and m <= 3)  # noqa: E731
    out = {"Z_RANGE": f(Z) + " (minis 0, 1, 2, 3)", "P_RANGE": f(P)}
    rP = {**by.get(("ref_v_engine.jsonl", None), {}), **by.get(("ext_r38_v_engine.jsonl", None), {})}
    rB = {**by.get(("ref_v_engine_nott.jsonl", None), {}), **by.get(("ext_r38_v_engine_nott.jsonl", None), {})}
    v = [rP[m] / rB[m] for m in rP if m in rB]
    import math
    out["EXT_R38"] = ("geometric mean x%.3f, band on better on %d of %d" % (math.exp(sum(map(math.log, v)) / len(v)),
                                                                          sum(x < 1 for x in v), len(v))) if v else "pending"
    sc = []
    for tag, lab in (("s000", "0"), ("s050", "0.5"), ("s075", "0.75"), ("s090", "0.9"), ("s125", "1.25")):
        d = by.get(("scale.jsonl", tag), {})
        rr = [d[m] / B[m] for m in (0, 1) if m in d and m in B]
        if rr:
            sc.append("s = %s: x%.1f / x%.1f" % (lab, rr[0], rr[-1]))
    out["SCALE_LINE"] = "; ".join(sc) + " (mini 0 / 1); s = 1 is B itself" if sc else "pending"
    d = by.get(("scale.jsonl", "dd"), {})
    out["DD_LINE"] = ("x%.0f / x%.0f of B" % (d[0] / B[0], d[1] / B[1])) if (0 in d and 1 in d) else "pending"
    dm = by.get(("scale.jsonl", "ddmm"), {})
    if 0 in dm and 1 in dm:
        out["DD_LINE"] += ", and with the truth off the diagonal as well (ddmm) x%.0f / x%.0f" % (dm[0] / B[0], dm[1] / B[1])
    return out


OPR30_ROWS = [  # (file, tag, label) on minis 0-3 for the opr30 file
    ("opr30_rec.jsonl", "rec_mech_prod", "a: band on, as submitted (wu2 v_engine)"),
    ("opr30_rec.jsonl", "rec_mech_nott", "B: band off (wu2 v_engine_nott)"),
    ("opr30_swap.jsonl", "z", "z: dropped entries 0 (clean band)"),
    ("opr30_swap.jsonl", "m", "m: dropped = Monte Carlo truth + fold"),
    ("opr30_swap.jsonl", "dmm", "dmm: every off-diagonal entry = truth + fold"),
    ("opr30_swap.jsonl", "s090", "s090: band off, off-diagonal x 0.9 (layers 1-14)"),
    ("opr30_swap.jsonl", "s125", "s125: band off, off-diagonal x 1.25"),
    ("opr30_swap.jsonl", "dd", "dd: band off, diagonal = Monte Carlo 3 t3 (layers 1-14)"),
    ("opr30_swap.jsonl", "mz", "mz: kept entries = truth + fold, dropped 0, diagonal exact"),
    ("opr30_swap.jsonl", "zlo", "zlo: dropped entries 0 at layers 2-10 only"),
    ("opr30_swap.jsonl", "zhi", "zhi: dropped entries 0 at layers 11-14 only"),
    ("opr30_swap.jsonl", "zrow", "zrow: 0 only where the row is out (a_i < -1.5)"),
    ("opr30_swap.jsonl", "zcol", "zcol: 0 only where the column is out (|a_j| >= 3)"),
    ("opr30_swap.jsonl", "ddmm", "ddmm: dd + every off-diagonal entry = truth + fold"),
    ("opr30_swap.jsonl", "ddmz", "ddmz: dd + kept = truth + fold, dropped 0"),
]


def opr30_table(runs):
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]
    B = by.get(("opr30_rec.jsonl", "rec_mech_nott"), {})
    ms = [0, 1, 2, 3]
    out = ["| run | " + " | ".join(f"mini {m} (x 1e-8)" for m in ms) + " | " + " | ".join(f"/ B mini {m}" for m in ms) + " |",
           "|---|" + "---|" * (2 * len(ms))]
    for f, tag, label in OPR30_ROWS:
        d = by.get((f, tag))
        if not d:
            continue
        v = lambda m: "%.4f" % (d[m] * 1e8) if m in d else "-"  # noqa: E731
        g = lambda m: "%.3f" % (d[m] / B[m]) if (m in d and m in B) else "-"  # noqa: E731
        out.append(f"| {label} | " + " | ".join(v(m) for m in ms) + " | " + " | ".join(g(m) for m in ms) + " |")
    return "\n".join(out)


GAIN_COLS = [  # (label, band-on file, band-off file); a None tag means an ablate.py file
    ("opr30", ["ext_opr30_v_engine.jsonl"], ["ext_opr30_v_engine_nott.jsonl"]),
    ("r37 = opr30 + rank 256 + L4", ["r37_v_engine.jsonl"], ["r37_v_engine_nott.jsonl"]),
    ("r38 with both halves of R38 undone (mini 0)", ["r38x_v_engine_lawon_noext.jsonl"], ["r38x_v_engine_nott_lawon_noext.jsonl"]),
    ("r38 without the extraction (law off)", ["r38x_v_engine_noext.jsonl"], ["r38x_v_engine_nott_noext.jsonl"]),
    ("r38 with the law back (extraction on)", ["r38x_v_engine_lawon.jsonl"], ["r38x_v_engine_nott_lawon.jsonl"]),
    ("r38 as submitted (law off, extraction on)", ["ref_v_engine.jsonl", "ext_r38_v_engine.jsonl"],
     ["ref_v_engine_nott.jsonl", "ext_r38_v_engine_nott.jsonl"]),
]


XPART_COLS = [  # the r38 file with one of the extraction's four parts off (minis 0-7), next to r38 and r38 without it
    ("r38 as submitted", ["ref_v_engine.jsonl", "ext_r38_v_engine.jsonl"], ["ref_v_engine_nott.jsonl", "ext_r38_v_engine_nott.jsonl"]),
    ("without the C' part", ["r38x_v_engine_noxc.jsonl"], ["r38x_v_engine_nott_noxc.jsonl"]),
    ("without the dA (K4 rows) part", ["r38x_v_engine_noxa.jsonl"], ["r38x_v_engine_nott_noxa.jsonl"]),
    ("without the E (K3 leg) part", ["r38x_v_engine_noxw.jsonl"], ["r38x_v_engine_nott_noxw.jsonl"]),
    ("without the feed pair", ["r38x_v_engine_noxf.jsonl"], ["r38x_v_engine_nott_noxf.jsonl"]),
    ("the whole extraction at half strength", ["r38x_v_engine_x05.jsonl"], ["r38x_v_engine_nott_x05.jsonl"]),
    ("without the extraction (all four)", ["r38x_v_engine_noext.jsonl"], ["r38x_v_engine_nott_noext.jsonl"]),
]


def gain_table(runs, spec=None, only=None):
    """Band on / off final-layer MSE ratio per public mini for each file, and the summary rows (spec: the column list,
    GAIN_COLS by default; only: the minis to show, default every mini that some column has)."""
    import math
    byf = {}
    for r in runs:
        if r.get("tag") is None:
            byf.setdefault(r["file"], {})[r["mlp_id"]] = r["final_mse"]
    cols = []
    for lab, on_f, off_f in (spec or GAIN_COLS):
        on, off = {}, {}
        for f in on_f:
            on.update(byf.get(f, {}))
        for f in off_f:
            off.update(byf.get(f, {}))
        cols.append((lab, on, off))
    ms = sorted({m for _, on, off in cols for m in on if m in off and (only is None or m in only)})
    out = ["| mini | " + " | ".join(lab for lab, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in ms:
        out.append(f"| {m} | " + " | ".join(("%.3f" % (on[m] / off[m])) if (m in on and m in off) else "-" for _, on, off in cols) + " |")
    gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    row1, row2, row3 = [], [], []
    for _, on, off in cols:
        v = [on[m] / off[m] for m in ms if m in on and m in off]
        row1.append("%.3f" % gm(v) if v else "-")
        row2.append(("%d of %d" % (sum(x < 1 for x in v), len(v))) if v else "-")
        w = [on[m] for m in ms if m in on]
        row3.append(("%.3f" % (gm(w) * 1e8)) if len(w) == len(ms) else ("%.3f (%d net%s)" % (gm(w) * 1e8, len(w), "" if len(w) == 1 else "s") if w else "-"))
    out.append("| geometric mean of on / off | " + " | ".join(row1) + " |")
    out.append("| band on better | " + " | ".join(row2) + " |")
    out.append("| band-on MSE, geometric mean (x 1e-8) | " + " | ".join(row3) + " |")
    return "\n".join(out)


XFILES = [  # (label, results.json key): the entry statistics of each file's band-off engine on minis 0-3
    ("opr30", "entries_opr30"),
    ("r37 = opr30 + rank 256 + L4", "entries_r37"),
    ("r38 without the extraction (law off)", "entries_r38noext"),
    ("r38 with the law back (extraction on)", "entries_r38lawon"),
    ("r38 as submitted", "entries"),
]


def _mean(v):
    return ("%.2f" % (sum(v) / len(v))) if v else "-"


def layer_table(E):
    """Per layer, kept and dropped entries of one file's band-off engine against the truth, means over the minis
    recorded (0-3): median relative error, sign agreement, corr, slope, signed bias / mean abs truth."""
    keys = [("rel_err_median", "%.2f"), ("sign_agree", "%.2f"), ("corr_all", "%.2f"), ("slope_engine_on_truth", "%.2f"),
            ("bias_over_mean_abs_truth", "%+.2f")]
    out = ["| t | kept med | kept sign | kept corr | kept slope | kept bias | dropped med | dropped sign | dropped corr | "
           "dropped slope | dropped bias |", "|---|" + "---|" * 10]
    for t in range(2, 15):
        cells = []
        for g in ("keep", "drop"):
            for k, f in keys:
                v = [((per.get(str(t)) or {}).get(g) or {}).get("exact", {}).get(k) for per in E.values()]
                v = [x for x in v if x is not None]
                m_ = sum(v) / len(v) if v else None
                cells.append("-" if m_ is None else ("0.00" if abs(m_) < 0.005 and "+" in f else f % m_))
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def xfile_table(res):
    """Per layer, each file's band-off engine against the truth, means over the minis recorded (0-3): kept slope /
    dropped slope (dropped corr)."""
    cols = [(lab, res[key]) for lab, key in XFILES if res.get(key)]
    out = ["| t | " + " | ".join(lab for lab, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for t in range(2, 15):
        cells = []
        for _, E in cols:
            ks, ds, dc = [], [], []
            for per in E.values():
                row = per.get(str(t)) or {}
                k = (row.get("keep") or {}).get("exact") or {}
                d = (row.get("drop") or {}).get("exact") or {}
                if k.get("slope_engine_on_truth") is not None:
                    ks.append(k["slope_engine_on_truth"])
                if d.get("slope_engine_on_truth") is not None:
                    ds.append(d["slope_engine_on_truth"])
                if d.get("corr_all") is not None:
                    dc.append(d["corr_all"])
            cells.append(f"{_mean(ks)} / {_mean(ds)} ({_mean(dc)})")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def xdiag_table(res):
    """Per layer, each file's band-off diagonal t3 against the truth: slope inside / outside the column band (means over
    minis 0-3)."""
    cols = [(lab, res[key]) for lab, key in XFILES if res.get(key)]
    out = ["| t | " + " | ".join(lab for lab, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for t in range(2, 15):
        cells = []
        for _, E in cols:
            si, so = [], []
            for per in E.values():
                dg = ((per.get(str(t)) or {}).get("diag") or {}).get("nott_exact") or {}
                if (dg.get("in_band") or {}).get("slope_engine_on_truth") is not None:
                    si.append(dg["in_band"]["slope_engine_on_truth"])
                if (dg.get("out_band") or {}).get("slope_engine_on_truth") is not None:
                    so.append(dg["out_band"]["slope_engine_on_truth"])
            cells.append(f"{_mean(si)} / {_mean(so)}")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


VAR_ROWS = [  # (label, band-off rows, oracle rows (file, tag), band on / off ablate files)
    ("opr30", [("opr30_rec.jsonl", "rec_mech_nott")], "opr30_swap.jsonl",
     (["ext_opr30_v_engine.jsonl"], ["ext_opr30_v_engine_nott.jsonl"])),
    ("r37 = opr30 + rank 256 + L4", [("r37_rec.jsonl", "rec_mech_nott")], "r37_rec.jsonl",
     (["r37_v_engine.jsonl"], ["r37_v_engine_nott.jsonl"])),
    ("r38 without the extraction (law off)", [("r38x_rec_noext.jsonl", "rec_mech_nott")], "r38x_rec_noext.jsonl",
     (["r38x_v_engine_noext.jsonl"], ["r38x_v_engine_nott_noext.jsonl"])),
    ("r38 with the law back (extraction on)", [("r38x_rec_lawon.jsonl", "rec_mech_nott")], "r38x_rec_lawon.jsonl",
     (["r38x_v_engine_lawon.jsonl"], ["r38x_v_engine_nott_lawon.jsonl"])),
    ("r38 as submitted", [("ref_v_engine_nott.jsonl", None), ("rec23.jsonl", "rec_mech_nott")], ("swap.jsonl", "swap23.jsonl"),
     (["ref_v_engine.jsonl", "rec23.jsonl:rec_mech_prod"], ["ref_v_engine_nott.jsonl", "rec23.jsonl:rec_mech_nott"])),
]


def var_table(runs):
    """Minis 0-3, each file's own band-off engine B: the submitted band (on / off), the clean band (z / B) and the dropped
    entries set to the truth (m / B)."""
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]

    def get(spec):
        f, _, tag = spec.partition(":")
        return by.get((f, tag or None), {})
    out = ["| file | band on / off (minis 0, 1, 2, 3) | z / B | m / B |", "|---|---|---|---|"]
    for lab, brows, ofiles, (on_f, off_f) in VAR_ROWS:
        B = {}
        for f, tag in brows:
            B.update(by.get((f, tag), {}))
        ofiles = ofiles if isinstance(ofiles, tuple) else (ofiles,)
        on, off = {}, {}
        for f in on_f:
            on.update(get(f))
        for f in off_f:
            off.update(get(f))
        cell = lambda num, den: ", ".join("%.3f" % (num[m] / den[m]) if (m in num and m in den) else "-" for m in range(4))  # noqa: E731
        z, mm = {}, {}
        for f in ofiles:
            z.update(by.get((f, "z"), {}))
            mm.update(by.get((f, "m"), {}))
        out.append(f"| {lab} | {cell(on, off)} | {cell(z, B)} | {cell(mm, B)} |")
    return "\n".join(out)


ZSPLIT_ROWS = [  # (label, band-off rows, files holding z / zrow / zcol)
    ("opr30", [("opr30_rec.jsonl", "rec_mech_nott")], ["opr30_swap.jsonl"]),
    ("r38 without the extraction's dA part", [("r38x_rec_noxa.jsonl", "rec_mech_nott")], ["r38x_rec_noxa.jsonl"]),
    ("r38 as submitted", [("ref_v_engine_nott.jsonl", None), ("rec23.jsonl", "rec_mech_nott")],
     ["swap.jsonl", "swap23.jsonl", "r38_zsplit.jsonl"]),
]


def zsplit_table(runs):
    """Minis 0-3: the clean band (z), only the entries dropped for their row (zrow), only those dropped for their column
    (zcol), each over the file's own band-off engine."""
    by = {}
    for r in runs:
        by.setdefault((r["file"], r.get("tag")), {})[r["mlp_id"]] = r["final_mse"]
    out = ["| file | z / B (minis 0, 1, 2, 3) | zrow / B | zcol / B |", "|---|---|---|---|"]
    for lab, brows, files in ZSPLIT_ROWS:
        B = {}
        for f, tag in brows:
            B.update(by.get((f, tag), {}))
        cols = []
        for tag in ("z", "zrow", "zcol"):
            d = {}
            for f in files:
                d.update(by.get((f, tag), {}))
            cols.append(", ".join("%.3f" % (d[m] / B[m]) if (m in d and m in B) else "-" for m in range(4)))
        out.append(f"| {lab} | " + " | ".join(cols) + " |")
    return "\n".join(out)


def main():
    t = (HERE / "README.template.md").read_text(encoding="utf-8")
    comp = (HERE / "tables_compact.md").read_text(encoding="utf-8")
    loc = (HERE / "local.md").read_text(encoding="utf-8")
    res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    ovp = HERE / "overlap.md"
    ov = ovp.read_text(encoding="utf-8") if ovp.exists() else ""
    ovt = lambda h: table_rows(ov, h) if h in ov else "(pending: mech_overlap.py)"  # noqa: E731
    spp = HERE / "sphere.md"
    sph = spp.read_text(encoding="utf-8") if spp.exists() else ""
    k_ = sph.find("truth: inputs on the sphere")
    sph_table = table_rows(sph[k_:], "| t |") if k_ >= 0 else "(pending: sphere_cmp.py)"
    rep = {
        "TABLE_ENTRIES": table_rows(comp, "| mini | t | Pa / Pb"),
        "TABLE_DIAG": table_rows(comp, "| mini | t | band off: in"),
        "TABLE_SWAP": swap_table(res["runs"]),
        "TABLE_LAYERS": "One-layer swaps, final-layer MSE / clean band:\n\n" + table_rows(loc, "| L | dropped (mini 0 / 1)"),
        "TABLE_LOCAL": "Truth on every off-diagonal entry of layer L, local effect (clean band -> truth):\n\n" + table_rows(loc, "| mini | L | C'_L rel err"),
        "TABLE_SWAP23": swap23_table(res["runs"]),
        "TABLE_OPR30_ENTRIES": table_rows(comp, "| mini (opr30) | t | Pa / Pb"),
        "TABLE_OPR30_LAYERS": layer_table(res.get("entries_opr30") or {}),
        "TABLE_R38_LAYERS": layer_table(res.get("entries") or {}),
        "TABLE_OPR30_DIAG": table_rows(comp, "| mini (opr30) | t | band off: in"),
        "TABLE_OPR30_SWAP": opr30_table(res["runs"]),
        "TABLE_GAIN": gain_table(res["runs"]),
        "TABLE_XPART": gain_table(res["runs"], XPART_COLS, only=range(8)),
        "TABLE_XFILE_DIAG": xdiag_table(res),
        "TABLE_XFILE": xfile_table(res),
        "TABLE_VAR": var_table(res["runs"]),
        "TABLE_SPHERE": sph_table,
        "TABLE_ZSPLIT": zsplit_table(res["runs"]),
        "TABLE_OVERLAP_FILES": ovt("| file | on / off (minis"),
        "TABLE_OVERLAP_PAIRS": ovt("| band change of |"),
        "TABLE_OVERLAP_TRANSFER": ovt("| band change of (row), applied to (column)"),
        "TABLE_OVERLAP": ovt("| mini | split |"),
        "TABLE_EXT": ext_table(res["runs"]),
    }
    for k, v in sorted(rep.items(), key=lambda kv: -len(kv[0])):  # longer keys first (TABLE_SWAP23 before TABLE_SWAP)
        assert k in t, k
        t = t.replace(k, v)
    for k, v in ratios(res["runs"]).items():
        t = t.replace(k, v)
    assert "TABLE_" not in t
    (HERE / "README.md").write_text(t, encoding="utf-8")
    print("wrote", HERE / "README.md")


if __name__ == "__main__":
    main()
