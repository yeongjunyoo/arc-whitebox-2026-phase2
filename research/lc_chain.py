"""Learned correction inside the propagation (lc chain): correct the mean of the last K layers with a per-neuron model
of each layer's local error, and carry each correction forward through the estimator's own readout.

Mean-only pathway. The shipped estimator reads the previous layer's mean only through a_L = W_L^T mu_{L-1} / sigma_L.
The corrected chain keeps every base quantity (a_L, the VEC rows, C', the K3 sources, dA, so sigma_L, t3_L, d4_L) and
carries a second mean alongside: p'_L = readout_L(m_L + W_L^T (p'_{L-1} - p_{L-1})) + delta_L, where readout is the
estimator's own pK1 = w01 + w31 t3 / 6 + w41 d4 / 24 and delta_L the model's correction. Offline this is exact:
p'_L = p_L + readout(m_L + dm) - readout(m_L) + delta_L. The final layer returns p'_15.

Local error of layer L (the models' target): l_L = truth_L - readout_L(m_L + W_L^T (truth_{L-1} - p_{L-1})), the
incoming mean replaced by the truth (teacher forcing). Features: the layer's own per-neuron quantities (base, at the
teacher-forced mean for fitting and at the chain's mean when applied) and, at layers 13-15, covariance statistics
of the preactivation (rich: lc_fit.cov_feats, independent of the mean). Pooled models carry the depth L / 15.

Split as lc_fit.py: fit 0-39, choose on 40-49, report 50-59 exactly once. Mini 60-99 are never read.

    .venv/bin/python research/lc_chain.py cache --mlps 0-49   (weights from the dataset; writes research/lc_cache/)
    .venv/bin/python research/lc_chain.py r2                   (local-error predictability per layer, 0-39 / 40-49)
    .venv/bin/python research/lc_chain.py chain                (chain candidates, fitted on 0-39, scored on 40-49)
    .venv/bin/python research/lc_chain.py pick                 (the rule over lc_fit's and the chain's candidates)
    .venv/bin/python research/lc_chain.py final                (the pick on 50-59, once; caches 50-59 first)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_fit as lf  # noqa: E402

CACHE = HERE / "lc_cache"
MODELS = HERE / "lc_chain_models"
LAYERS = list(range(1, 16))
BASE = ["a", "s", "Phi", "phi", "k3", "k4", "kA", "R3n", "R4n", "E33n", "E34n", "E44n", "W5n", "p"]
RICH = ["soff", "soffsum", "dfrac", "dk3", "ok3", "dk4", "coff", "s0off", "t3row", "t3col"]
RICH_LAYERS = (13, 14, 15)
TOL = 0.003  # the selection rule: lowest 40-49 ratio of means, simplest candidate within TOL (as lc_fit select)


def ids(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def feats_raw(m_new, s, t3, d4, kA, p_cur):
    """BASE features (rows, n) at a given mean; identical to lc_fit.base_feats."""
    a = m_new / s
    w, ph, Ph = lf.wick(a, s)
    F = {"a": a, "s": s, "Phi": Ph, "phi": ph, "k3": t3 / s ** 3, "k4": d4 / s ** 4, "kA": kA,
         "R3n": w[3] * t3 / 6.0 / s, "R4n": w[4] * d4 / 24.0 / s, "E33n": w[6] * t3 ** 2 / 72.0 / s,
         "E34n": w[7] * t3 * d4 / 144.0 / s, "E44n": w[8] * d4 ** 2 / 1152.0 / s, "W5n": w[5] * s ** 4,
         "p": p_cur / s}
    return np.stack([F[k] for k in BASE], 0)


def cmd_cache(args):
    from k3_measure import load_row
    CACHE.mkdir(exist_ok=True)
    for m in ids(args.mlps):
        if m >= lf.HOLDOUT_FIRST:
            sys.exit("refused: 60-99")
        D = lf.load(Path(args.dump), m)
        W, _, _, _ = load_row(m)
        out = {k: [] for k in ("m", "s", "t3", "d4", "kA", "pred", "gt", "loc", "F", "dm_tf")}
        out["R"] = []
        for L in LAYERS:
            st = lf.state(D, L)
            st["W"] = W[L].astype(np.float64)
            dm = st["W"].T @ (np.asarray(D["gt"][L - 1], dtype=np.float64) - np.asarray(D["pred"][L - 1], dtype=np.float64))
            p_tf = st["pred"] + lf.readout(st["m"] + dm, st["s"], st["t3"], st["d4"]) - lf.readout(
                st["m"], st["s"], st["t3"], st["d4"])
            kA = st["dA"] / st["s"] ** 4
            out["loc"].append(st["gt"] - p_tf)
            out["F"].append(feats_raw(st["m"] + dm, st["s"], st["t3"], st["d4"], kA, p_tf))
            out["dm_tf"].append(dm)
            out["kA"].append(kA)
            for k in ("m", "s", "t3", "d4", "pred", "gt"):
                out[k].append(st[k])
            if L in RICH_LAYERS:
                cf = lf.cov_feats(D, st)
                out["R"].append(np.stack([cf[k] for k in RICH], 0))
        np.savez(CACHE / f"mlp_{m:03d}.npz", **{k: np.array(v) for k, v in out.items()})
        print(m, "cached", time.strftime("%H:%M:%S"), flush=True)
        del W, D


def load_cache(ms):
    return [dict(np.load(CACHE / f"mlp_{m:03d}.npz")) for m in ms]


def cmd_r2(args):
    tr, va = load_cache(lf.FIT), load_cache(lf.SEL)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "features": BASE, "per_layer": {}}
    for L in LAYERS:
        for tag, rich in (("base", False), ("rich", True)):
            if rich and L not in RICH_LAYERS:
                continue
            seg = {"layers": [L], "rich": rich, "depth": False}
            Xt, yt = seg_xy(tr, seg)
            Xv, yv = seg_xy(va, seg)
            row = {"mse_local_fit": float((yt ** 2).mean())}
            for lam in (1e-5, 1e-2):
                M = lf.ridge_fit(Xt, yt, lam)
                row[f"r2_in_lam{lam:g}"] = float(1 - ((yt - lf.ridge_pred(M, Xt)) ** 2).sum() / (yt ** 2).sum())
                row[f"r2_sel_lam{lam:g}"] = float(1 - ((yv - lf.ridge_pred(M, Xv)) ** 2).sum() / (yv ** 2).sum())
            out["per_layer"][f"{L}_{tag}"] = row
            print(L, tag, json.dumps({k: round(v, 5) if "r2" in k else v for k, v in row.items()}), flush=True)
    p = HERE / f"lc_chain_r2_{time.strftime('%m%d_%H%M')}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", p.name)


# ------------------------------------------------------------------------------------------------ segments
def seg_rows(c, L, seg, base_rows=None):
    """(n, f) design of one layer of one MLP: base rows (cached at the teacher-forced mean unless given), rich rows,
    depth."""
    parts = [c["F"][L - 1] if base_rows is None else base_rows]
    if seg["rich"]:
        parts.append(c["R"][L - RICH_LAYERS[0]][[RICH.index(k) for k in seg.get("rich_names", RICH)]])
    if seg["depth"]:
        parts.append(np.full((1, lf.N), L / 15.0))
    return np.concatenate(parts, 0).T


def seg_xy(C, seg):
    X = np.concatenate([seg_rows(c, L, seg) for c in C for L in seg["layers"]], 0)
    y = np.concatenate([c["loc"][L - 1] for c in C for L in seg["layers"]], 0)
    return X, y


def fit_segment(tr, va, seg):
    X, y = seg_xy(tr, seg)
    if seg["model"] == "ridge":
        return [lf.ridge_fit(X, y, seg["lam"])]
    Xv, yv = seg_xy(va, seg)
    return [lf.mlp_fit(X, y, Xv, yv, seg["hidden"], seg["wd"], seed=sd, batch=2048, epochs=150, patience=20)
            for sd in seg.get("seeds", [0])]


def seg_pred(models, X):
    return np.mean([lf.predict(M, X) for M in models], 0)


def run_chain(c, W64, spec, fitted, perm=None, only=None):
    """Final-layer prediction of the corrected chain (every correction carried forward exactly).
    perm: (segment, column, rng) shuffles one input column across neurons at every layer of that segment;
    only: the set of layers that apply their correction (the others pass the carried mean through)."""
    where = {L: i for i, seg in enumerate(spec["segments"]) for L in seg["layers"] if only is None or L in only}
    dp = np.zeros(lf.N)  # p' - p of the previous layer
    for L in LAYERS:
        i = L - 1
        m, s, t3, d4, p = c["m"][i], c["s"][i], c["t3"][i], c["d4"][i], c["pred"][i]
        if dp.any():
            dm = W64[L].T @ dp
            p_new = p + lf.readout(m + dm, s, t3, d4) - lf.readout(m, s, t3, d4)
        else:
            dm, p_new = 0.0, p
        if L in where:
            seg = spec["segments"][where[L]]
            X = seg_rows(c, L, seg, base_rows=feats_raw(m + dm, s, t3, d4, c["kA"][i], p_new))
            if perm is not None and perm[0] == where[L]:
                X[:, perm[1]] = perm[2].permutation(X[:, perm[1]])
            p_new = p_new + seg_pred(fitted[where[L]], X)
        dp = p_new - p
    return c["pred"][14] + dp


def complexity(spec):
    segs = spec["segments"]
    mod = max({"ridge": 0, "mlp": 1}[s["model"]] + (1 if len(s.get("seeds", [0])) > 1 else 0) for s in segs)
    return 100 + 10 * mod + (1 if any(s["rich"] for s in segs) else 0) + max(len(s["layers"]) for s in segs) / 100.0


def chain_specs():
    out = []
    for K in (8, 15):
        out.append({"name": f"ridge_base_K{K}", "segments": [
            {"layers": LAYERS[-K:], "model": "ridge", "lam": 1e-4, "rich": False, "depth": True}]})
    for hidden, wd in ((16, 1e-4), (32, 1e-4), (16, 1e-3)):
        out.append({"name": f"mlp_base_K15_h{hidden}_wd{wd:g}", "segments": [
            {"layers": LAYERS, "model": "mlp", "hidden": hidden, "wd": wd, "rich": False, "depth": True}]})
    out.append({"name": "mlp_base_K15_h16_wd0.0001_ens3", "segments": [
        {"layers": LAYERS, "model": "mlp", "hidden": 16, "wd": 1e-4, "seeds": [0, 1, 2], "rich": False, "depth": True}]})
    out.append({"name": "mlp_base_K8_h16_wd0.0001", "segments": [
        {"layers": LAYERS[-8:], "model": "mlp", "hidden": 16, "wd": 1e-4, "rich": False, "depth": True}]})
    # split: base model on layers 1-12, rich model on 13-15 (rich without soff/soffsum avoids a dense product at 15)
    for tail in ("ridge", "mlp"):
        for rn in (RICH, [k for k in RICH if k not in ("soff", "soffsum")]):
            tag = "rich" if len(rn) == len(RICH) else "richnos"
            t = {"layers": list(RICH_LAYERS), "model": tail, "rich": True, "rich_names": rn, "depth": True}
            t.update({"lam": 1e-4} if tail == "ridge" else {"hidden": 16, "wd": 1e-4})
            out.append({"name": f"split_mlp_base_1_12__{tail}_{tag}_13_15", "segments": [
                {"layers": LAYERS[:12], "model": "mlp", "hidden": 16, "wd": 1e-4, "rich": False, "depth": True}, t]})
    t = {"layers": list(RICH_LAYERS), "model": "mlp", "hidden": 16, "wd": 1e-4, "seeds": [0, 1, 2], "rich": True,
         "rich_names": [k for k in RICH if k not in ("soff", "soffsum")], "depth": True}
    out.append({"name": "split_mlp_base_1_12__mlp_richnos_13_15_ens3", "segments": [
        {"layers": LAYERS[:12], "model": "mlp", "hidden": 16, "wd": 1e-4, "seeds": [0, 1, 2], "rich": False,
         "depth": True}, t]})
    # added after the first chain scores (40-49): the rich tail with ensembles and a wider tail
    t = {"layers": list(RICH_LAYERS), "model": "mlp", "hidden": 16, "wd": 1e-4, "seeds": [0, 1, 2], "rich": True,
         "rich_names": RICH, "depth": True}
    out.append({"name": "split_mlp_base_1_12__mlp_rich_13_15_ens3", "segments": [
        {"layers": LAYERS[:12], "model": "mlp", "hidden": 16, "wd": 1e-4, "seeds": [0, 1, 2], "rich": False,
         "depth": True}, t]})
    t = {"layers": list(RICH_LAYERS), "model": "mlp", "hidden": 32, "wd": 1e-4, "rich": True, "rich_names": RICH,
         "depth": True}
    out.append({"name": "split_mlp_base_1_12__mlp_rich_h32_13_15", "segments": [
        {"layers": LAYERS[:12], "model": "mlp", "hidden": 16, "wd": 1e-4, "rich": False, "depth": True}, t]})
    return out


def save_fitted(path, spec, fitted):
    arrs = {}
    for i, models in enumerate(fitted):
        for j, M in enumerate(models):
            pre = f"s{i}_m{j}_"
            for k, v in M.items():
                if k == "P":
                    for kk, vv in v.items():
                        arrs[pre + "P_" + kk] = vv
                elif k not in ("depth",):
                    arrs[pre + k] = np.asarray(v)
    np.savez(path, **arrs)
    (path.with_suffix(".json")).write_text(json.dumps(spec, indent=1), encoding="utf-8")


def load_fitted(path):
    spec = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(path)
    fitted = []
    for i, seg in enumerate(spec["segments"]):
        models = []
        for j in range(len(seg.get("seeds", [0])) if seg["model"] == "mlp" else 1):
            pre = f"s{i}_m{j}_"
            if str(z[pre + "kind"]) == "ridge":
                models.append({"kind": "ridge", "mu": z[pre + "mu"], "sd": z[pre + "sd"], "beta": z[pre + "beta"],
                               "b0": float(z[pre + "b0"])})
            else:
                models.append({"kind": "mlp", "mu": z[pre + "mu"], "sd": z[pre + "sd"], "ys": float(z[pre + "ys"]),
                               "P": {k: z[pre + "P_" + k] for k in ("W1", "b1", "w2", "b2")}})
        fitted.append(models)
    return spec, fitted


def score(C, ms, spec, fitted, load_row):
    base, corr = np.zeros(len(ms)), np.zeros(len(ms))
    for j, m in enumerate(ms):
        W, _, _, _ = load_row(m)
        W64 = W.astype(np.float64)
        del W
        c = C[j]
        base[j] = float(((c["gt"][14] - c["pred"][14]) ** 2).mean())
        corr[j] = float(((c["gt"][14] - run_chain(c, W64, spec, fitted)) ** 2).mean())
        del W64
    return base, corr


def cmd_chain(args):
    from k3_measure import load_row
    if (HERE / "lc_final.json").exists():
        sys.exit("refused: the 50-59 report exists; the choice is frozen")
    t0 = time.time()
    tr, va = load_cache(lf.FIT), load_cache(lf.SEL)
    MODELS.mkdir(exist_ok=True)
    specs = chain_specs()
    fitted_all = []
    for spec in specs:
        fitted = [fit_segment(tr, va, seg) for seg in spec["segments"]]
        fitted_all.append(fitted)
        save_fitted(MODELS / f"{spec['name']}.npz", spec, fitted)
        print("fitted", spec["name"], round(time.time() - t0, 1), flush=True)
    res = []
    base = np.zeros(len(lf.SEL))
    corr = np.zeros((len(specs), len(lf.SEL)))
    for j, m in enumerate(lf.SEL):
        W, _, _, _ = load_row(m)
        W64 = W.astype(np.float64)
        del W
        c = va[j]
        base[j] = float(((c["gt"][14] - c["pred"][14]) ** 2).mean())
        for q, spec in enumerate(specs):
            corr[q, j] = float(((c["gt"][14] - run_chain(c, W64, spec, fitted_all[q])) ** 2).mean())
        del W64
    for q, spec in enumerate(specs):
        st = lf.ratio_stats(base, corr[q])
        res.append({"name": spec["name"], "complexity": complexity(spec), "spec": spec, "val": st})
        print(spec["name"], round(st["ratio_of_means"], 5), round(st["mean_ratio"], 5), st["better_on"], flush=True)
    p = HERE / "lc_chain_sel.json"
    p.write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "base_mse": list(base), "results": res},
                            indent=1, default=float), encoding="utf-8")
    print("saved", p.name, round(time.time() - t0, 1))


def cmd_importance(args):
    """Which inputs matter for the pick, on the selection set 40-49 only: the rise of the final MSE ratio when one input
    column is shuffled across neurons (every layer of its segment, 3 shuffles), and the ratio when only some layers
    apply their correction."""
    from k3_measure import load_row
    pk = json.loads((HERE / "lc_pick.json").read_text(encoding="utf-8"))["pick"]
    spec, fitted = load_fitted(MODELS / f"{pk['name']}.npz")
    C = load_cache(lf.SEL)
    Ws = []
    for m in lf.SEL:
        W, _, _, _ = load_row(m)
        Ws.append(W.astype(np.float64))
        del W
    base = np.array([float(((c["gt"][14] - c["pred"][14]) ** 2).mean()) for c in C])

    def ratio(**kw):
        corr = np.array([float(((c["gt"][14] - run_chain(c, W, spec, fitted, **kw)) ** 2).mean()) for c, W in zip(C, Ws)])
        return float(corr.sum() / base.sum())

    r0 = ratio()
    out = {"pick": pk["name"], "ratio_40_49": r0, "shuffle": {}, "layers_only": {}}
    rng = np.random.default_rng(11)
    for si, seg in enumerate(spec["segments"]):
        cols = BASE + (seg.get("rich_names", RICH) if seg["rich"] else []) + ["depth"]
        for j, name in enumerate(cols[:-1]):
            vals = [ratio(perm=(si, j, rng)) for _ in range(3)]
            out["shuffle"][f"seg{si}:{name}"] = round(float(np.mean(vals)) - r0, 5)
            print(si, name, out["shuffle"][f"seg{si}:{name}"], flush=True)
    for tag, only in (("1-12", set(range(1, 13))), ("13-15", {13, 14, 15}), ("15", {15}), ("8-15", set(range(8, 16)))):
        out["layers_only"][tag] = ratio(only=only)
        print(tag, out["layers_only"][tag], flush=True)
    out["shuffle"] = dict(sorted(out["shuffle"].items(), key=lambda kv: -kv[1]))
    (HERE / "lc_importance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def cmd_pick(args):
    """The pre-declared rule over every candidate scored on 40-49: lc_fit's final-layer candidates and the chains."""
    sel = json.loads((HERE / "lc_select.json").read_text(encoding="utf-8"))
    ch = json.loads((HERE / "lc_chain_sel.json").read_text(encoding="utf-8"))
    pool = [{"source": "lc_fit", "name": f"{c['cls']} {c['hyper']}", "complexity": lf.RANK[c["cls"]],
             "ratio": c["val"]["ratio_of_means"], "is_lc_fit_pick": c == sel["pick"]} for c in sel["candidates"]]
    pool += [{"source": "lc_chain", "name": r["name"], "complexity": r["complexity"], "ratio": r["val"]["ratio_of_means"]}
             for r in ch["results"]]
    best = min(pool, key=lambda r: r["ratio"])
    pick = min((r for r in pool if r["ratio"] <= best["ratio"] + TOL), key=lambda r: (r["complexity"], r["ratio"]))
    if pick["source"] == "lc_fit" and not pick.get("is_lc_fit_pick"):
        sys.exit("the rule picked an lc_fit candidate other than lc_fit's own pick: rerun lc_fit select")
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": f"lowest 40-49 ratio of means; simplest within {TOL}",
           "best": best, "pick": pick, "n_candidates": len(pool)}
    (HERE / "lc_pick.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def cmd_final(args):
    """The pick on 50-59, exactly once: per-MLP final MSE ratio corrected / base with a paired bootstrap."""
    from k3_measure import load_row
    lock = HERE / "lc_final.json"
    if lock.exists():
        sys.exit("refused: the 50-59 report exists (it is made exactly once)")
    pk = json.loads((HERE / "lc_pick.json").read_text(encoding="utf-8"))["pick"]
    if pk["source"] == "lc_fit":
        return lf.cmd_final(args)
    lock.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pick": pk["name"]}), encoding="utf-8")
    missing = [m for m in lf.REP if not (CACHE / f"mlp_{m:03d}.npz").exists()]
    if missing:
        cmd_cache(argparse.Namespace(dump=args.dump, mlps=",".join(str(m) for m in missing)))
    spec, fitted = load_fitted(MODELS / f"{pk['name']}.npz")
    C = load_cache(lf.REP)
    base, corr = score(C, lf.REP, spec, fitted, load_row)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mlps": lf.REP, "pick": pk, "spec": spec,
           "base_final_mse": [float(x) for x in base], "corr_final_mse": [float(x) for x in corr],
           "mean_base": float(base.mean()), "mean_corr": float(corr.mean())}
    out.update(lf.ratio_stats(base, corr))
    out.update(lf.boot_ci(base, corr))
    lock.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "r2", "chain", "pick", "final", "importance"])
    ap.add_argument("--dump", default=str(HERE / "lc_dump_2lb"))
    ap.add_argument("--mlps", default="0-49")
    args = ap.parse_args()
    {"cache": cmd_cache, "r2": cmd_r2, "chain": cmd_chain, "pick": cmd_pick, "final": cmd_final,
     "importance": cmd_importance}[args.cmd](args)


if __name__ == "__main__":
    main()
