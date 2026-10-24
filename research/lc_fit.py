"""Learned correction (lc) of the final-layer means of estimators/k3_fast_aug_sc_2l_b.py, fitted offline on the dump
written by research/lc_dump.py (the shipped estimator's own per-neuron quantities, float32, next to the dataset truth).

Split, fixed before any fit: fit on public mini 0-39, choose the model class and hyperparameters on 40-49, report on
50-59 exactly once (`final` reads the choice written by `select` and refuses to run a second time). Mini 60-99, the
final-designation hold-out, are never read (the dump has none of them, and `load` refuses them).

    .venv/bin/python research/lc_fit.py diag   --dump research/lc_dump_2lb      (fit set only: 0-39, plus 40-49 MSEs)
    .venv/bin/python research/lc_fit.py select --dump research/lc_dump_2lb      (fit 0-39, score 40-49, write the choice)
    .venv/bin/python research/lc_fit.py final  --dump research/lc_dump_2lb      (the chosen model on 50-59, once)

Target: the final-layer residual y = truth - estimate. The readout of a layer is the estimator's own
pK1 = w01 + w31 t3 / 6 + w41 d4 / 24 with d4 = g dA + g lam' sigma^2 (g = 2); `check` asserts it reproduces the dumped
prediction. A mean correction at layer 14 reaches layer 15 only through m15 = W15^T mu14 (the variance, t3 and dA of
layer 15 do not read mu14 in the shipped code), so the two-layer chain is exact offline:
p15' = p15 + readout(m15 + W15^T delta14) - readout(m15).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIT, SEL, REP = list(range(0, 40)), list(range(40, 50)), list(range(50, 60))
HOLDOUT_FIRST = 60
G = 2.0
N = 1024
V_W11, V_W11T, V_W11D, V_PK1, V_PK2, V_PK3, V_PK4 = 2, 8, 11, 17, 18, 19, 20
SQ2PI = math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf, otypes=[float])


def Phi(a):
    return 0.5 * (1.0 + _erf(a / math.sqrt(2.0)))


def wick(a, s):
    """w_k1 = E[d^k ReLU(Z)], Z ~ N(m, s^2), a = m / s, k = 0..8 (k >= 2: (-1)^k He_{k-2}(a) phi(a) / s^(k-1))."""
    ph = np.exp(-0.5 * a * a) / SQ2PI
    Ph = Phi(a)
    He = [np.ones_like(a), a, a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3, a ** 5 - 10 * a ** 3 + 15 * a,
          a ** 6 - 15 * a ** 4 + 45 * a * a - 15]
    w = {0: s * (ph + a * Ph), 1: Ph}
    for k in range(2, 9):
        w[k] = (-1) ** k * He[k - 2] * ph / s ** (k - 1)
    return w, ph, Ph


def readout(m, s, t3, d4):
    w, _, _ = wick(m / s, s)
    return w[0] + w[3] * t3 / 6.0 + w[4] * d4 / 24.0


def load(dump: Path, m: int) -> dict:
    if m >= HOLDOUT_FIRST:
        raise SystemExit(f"refused: mini {m} is in the final-designation hold-out (60-99)")
    with np.load(dump / f"mlp_{m:03d}.npz") as z:
        return {k: z[k] for k in z.files}


def state(D: dict, L: int) -> dict:
    """The estimator's own quantities of layer L (14 or 15), float64."""
    f8 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    VX = D["VX"]
    if L == 15:
        var, a, dA, lam = f8(D["L15_var"]), f8(D["L15_a"]), f8(D["L15_dA"]), float(D["L15_FF"][4][0])
        t3b3, t3f = f8(D["L15_t3b"]) / 3.0, f8(D["L15_t3f"])
        t3 = t3b3 + t3f
    else:
        var, a = f8(D[f"L{L}_var"]), f8(D[f"L{L}_a"])
        dA, lam = f8(D[f"L{L}_FF"][3]), float(D[f"L{L}_FF"][4][0])
        w11 = f8(VX[L, V_W11])
        t3 = np.where(w11 > 1e-30, f8(VX[L, V_W11T]) / np.maximum(w11, 1e-30), 0.0)
        t3b3, t3f = None, None
    s = np.sqrt(var)
    W = D.get(f"W{L}")
    return {"L": L, "var": var, "s": s, "a": a, "m": a * s, "t3": t3, "t3b3": t3b3, "t3f": t3f, "dA": dA, "lam": lam,
            "d4": G * dA + G * lam * var, "pred": f8(D["pred"][L]), "gt": f8(D["gt"][L]),
            "W": None if W is None else f8(W)}


def check(D: dict, L: int) -> dict:
    st = state(D, L)
    r = readout(st["m"], st["s"], st["t3"], st["d4"])
    out = {"readout_maxabs": float(np.abs(r - st["pred"]).max()), "pred_absmax": float(np.abs(st["pred"]).max())}
    if st["W"] is not None:
        mW = st["W"].T @ np.asarray(D["pred"][L - 1], dtype=np.float64)
        out.update({"m_maxabs": float(np.abs(mW - st["m"]).max()), "m_absmax": float(np.abs(st["m"]).max())})
    return out


def prev_cumulants(D: dict, L: int):
    """Activation cumulants K2..K4 of layer L - 1 from its VEC power rows (pK1..pK4)."""
    p1, p2, p3, p4 = (np.asarray(D["VX"][L - 1, r], dtype=np.float64) for r in (V_PK1, V_PK2, V_PK3, V_PK4))
    K2 = p2 - p1 * p1
    K3 = p3 - 3 * p2 * p1 + 2 * p1 ** 3
    K4 = p4 - 4 * p3 * p1 - 3 * p2 * p2 + 12 * p2 * p1 * p1 - 6 * p1 ** 4
    return p1, K2, K3, K4


def base_feats(st: dict) -> dict:
    """Per-neuron features of one layer from its own quantities (m may be a corrected mean)."""
    a, s, var, t3, dA, d4 = st["m"] / st["s"], st["s"], st["var"], st["t3"], st["dA"], st["d4"]
    w, ph, Ph = wick(a, s)
    F = {"a": a, "s": s, "Phi": Ph, "phi": ph, "k3": t3 / s ** 3, "k4": d4 / s ** 4, "kA": dA / s ** 4,
         "R3n": w[3] * t3 / 6.0 / s, "R4n": w[4] * d4 / 24.0 / s, "E33n": w[6] * t3 ** 2 / 72.0 / s,
         "E34n": w[7] * t3 * d4 / 144.0 / s, "E44n": w[8] * d4 ** 2 / 1152.0 / s, "W5n": w[5] * s ** 4,
         "p": st["pred_cur"] / s}
    if st["t3f"] is not None:
        F["t3f_n"] = st["t3f"] / s ** 3
    return F


def cov_feats(D: dict, st: dict) -> dict:
    """Covariance and previous-layer statistics of layer L's preactivation (the dump's layer L - 1 statistics)."""
    L, var, s, W = st["L"], st["var"], st["s"], st["W"]
    P = f"L{L - 1}_"
    W2 = W * W
    W3, W4 = W2 * W, W2 * W2
    p1, K2, K3, K4 = prev_cumulants(D, L)
    dk3 = (W3.T @ K3) / s ** 3
    F = {"soff": np.log(np.asarray(D[P + "next_s_off_sq"], dtype=np.float64) / var ** 2),
         "soffsum": np.asarray(D[P + "next_s_off_sum"], dtype=np.float64) / var,
         "dfrac": np.asarray(D[P + "next_s_diagonly"], dtype=np.float64) / var,
         "dk3": dk3, "ok3": st["t3"] / s ** 3 - dk3, "dk4": (W4.T @ K4) / s ** 4,
         "coff": np.log((W2.T @ np.asarray(D[P + "c_off_sq"], dtype=np.float64)) / var ** 2),
         "s0off": np.log((W2.T @ np.asarray(D[P + "s0_off_sq"], dtype=np.float64)) / var ** 2),
         "t3row": np.log((W2.T @ np.asarray(D[P + "t3t_row_sq"], dtype=np.float64)) / var ** 3),
         "t3col": np.log((W2.T @ np.asarray(D[P + "t3t_col_sq"], dtype=np.float64)) / var ** 3)}
    return F


def prop_feats(D: dict, st: dict, prev: dict) -> dict:
    """Previous-layer per-neuron features pushed through W_L and scaled by w11_L (the mean map's Jacobian)."""
    Ph = Phi(st["m"] / st["s"])
    W = st["W"]
    out = {}
    for k in ("a", "k3", "k4", "kA", "R3n", "R4n", "E33n"):
        x = prev[k] - prev[k].mean()
        out["P1_" + k] = Ph * (W.T @ x)
    return out


FEW = ["a", "s", "k3", "k4", "kA", "R3n", "R4n", "Phi", "p"]


def layer_table(D: dict, L: int, dm=None, pred_cur=None, prev_F=None) -> dict:
    """All features of layer L; dm: a correction of the preactivation mean (from a corrected layer L - 1)."""
    st = state(D, L)
    if dm is not None:
        st["m"] = st["m"] + dm
    st["pred_cur"] = st["pred"] if pred_cur is None else pred_cur
    F = base_feats(st)
    F.update(cov_feats(D, st))
    if prev_F is not None:
        F.update(prop_feats(D, st, prev_F))
    return F


def prev_base(D: dict, L: int) -> dict:
    st = state(D, L)
    st["pred_cur"] = st["pred"]
    return base_feats(st)


# ------------------------------------------------------------------------------------------------ models
class Std:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12

    def __call__(self, X):
        return (X - self.mu) / self.sd


def ridge_fit(X, y, lam):
    """min |y - b0 - Z beta|^2 + lam N |beta|^2 on z-scored Z (intercept unpenalized)."""
    sc = Std(X)
    Z = sc(X)
    ym = y.mean()
    A = Z.T @ Z + lam * len(y) * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - ym))
    return {"kind": "ridge", "mu": sc.mu, "sd": sc.sd, "beta": beta, "b0": ym}


def ridge_pred(M, X):
    return M["b0"] + ((X - M["mu"]) / M["sd"]) @ M["beta"]


def mlp_fit(Xtr, ytr, Xva, yva, hidden, wd, seed=0, lr=1e-3, epochs=400, batch=512, patience=40):
    """f -> hidden (ReLU) -> 1, Adam, L2 weight decay, early stopping on the selection set; z-scored inputs, scaled y."""
    sc = Std(Xtr)
    Ztr, Zva = sc(Xtr), sc(Xva)
    ys = ytr.std()
    ttr, tva = ytr / ys, yva / ys
    rng = np.random.default_rng(seed)
    f = Ztr.shape[1]
    P = {"W1": rng.standard_normal((f, hidden)) * np.sqrt(2.0 / f), "b1": np.zeros(hidden),
         "w2": np.zeros(hidden), "b2": np.zeros(1)}
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    vel = {k: np.zeros_like(v) for k, v in P.items()}
    b1_, b2_, eps = 0.9, 0.999, 1e-8

    def fwd(Z, P):
        h = np.maximum(Z @ P["W1"] + P["b1"], 0.0)
        return h, h @ P["w2"] + P["b2"][0]

    best, best_loss, bad, step = None, np.inf, 0, 0
    n = len(ttr)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            Z, t = Ztr[idx], ttr[idx]
            h, o = fwd(Z, P)
            g = 2.0 * (o - t) / len(idx)
            gr = {"w2": h.T @ g + 2 * wd * P["w2"], "b2": np.array([g.sum()])}
            gh = np.outer(g, P["w2"]) * (h > 0)
            gr["W1"] = Z.T @ gh + 2 * wd * P["W1"]
            gr["b1"] = gh.sum(0)
            step += 1
            for k in P:
                mom[k] = b1_ * mom[k] + (1 - b1_) * gr[k]
                vel[k] = b2_ * vel[k] + (1 - b2_) * gr[k] ** 2
                P[k] = P[k] - lr * (mom[k] / (1 - b1_ ** step)) / (np.sqrt(vel[k] / (1 - b2_ ** step)) + eps)
        _, ov = fwd(Zva, P)
        lv = float(((ov - tva) ** 2).mean())
        hist.append(lv)
        if lv < best_loss - 1e-6:
            best_loss, best, bad = lv, {k: v.copy() for k, v in P.items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return {"kind": "mlp", "mu": sc.mu, "sd": sc.sd, "ys": ys, "P": best, "epochs": len(hist),
            "best_epoch": int(np.argmin(hist)) + 1}


def mlp_pred(M, X):
    Z = (X - M["mu"]) / M["sd"]
    P = M["P"]
    h = np.maximum(Z @ P["W1"] + P["b1"], 0.0)
    return (h @ P["w2"] + P["b2"][0]) * M["ys"]


def predict(M, X):
    return ridge_pred(M, X) if M["kind"] == "ridge" else mlp_pred(M, X)


# ------------------------------------------------------------------------------------------------ data
def mat(rows, key, names):
    return np.concatenate([np.stack([r[key][k] for k in names], 1) for r in rows], 0)


def per_mlp_mse(rows, yhat_list, key="y15"):
    return np.array([float(((r[key] - yh) ** 2).mean()) for r, yh in zip(rows, yhat_list)])


def split_pred(M, rows, key, names):
    return [predict(M, np.stack([r[key][k] for k in names], 1)) for r in rows]


def ratio_stats(base, corr):
    r = corr / base
    return {"ratio_of_means": float(corr.sum() / base.sum()), "mean_ratio": float(r.mean()),
            "better_on": int((r < 1).sum()), "n": int(len(r)), "per_mlp": [float(x) for x in r]}


def boot_ci(base, corr, reps=20000, seed=12345):
    rng = np.random.default_rng(seed)
    k = len(base)
    idx = rng.integers(0, k, size=(reps, k))
    rom = corr[idx].sum(1) / base[idx].sum(1)
    mr = (corr[idx] / base[idx]).mean(1)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"ratio_of_means_ci95": q(rom), "mean_ratio_ci95": q(mr), "reps": reps, "seed": seed}


# ------------------------------------------------------------------------------------------------ data (lean)
def build_rows(dump: Path, mlps, chain=True):
    """Per MLP: the layer-15 table (with layer-14 features pushed through W15), the residual, and what the two-layer
    chain needs (the layer-14 table and residual, W15, the layer-15 state)."""
    rows = []
    for m in mlps:
        D = load(dump, m)
        prev14 = prev_base(D, 14)
        st15 = state(D, 15)
        rec = {"m": m, "F15": layer_table(D, 15, prev_F=prev14), "y15": st15["gt"] - st15["pred"],
               "pred15": st15["pred"], "gt15": st15["gt"]}
        if chain:
            st14 = state(D, 14)
            rec["F14"] = layer_table(D, 14, prev_F=prev_base(D, 13))
            rec["y14"] = st14["gt"] - st14["pred"]
            rec["dm15_tf"] = st15["W"].T @ (st14["gt"] - st14["pred"])  # teacher forcing of the mean at layer 14
            rec["Dkeep"] = {k: D[k] for k in D if k.startswith(("L14_", "L15_", "VX", "pred", "gt", "W15"))}
            rec["prev14"] = prev14
        rows.append(rec)
        del D
    return rows


def chain_l15(rec, delta14):
    """Layer-15 table and residual after adding delta14 to layer 14's mean (exact for the shipped readout)."""
    D = rec["Dkeep"]
    st15 = state(D, 15)
    dm = st15["W"].T @ delta14
    p_new = st15["pred"] + readout(st15["m"] + dm, st15["s"], st15["t3"], st15["d4"]) - readout(
        st15["m"], st15["s"], st15["t3"], st15["d4"])
    F = layer_table(D, 15, dm=dm, pred_cur=p_new, prev_F=rec["prev14"])
    return F, st15["gt"] - p_new


def names_of(rows, key, exclude=()):
    return [k for k in rows[0][key] if k not in exclude]


QUAD_BASE = ["a", "k3", "k4", "kA", "soff", "dfrac", "ok3"]


def add_quad(rows, key, src_names):
    out = []
    for i, x in enumerate(QUAD_BASE):
        for y in QUAD_BASE[i:]:
            out.append(f"{x}*{y}")
            for r in rows:
                r[key][f"{x}*{y}"] = r[key][x] * r[key][y]
    return src_names + out


# ------------------------------------------------------------------------------------------------ commands
def cmd_diag(args):
    dump = Path(args.dump)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "dump": str(dump)}
    D0 = load(dump, 0)
    out["check_mlp0"] = {"L15": check(D0, 15), "L14": check(D0, 14), "L13": check(D0, 13)}
    del D0
    meta = json.loads((dump / "meta.json").read_text(encoding="utf-8"))
    lm = {r["mlp_id"]: r["layer_mse"] for r in meta["mlps"]}
    out["layer_mse_fit"] = [float(v) for v in np.mean([lm[m] for m in FIT], 0)]
    out["final_mse_fit"] = float(np.mean([lm[m][15] for m in FIT]))
    out["final_mse_sel"] = float(np.mean([lm[m][15] for m in SEL]))
    rows = build_rows(dump, FIT)
    # teacher forcing: how much of the final error is the mean error inherited from layer 14
    tf = []
    for r in rows:
        st = state(r["Dkeep"], 15)
        p_tf = st["pred"] + readout(st["m"] + r["dm15_tf"], st["s"], st["t3"], st["d4"]) - readout(
            st["m"], st["s"], st["t3"], st["d4"])
        e_loc, e_inh = st["gt"] - p_tf, p_tf - st["pred"]
        tf.append([float((r["y15"] ** 2).mean()), float((e_loc ** 2).mean()), float((e_inh ** 2).mean()),
                   float(np.corrcoef(e_loc, e_inh)[0, 1])])
    tf = np.array(tf)
    out["teacher_forcing_L15_fit"] = {"mse_base": float(tf[:, 0].mean()), "mse_local": float(tf[:, 1].mean()),
                                      "mse_inherited": float(tf[:, 2].mean()),
                                      "corr_local_inherited": float(tf[:, 3].mean())}
    # the same at layer 14 (mean inherited from layer 13), on 0-9: needs W14
    tf14 = []
    for m in FIT[:10]:
        D = load(dump, m)
        st = state(D, 14)
        dm = st["W"].T @ (np.asarray(D["gt"][13], dtype=np.float64) - np.asarray(D["pred"][13], dtype=np.float64))
        p_tf = st["pred"] + readout(st["m"] + dm, st["s"], st["t3"], st["d4"]) - readout(st["m"], st["s"], st["t3"],
                                                                                          st["d4"])
        tf14.append([float(((st["gt"] - st["pred"]) ** 2).mean()), float(((st["gt"] - p_tf) ** 2).mean()),
                     float(((p_tf - st["pred"]) ** 2).mean())])
        del D
    tf14 = np.array(tf14)
    out["teacher_forcing_L14_fit0_9"] = {"mse_base": float(tf14[:, 0].mean()), "mse_local": float(tf14[:, 1].mean()),
                                         "mse_inherited": float(tf14[:, 2].mean())}
    # feature correlations with the final residual (pooled over 0-39, and the mean of per-MLP correlations)
    names = names_of(rows, "F15")
    X, y = mat(rows, "F15", names), np.concatenate([r["y15"] for r in rows])
    cor = {}
    for j, k in enumerate(names):
        c = float(np.corrcoef(X[:, j], y)[0, 1])
        pm = float(np.mean([np.corrcoef(r["F15"][k], r["y15"])[0, 1] for r in rows]))
        cor[k] = [round(c, 4), round(pm, 4)]
    out["corr_y15_fit"] = dict(sorted(cor.items(), key=lambda kv: -abs(kv[1][0])))
    out["common_mode_share_fit"] = float(np.mean([r["y15"].mean() ** 2 / (r["y15"] ** 2).mean() for r in rows]))
    for key, ykey in (("F15", "y15"), ("F14", "y14")):
        nm = names_of(rows, key)
        Xk, yk = mat(rows, key, nm), np.concatenate([r[ykey] for r in rows])
        for lam in (1e-6, 1e-3):
            M = ridge_fit(Xk, yk, lam)
            res = yk - ridge_pred(M, Xk)
            out[f"insample_r2_{key}_lam{lam:g}"] = float(1 - (res ** 2).sum() / (yk ** 2).sum())
    p = HERE / f"lc_diag_{time.strftime('%m%d_%H%M')}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("saved", p.name)


def eval_model(rows, key, names, M, ykey="y15"):
    yh = split_pred(M, rows, key, names)
    base = np.array([float((r[ykey] ** 2).mean()) for r in rows])
    return base, per_mlp_mse(rows, yh, ykey)


LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RANK = {"ridge_few": 0, "ridge_base": 1, "pooled_ridge": 2, "ridge_rich": 3, "ridge_rich_quad": 4, "chain_ridge": 5,
        "mlp": 6}
BASE_NAMES = ("a", "s", "Phi", "phi", "k3", "k4", "kA", "R3n", "R4n", "E33n", "E34n", "E44n", "W5n", "p", "t3f_n")


def cmd_select(args):
    dump = Path(args.dump)
    if (HERE / "lc_final.json").exists():
        sys.exit("refused: the 50-59 report already exists; the choice is frozen")
    t0 = time.time()
    tr, va = build_rows(dump, FIT), build_rows(dump, SEL)
    print(f"built {len(tr)} + {len(va)} MLPs in {time.time() - t0:.1f}s", flush=True)
    rich = names_of(tr, "F15")
    base_names = [k for k in rich if k in BASE_NAMES]
    sets = {"ridge_few": FEW, "ridge_base": base_names, "ridge_rich": rich}
    cands = []
    ytr = np.concatenate([r["y15"] for r in tr])
    for cname, names in sets.items():
        Xtr = mat(tr, "F15", names)
        for lam in LAMS:
            M = ridge_fit(Xtr, ytr, lam)
            b, c = eval_model(va, "F15", names, M)
            cands.append({"cls": cname, "hyper": {"lam": lam}, "names": names, "val": ratio_stats(b, c), "model": M})
    quad = add_quad(tr, "F15", list(rich))
    add_quad(va, "F15", list(rich))
    Xtr = mat(tr, "F15", quad)
    for lam in LAMS:
        M = ridge_fit(Xtr, ytr, lam)
        b, c = eval_model(va, "F15", quad, M)
        cands.append({"cls": "ridge_rich_quad", "hyper": {"lam": lam}, "names": quad, "val": ratio_stats(b, c),
                      "model": M})
    # pooled over layers 14 and 15 (one coefficient set), applied at layer 15
    common = [k for k in rich if k in tr[0]["F14"]]
    Xp = np.concatenate([mat(tr, "F15", common), mat(tr, "F14", common)], 0)
    yp = np.concatenate([ytr, np.concatenate([r["y14"] for r in tr])])
    for lam in LAMS:
        M = ridge_fit(Xp, yp, lam)
        b, c = eval_model(va, "F15", common, M)
        cands.append({"cls": "pooled_ridge", "hyper": {"lam": lam}, "names": common, "val": ratio_stats(b, c),
                      "model": M})
    # two-layer chain: ridge at layer 14 (its own table), propagated exactly, then ridge at layer 15
    n14 = names_of(tr, "F14")
    X14 = mat(tr, "F14", n14)
    y14 = np.concatenate([r["y14"] for r in tr])
    for lam14 in (1e-5, 1e-3, 1e-1):
        M14 = ridge_fit(X14, y14, lam14)
        trc, vac = [], []
        for src, dst in ((tr, trc), (va, vac)):
            for r in src:
                d14 = ridge_pred(M14, np.stack([r["F14"][k] for k in n14], 1))
                F, y = chain_l15(r, d14)
                dst.append({"F15": F, "y15": y, "y15_base": r["y15"]})
        b_only = np.array([float((r["y15_base"] ** 2).mean()) for r in vac])
        c_only = np.array([float((r["y15"] ** 2).mean()) for r in vac])
        cands.append({"cls": "chain_ridge", "hyper": {"lam14": lam14, "lam15": None}, "names": [], "names14": n14,
                      "val": ratio_stats(b_only, c_only), "model": {"M14": M14}})
        Xc = mat(trc, "F15", rich)
        yc = np.concatenate([r["y15"] for r in trc])
        for lam15 in LAMS:
            M15 = ridge_fit(Xc, yc, lam15)
            yh = split_pred(M15, vac, "F15", rich)
            c = np.array([float(((r["y15"] - h) ** 2).mean()) for r, h in zip(vac, yh)])
            cands.append({"cls": "chain_ridge", "hyper": {"lam14": lam14, "lam15": lam15}, "names": rich,
                          "names14": n14, "val": ratio_stats(b_only, c), "model": {"M14": M14, "M15": M15}})
        del trc, vac
    print(f"ridge family done {time.time() - t0:.1f}s", flush=True)
    for r in cands:
        print(r["cls"], r["hyper"], round(r["val"]["ratio_of_means"], 5), round(r["val"]["mean_ratio"], 5),
              r["val"]["better_on"], flush=True)
    # tiny per-neuron MLPs on the rich layer-15 table (early stopping on 40-49 is part of the selection)
    Xtr, Xva = mat(tr, "F15", rich), mat(va, "F15", rich)
    yva = np.concatenate([r["y15"] for r in va])
    for hidden in (8, 16, 32):
        for wd in (0.0, 1e-4, 1e-3, 1e-2):
            M = mlp_fit(Xtr, ytr, Xva, yva, hidden, wd, seed=0)
            b, c = eval_model(va, "F15", rich, M)
            cands.append({"cls": "mlp", "hyper": {"hidden": hidden, "wd": wd, "epochs": M["epochs"],
                                                  "best_epoch": M["best_epoch"]},
                          "names": rich, "val": ratio_stats(b, c), "model": M})
            print("mlp", hidden, wd, M["best_epoch"], round(cands[-1]["val"]["ratio_of_means"], 5),
                  round(time.time() - t0, 1), flush=True)
    best = min(cands, key=lambda r: r["val"]["ratio_of_means"])
    tol = 0.003
    pick = min((r for r in cands if r["val"]["ratio_of_means"] <= best["val"]["ratio_of_means"] + tol),
               key=lambda r: (RANK[r["cls"]], r["val"]["ratio_of_means"]))
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "dump": str(dump), "fit": [FIT[0], FIT[-1]],
           "select": [SEL[0], SEL[-1]], "rule": f"lowest 40-49 ratio of means; simplest class within {tol}",
           "candidates": [{k: v for k, v in r.items() if k != "model"} for r in cands],
           "best": {k: v for k, v in best.items() if k != "model"},
           "pick": {k: v for k, v in pick.items() if k != "model"}}
    np.savez(HERE / "lc_select_model.npz", **flat_model(pick))
    (HERE / "lc_select.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("best", best["cls"], best["hyper"], best["val"]["ratio_of_means"])
    print("pick", pick["cls"], pick["hyper"], pick["val"]["ratio_of_means"], pick["val"]["mean_ratio"])


def flat_model(c):
    out = {"cls": np.array(c["cls"]), "names": np.array(c["names"] or [""]),
           "names14": np.array(c.get("names14") or [""])}
    M = c["model"]
    parts = {"": M} if "kind" in M else M
    for pre, m in parts.items():
        for k, v in m.items():
            if k == "P":
                for kk, vv in v.items():
                    out[f"{pre}P_{kk}"] = vv
            elif k != "kind":
                out[f"{pre}{k}"] = np.asarray(v)
        out[f"{pre}kind"] = np.array(m["kind"])
    return out


def unflat(z, pre=""):
    if str(z[pre + "kind"]) == "ridge":
        return {"kind": "ridge", "mu": z[pre + "mu"], "sd": z[pre + "sd"], "beta": z[pre + "beta"],
                "b0": float(z[pre + "b0"])}
    return {"kind": "mlp", "mu": z[pre + "mu"], "sd": z[pre + "sd"], "ys": float(z[pre + "ys"]),
            "P": {k: z[f"{pre}P_{k}"] for k in ("W1", "b1", "w2", "b2")}}


def picked(z):
    cls = str(z["cls"])
    names = [str(x) for x in z["names"] if str(x)]
    names14 = [str(x) for x in z["names14"] if str(x)]
    return cls, names, names14


def apply_pick(z, rows):
    """Corrected final-layer residuals of the picked model on rows built by build_rows."""
    cls, names, names14 = picked(z)
    if cls == "ridge_rich_quad":
        add_quad(rows, "F15", [])
    if cls == "chain_ridge":
        M14 = unflat(z, "M14")
        M15 = unflat(z, "M15") if "M15kind" in z.files else None
        out = []
        for r in rows:
            F, y = chain_l15(r, ridge_pred(M14, np.stack([r["F14"][k] for k in names14], 1)))
            out.append(y if M15 is None else y - ridge_pred(M15, np.stack([F[k] for k in names], 1)))
        return out
    M = unflat(z)
    return [r["y15"] - h for r, h in zip(rows, split_pred(M, rows, "F15", names))]


def cmd_importance(args):
    """Which features matter for the picked model, on the selection set 40-49 only: drop in the 40-49 gain when one
    feature is shuffled within each MLP (mean of 5 shuffles), and the ridge's standardized coefficients."""
    dump = Path(args.dump)
    z = np.load(HERE / "lc_select_model.npz")
    cls, names, names14 = picked(z)
    va = build_rows(dump, SEL, chain=(cls == "chain_ridge"))
    if cls == "ridge_rich_quad":
        add_quad(va, "F15", [])
    base = np.array([float((r["y15"] ** 2).mean()) for r in va])
    corr = np.array([float((y ** 2).mean()) for y in apply_pick(z, va)])
    gain = 1.0 - corr.sum() / base.sum()
    rng = np.random.default_rng(7)
    imp = {}
    key = "F15" if cls != "chain_ridge" else None
    for k in names if key else []:
        vals = []
        for _ in range(5):
            saved = [r[key][k] for r in va]
            for r in va:
                r[key][k] = rng.permutation(r[key][k])
            c = np.array([float((y ** 2).mean()) for y in apply_pick(z, va)])
            vals.append(1.0 - c.sum() / base.sum())
            for r, s_ in zip(va, saved):
                r[key][k] = s_
        imp[k] = round(gain - float(np.mean(vals)), 6)
    out = {"cls": cls, "gain_40_49": gain, "perm_importance_gain_drop": dict(sorted(imp.items(), key=lambda kv: -kv[1]))}
    if str(z["kind"] if "kind" in z.files else "") == "ridge":
        out["std_coef"] = dict(sorted(zip(names, [float(b) for b in z["beta"]]), key=lambda kv: -abs(kv[1])))
    p = HERE / "lc_importance.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def cmd_final(args):
    """The picked model on 50-59, exactly once: per-MLP final MSE ratio corrected / base with a paired bootstrap."""
    dump = Path(args.dump)
    lock = HERE / "lc_final.json"
    if lock.exists():
        sys.exit("refused: the 50-59 report exists (it is made exactly once)")
    sel = json.loads((HERE / "lc_select.json").read_text(encoding="utf-8"))
    z = np.load(HERE / "lc_select_model.npz")
    cls, names, names14 = picked(z)
    lock.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cls": cls}), encoding="utf-8")
    rows = build_rows(dump, REP, chain=(cls == "chain_ridge"))
    base = np.array([float((r["y15"] ** 2).mean()) for r in rows])
    corr = np.array([float((y ** 2).mean()) for y in apply_pick(z, rows)])
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mlps": REP, "cls": cls, "pick": sel["pick"]["hyper"],
           "base_final_mse": [float(x) for x in base], "corr_final_mse": [float(x) for x in corr],
           "mean_base": float(base.mean()), "mean_corr": float(corr.mean())}
    out.update(ratio_stats(base, corr))
    out.update(boot_ci(base, corr))
    lock.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "select", "importance", "final"])
    ap.add_argument("--dump", default=str(HERE / "lc_dump_2lb"))
    args = ap.parse_args()
    {"diag": cmd_diag, "select": cmd_select, "importance": cmd_importance, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
