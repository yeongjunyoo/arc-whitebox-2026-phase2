"""lc3: does a bigger or richer learned correction break the lc saturation?

The lc form of rounds 1 and 2: per-layer models of the teacher-forced local error in the "base" feature mode (features
at the base engine's own mean), carried to the last layer by the linear chain dp_L = D_L * (W_L^T dp_{L-1}) + u_L,
D = w11 + w41 t3 / 6 + w51 d4 / 24 (research/r2_fit.final_pred, chain "lin"; within 0.0002 of the exact chain there).

Held out: S = full 0-47 (research/r2_split.json "select", spent as a selection set in round 2; here it only measures).
Fit pool: mini 20-43 and full 96-111 (research/r2_cache/s6_192), full 112-303 (research/lc3_cache/s6_192, shards 7-18,
the round-2 fit shards, dumped with research/r2_dump.py). Early stopping: mini 44-49 (fixed). Never read: mini 0-19
and 50-99 here, R = full 48-95, full shards 19-21 (id >= 304).

Neighbour features (feats): for neuron i of layer L, through the incoming column W_L[:, i] and the previous layer's
ReLU outputs under the Gaussian approximation of the engine's (m, s): diagonal K2..K6 sums, their readout terms,
effective input count, W^2-weighted averages of the previous layer's base features, and the off-diagonal energy of
W^T diag(k2) W (the diagonal-C proxy of the preactivation covariance).

    .venv/bin/python research/lc3_curve.py feats --set S,VAL,POOL           (research/lc3_feat/<tag>.npz)
    .venv/bin/python research/lc3_curve.py curve --sizes 5,10,20,40 --seeds 3 --model r1 --tag base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from math import comb
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_chain as lc  # noqa: E402
import lc_fit as lf  # noqa: E402

N = 1024
R2C = HERE / "r2_cache" / "s6_192"
L3C = HERE / "lc3_cache" / "s6_192"
FEAT = HERE / "lc3_feat"
S_TAGS = [f"full_{i:04d}" for i in range(48)]
VAL_TAGS = [f"mini_{m:03d}" for m in range(44, 50)]
POOL0 = [f"mini_{m:03d}" for m in range(20, 44)] + [f"full_{i:04d}" for i in range(96, 112)]
NB = ["dvar", "neff", "k3d", "k3o", "k4d", "k4o", "k5d", "k6d", "R5n", "R6n", "xa", "xR4n", "xk4", "xPhi", "cn2",
      "cn4", "soffd"]


def check_tag(tag, report=False):
    """research/lc3_split.json: fit = mini 0-59 and full shards 6-18, 22-62; S = full 0-47; R = full 48-95 (only the
    locked report reads it); never mini 60-99 or full shards 19-21 (ids 304-351)."""
    src, i = tag.split("_")
    i = int(i)
    if src == "mini" and not 0 <= i <= 59:
        sys.exit(f"refused: {tag}")
    if src == "full" and (304 <= i <= 351 or i > 999 or (48 <= i <= 95 and not report)):
        sys.exit(f"refused: {tag}")


def cache_path(tag, report=False):
    check_tag(tag, report)
    src, i = tag.split("_")
    i = int(i)
    if (src == "full" and i >= 112) or (src == "mini" and not 20 <= i <= 49):
        return L3C / f"{tag}.npz"
    return R2C / f"{tag}.npz"


def pool1():
    """Fit MLPs added in lc3 (full shards 7-18 and 22-62, mini 0-19 and 50-59 converted from round 1), only those whose
    log line exists (written after the npz is complete)."""
    log = L3C / "log.jsonl"
    if not log.exists():
        return []
    tags = {json.loads(ln)["tag"] for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()}
    ok = [t for t in tags if (t.startswith("full_") and 112 <= int(t[5:]) <= 999 and not 304 <= int(t[5:]) <= 351)
          or (t.startswith("mini_") and (int(t[5:]) < 20 or 50 <= int(t[5:]) <= 59))]
    return sorted(ok)


MAC_MINI = (Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots"
            / "aa99830fdc09fad15407b10e8e3459d3e18bba0a/prepared/mini")


def weights(tag, with_gt=False, report=False):
    """(16, n, n) float32 weights (and the (16, n) truth) of a fit-pool, validation or S MLP (R only for the report)."""
    check_tag(tag, report)
    if tag.startswith("full_"):
        z = np.load(HERE / "r2_data" / f"{tag}.npz")
        return (z["W"], z["gt"]) if with_gt else z["W"]
    import k3_measure as km
    if not list(km.MINI.glob("data-*.arrow")):
        km.MINI = MAC_MINI
    W, gt, _, _ = km.load_row(int(tag[5:]))
    return (W, gt) if with_gt else W


# ------------------------------------------------------------------------------------------------ features
def relu_cumulants(a, s):
    """kappa_1..6 of ReLU(Z), Z ~ N(a s, s^2), via M_k = E[(a + G)_+^k] = a M_{k-1} + (k - 1) M_{k-2}."""
    ph = np.exp(-0.5 * a * a) / lf.SQ2PI
    Ph = lf.Phi(a)
    M = [Ph, a * Ph + ph]
    for k in range(2, 7):
        M.append(a * M[k - 1] + (k - 1) * M[k - 2])
    m1 = M[1]
    c = {}
    for k in range(2, 7):
        c[k] = sum(comb(k, j) * (M[j] if j > 0 else 1.0) * (-m1) ** (k - j) for j in range(k + 1))
    k2, k3 = c[2], c[3]
    k4 = c[4] - 3 * c[2] ** 2
    k5 = c[5] - 10 * c[3] * c[2]
    k6 = c[6] - 15 * c[4] * c[2] - 10 * c[3] ** 2 + 30 * c[2] ** 3
    return [m1 * s, k2 * s ** 2, k3 * s ** 3, k4 * s ** 4, k5 * s ** 5, k6 * s ** 6]


def nb_feats(W, z):
    """(15, len(NB), n) float32 neighbour features of one MLP (cache z, weights W (16, n, n))."""
    m, s, t3, d4, kA, pred = (z[k].astype(np.float64) for k in ("m", "s", "t3", "d4", "kA", "pred"))
    out = np.zeros((15, len(NB), N), np.float32)
    W0 = W[0].astype(np.float64)
    for L in range(1, 16):
        i = L - 1
        if L == 1:  # layer 0 preactivation is exactly N(0, |W0[:, j]|^2)
            ps = np.sqrt((W0 * W0).sum(0))
            pa = np.zeros(N)
            pR4n, pk4, pPhi = np.zeros(N), np.zeros(N), np.full(N, 0.5)
        else:
            ps = s[i - 1]
            pa = m[i - 1] / ps
            w, _, Ph = lf.wick(pa, ps)
            pR4n, pk4, pPhi = w[4] * d4[i - 1] / 24.0 / ps, d4[i - 1] / ps ** 4, Ph
        _, k2, k3, k4, k5, k6 = relu_cumulants(pa, ps)
        WL = W[L].astype(np.float64)
        W2 = WL * WL
        W4 = W2 * W2
        S2, S3, S4 = W2.T @ k2, (W2 * WL).T @ k3, W4.T @ k4
        S5, S6, Q4 = (W4 * WL).T @ k5, (W4 * W2).T @ k6, W4.T @ (k2 * k2)
        cn2 = W2.sum(0)
        si = s[i]
        a = m[i] / si
        w, _, _ = lf.wick(a, si)
        Wk = (WL * k2[:, None]).astype(np.float32)
        C = Wk.T @ WL.astype(np.float32)  # W^T diag(k2) W
        soff = ((C.astype(np.float64) ** 2).sum(1) - np.diag(C).astype(np.float64) ** 2) / si ** 4
        del C, Wk
        F = {"dvar": S2 / si ** 2, "neff": np.log(S2 * S2 / Q4) / math.log(N), "k3d": S3 / si ** 3,
             "k3o": t3[i] / si ** 3 - S3 / si ** 3, "k4d": S4 / si ** 4, "k4o": d4[i] / si ** 4 - S4 / si ** 4,
             "k5d": S5 / si ** 5, "k6d": S6 / si ** 6, "R5n": w[5] * S5 / 120.0 / si, "R6n": w[6] * S6 / 720.0 / si,
             "xa": (W2.T @ pa) / cn2, "xR4n": (W2.T @ pR4n) / cn2, "xk4": (W2.T @ pk4) / cn2,
             "xPhi": (W2.T @ pPhi) / cn2, "cn2": cn2, "cn4": W4.sum(0) / cn2 ** 2 * N, "soffd": soff}
        out[i] = np.stack([F[k] for k in NB], 0).astype(np.float32)
    return out


def cmd_feats(args):
    FEAT.mkdir(exist_ok=True)
    tags = expand(args.set)
    t0 = time.time()
    for tag in tags:
        dst = FEAT / f"{tag}.npz"
        if dst.exists():
            continue
        p = cache_path(tag)
        if not p.exists():
            continue
        z = np.load(p)
        W, gt = weights(tag, with_gt=True)
        assert np.allclose(z["gt"], gt[1:16].astype(np.float64), atol=1e-7), tag
        np.savez(dst, NB=nb_feats(W, z))
        del W
        print(tag, f"{time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


def expand(spec):
    out = []
    for part in spec.split(","):
        out += {"S": S_TAGS, "VAL": VAL_TAGS, "POOL0": POOL0, "POOL1": pool1(), "POOL": POOL0 + pool1()}.get(part, [part])
    return out


# ------------------------------------------------------------------------------------------------ data
def load_lean(tag, chain=False, nb=False, vx=True):
    z = np.load(cache_path(tag))
    m, s, t3, d4, kA, pred, gt, loc = (z[k].astype(np.float64) for k in ("m", "s", "t3", "d4", "kA", "pred", "gt", "loc"))
    c = {"tag": tag, "loc": loc, "R": z["R"].astype(np.float32),
         "FB": np.stack([lc.feats_raw(m[i], s[i], t3[i], d4[i], kA[i], pred[i]) for i in range(15)]).astype(np.float32)}
    if nb:
        c["NB"] = np.load(FEAT / f"{tag}.npz")["NB"]
    if vx:
        c["VX"] = z["VX"][:, 1:].astype(np.float32)  # the engine's VEC rows without the constant row, (16, 25, n)
    if chain:
        D = np.zeros((15, N))
        for i in range(15):
            w, _, _ = lf.wick(m[i] / s[i], s[i])
            D[i] = w[1] + w[4] * t3[i] / 6.0 + w[5] * d4[i] / 24.0
        c.update({"D": D, "p15": pred[14], "g15": gt[14], "base_mse": float(((gt[14] - pred[14]) ** 2).mean())})
    return c


def seg_X(c, L, seg):
    parts = [c["FB"][L - 1]]
    if seg.get("rich"):
        parts.append(c["R"][L - 13])
    if seg.get("nb"):
        parts.append(c["NB"][L - 1][[NB.index(k) for k in seg["nb"]]])
    if seg.get("vx"):
        parts.append(c["VX"][L])
    if seg.get("depth"):
        parts.append(np.full((1, N), L / 15.0, np.float32))
    return np.concatenate(parts, 0).T


def seg_xy(C, seg):
    X = np.concatenate([seg_X(c, L, seg) for c in C for L in seg["layers"]], 0)
    y = np.concatenate([c["loc"][L - 1] for c in C for L in seg["layers"]], 0)
    return X, y


# ------------------------------------------------------------------------------------------------ models
def mlp2_fit(Xtr, ytr, Xva, yva, h1, h2, wd, seed=0, lr=1e-3, epochs=150, batch=2048, patience=20):
    """f -> h1 -> h2 -> 1 (ReLU), Adam, L2, early stopping on the validation part; z-scored inputs, scaled y."""
    sc = lf.Std(Xtr)
    Ztr, Zva = sc(Xtr), sc(Xva)
    ys = ytr.std()
    ttr, tva = ytr / ys, yva / ys
    rng = np.random.default_rng(seed)
    f = Ztr.shape[1]
    P = {"W1": rng.standard_normal((f, h1)) * np.sqrt(2.0 / f), "b1": np.zeros(h1),
         "W2": rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1), "b2": np.zeros(h2), "w3": np.zeros(h2),
         "b3": np.zeros(1)}
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    vel = {k: np.zeros_like(v) for k, v in P.items()}

    def fwd(Z, P):
        a1 = np.maximum(Z @ P["W1"] + P["b1"], 0.0)
        a2 = np.maximum(a1 @ P["W2"] + P["b2"], 0.0)
        return a1, a2, a2 @ P["w3"] + P["b3"][0]

    best, best_loss, bad, step, hist = None, np.inf, 0, 0, []
    n = len(ttr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            Z, t = Ztr[idx], ttr[idx]
            a1, a2, o = fwd(Z, P)
            g = 2.0 * (o - t) / len(idx)
            gr = {"w3": a2.T @ g + 2 * wd * P["w3"], "b3": np.array([g.sum()])}
            g2 = np.outer(g, P["w3"]) * (a2 > 0)
            gr["W2"] = a1.T @ g2 + 2 * wd * P["W2"]
            gr["b2"] = g2.sum(0)
            g1 = (g2 @ P["W2"].T) * (a1 > 0)
            gr["W1"] = Z.T @ g1 + 2 * wd * P["W1"]
            gr["b1"] = g1.sum(0)
            step += 1
            for k in P:
                mom[k] = 0.9 * mom[k] + 0.1 * gr[k]
                vel[k] = 0.999 * vel[k] + 0.001 * gr[k] ** 2
                P[k] = P[k] - lr * (mom[k] / (1 - 0.9 ** step)) / (np.sqrt(vel[k] / (1 - 0.999 ** step)) + 1e-8)
        lv = float(((fwd(Zva, P)[2] - tva) ** 2).mean())
        hist.append(lv)
        if lv < best_loss - 1e-6:
            best_loss, best, bad = lv, {k: v.copy() for k, v in P.items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return {"kind": "mlp2", "mu": sc.mu, "sd": sc.sd, "ys": ys, "P": best, "epochs": len(hist)}


def model_pred(M, X):
    if M["kind"] == "mlp2":
        Z = (X - M["mu"]) / M["sd"]
        P = M["P"]
        a1 = np.maximum(Z @ P["W1"] + P["b1"], 0.0)
        a2 = np.maximum(a1 @ P["W2"] + P["b2"], 0.0)
        return (a2 @ P["w3"] + P["b3"][0]) * M["ys"]
    if M["kind"] == "lgb":
        return M["booster"].predict(X) * M["ys"]
    return lf.predict(M, X)


def fit_seg(tr, va, seg, seed):
    X, y = seg_xy(tr, seg)
    if seg["model"] == "ridge":
        return lf.ridge_fit(X.astype(np.float64), y, seg.get("lam", 1e-4))
    Xv, yv = seg_xy(va, seg)
    if seg["model"] == "mlp":
        return lf.mlp_fit(X, y, Xv, yv, seg["hidden"], seg["wd"], seed=seed, batch=2048, epochs=150, patience=20)
    if seg["model"] == "mlp2":
        return mlp2_fit(X, y, Xv, yv, seg["h1"], seg["h2"], seg["wd"], seed=seed)
    if seg["model"] == "lgb":
        import lightgbm as lgb
        ys = y.std()
        dtr, dva = lgb.Dataset(X, y / ys), lgb.Dataset(Xv, yv / ys)
        prm = {"objective": "l2", "learning_rate": seg.get("lr", 0.05), "num_leaves": seg.get("leaves", 31),
               "min_data_in_leaf": seg.get("min_leaf", 200), "feature_fraction": 0.9, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": seg.get("threads", 1), "seed": seed}
        bst = lgb.train(prm, dtr, num_boost_round=seg.get("rounds", 2000), valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        return {"kind": "lgb", "booster": bst, "ys": ys, "rounds": bst.best_iteration}
    raise ValueError(seg["model"])


def model_spec(name):
    A = {"layers": list(range(1, 13)), "model": "mlp", "hidden": 16, "wd": 1e-4, "depth": True}
    B = {"layers": [13, 14, 15], "model": "mlp", "hidden": 32, "wd": 1e-4, "rich": True, "depth": True}
    nb_all = list(NB)
    specs = {
        "r1": [A, B],  # the round-1 pick (r2_fit r1pick_refit_base_lin)
        "r1nb": [dict(A, nb=nb_all), dict(B, nb=nb_all)],
        "r1nbk": [dict(A, nb=["k5d", "k6d", "R5n", "R6n", "k3d", "k4d", "neff"]),
                  dict(B, nb=["k5d", "k6d", "R5n", "R6n", "k3d", "k4d", "neff"])],
        "r1x": [dict(A, nb=["xa", "xR4n", "xk4", "xPhi"]), dict(B, nb=["xa", "xR4n", "xk4", "xPhi"])],
        "r1soff": [dict(A, nb=["soffd", "dvar"]), dict(B, nb=["soffd", "dvar"])],
        "wide": [dict(A, hidden=64), dict(B, hidden=64)],
        "deep": [dict(A, model="mlp2", h1=64, h2=32), dict(B, model="mlp2", h1=64, h2=32)],
        "deepnb": [dict(A, model="mlp2", h1=64, h2=32, nb=nb_all), dict(B, model="mlp2", h1=64, h2=32, nb=nb_all)],
        "lgb": [dict(A, model="lgb"), dict(B, model="lgb")],
        "lgbnb": [dict(A, model="lgb", nb=nb_all), dict(B, model="lgb", nb=nb_all)],
        "perlayer": [dict(A, layers=[L]) for L in range(1, 13)] + [dict(B, layers=[L]) for L in (13, 14, 15)],
        "r1vx": [dict(A, vx=True), dict(B, vx=True)],
        "r1nbvx": [dict(A, nb=nb_all, vx=True), dict(B, nb=nb_all, vx=True)],
        "deepnbvx": [dict(A, model="mlp2", h1=64, h2=32, nb=nb_all, vx=True),
                     dict(B, model="mlp2", h1=64, h2=32, nb=nb_all, vx=True)],
    }
    return specs[name]


def outputs(c, segs, fitted):
    u = np.zeros((15, N))
    for seg, M in zip(segs, fitted):
        for L in seg["layers"]:
            u[L - 1] = model_pred(M, seg_X(c, L, seg))
    return u


# ------------------------------------------------------------------------------------------------ scoring
def chain_final(c, Ws, u):
    dp = u[0].copy()
    for L in range(2, 16):
        dp = c["D"][L - 1] * (Ws[L].T @ dp) + u[L - 1]
    return c["p15"] + dp


def yardsticks(tags):
    sel = json.loads((HERE / "r2_select_s6_192.json").read_text(encoding="utf-8"))["mse"]
    lc2k = {}
    for f in ("r2_metric_cfg_s_a_1005_052508.json", "r2_metric_cfg_s_b_1005_052528.json"):
        for row in json.loads((HERE / f).read_text(encoding="utf-8"))["rows"]:
            lc2k[row["tag"]] = row["lc2k"]["final_mse"]
    return ({t: sel["base"][t] for t in tags}, {t: sel["lc_r1"][t] for t in tags}, {t: lc2k[t] for t in tags})


def stats(num, den, tags):
    a = np.array([num[t] for t in tags])
    b = np.array([den[t] for t in tags])
    out = lf.ratio_stats(b, a)
    out.update(lf.boot_ci(b, a, reps=5000))
    out.pop("per_mlp")
    return out


def cmd_curve(args):
    t0 = time.time()
    nb = any(seg.get("nb") for name in args.model.split(",") for seg in model_spec(name))
    vx = any(seg.get("vx") for name in args.model.split(",") for seg in model_spec(name))
    pool = POOL0 + pool1()
    if nb:
        pool = [t for t in pool if (FEAT / f"{t}.npz").exists()]
    if args.pool_max:
        pool = pool[: args.pool_max]
    sizes = [int(x) for x in args.sizes.split(",") if int(x) <= len(pool)]
    va = [load_lean(t, nb=nb, vx=vx) for t in VAL_TAGS]
    need = set()
    orders = []
    for sd in range(args.seeds):
        o = list(np.random.default_rng(1000 + sd).permutation(pool))
        orders.append(o)
        need |= set(o[: max(sizes)])
    data = {t: load_lean(t, nb=nb, vx=vx) for t in sorted(need)}
    print(f"pool {len(pool)} sizes {sizes} seeds {args.seeds} loaded {len(data)} {time.time() - t0:.0f}s", flush=True)
    runs = []
    for name in args.model.split(","):
        segs = model_spec(name)
        for size in sizes:
            for sd in range(args.seeds if size < len(pool) else min(args.seeds, args.seeds_full)):
                tr = [data[t] for t in orders[sd][:size]]
                fitted = [fit_seg(tr, va, seg, sd) for seg in segs]
                runs.append({"model": name, "size": size, "seed": sd, "segs": segs, "fitted": fitted,
                             "train": orders[sd][:size]})
                print(f"fit {name} n={size} seed={sd} {time.time() - t0:.0f}s "
                      f"{[M.get('epochs', M.get('rounds')) for M in fitted]}", flush=True)
    del data
    base, lcr1, lc2k = yardsticks(S_TAGS)
    res = [dict() for _ in runs]
    loc_err = np.zeros((len(runs), 15))
    loc_tot = np.zeros(15)
    for tag in S_TAGS:
        c = load_lean(tag, chain=True, nb=nb, vx=vx)
        assert abs(c["base_mse"] - base[tag]) < 1e-6 * base[tag], (tag, c["base_mse"], base[tag])
        Wf, gt = weights(tag, with_gt=True)
        assert np.allclose(c["g15"], gt[15].astype(np.float64), atol=1e-7), tag
        Ws = [None] + [Wf[L].astype(np.float64) for L in range(1, 16)]
        del Wf
        loc_tot += (c["loc"] ** 2).sum(1)
        for j, r in enumerate(runs):
            u = outputs(c, r["segs"], r["fitted"])
            res[j][tag] = float(((c["g15"] - chain_final(c, Ws, u)) ** 2).mean())
            loc_err[j] += ((c["loc"] - u) ** 2).sum(1)
        del Ws
    rows = []
    for j, r in enumerate(runs):
        row = {"model": r["model"], "size": r["size"], "seed": r["seed"],
               "vs_base": stats(res[j], base, S_TAGS), "vs_lc": stats(res[j], lcr1, S_TAGS),
               "vs_lc2k": stats(res[j], lc2k, S_TAGS),
               "local_r2_S": [float(1 - loc_err[j][i] / loc_tot[i]) for i in range(15)],
               "epochs": [M.get("epochs", M.get("rounds")) for M in r["fitted"]], "mse_S": res[j]}
        rows.append(row)
        print(f"{r['model']:8s} n={r['size']:4d} s{r['seed']} vs base {row['vs_base']['ratio_of_means']:.4f} "
              f"vs lc {row['vs_lc']['ratio_of_means']:.4f} {[round(x, 4) for x in row['vs_lc']['ratio_of_means_ci95']]} "
              f"vs lc2k {row['vs_lc2k']['ratio_of_means']:.4f}  localR2 L1 {row['local_r2_S'][0]:.3f} "
              f"L8 {row['local_r2_S'][7]:.3f} L15 {row['local_r2_S'][14]:.3f}", flush=True)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "pool": pool, "val": VAL_TAGS, "S": S_TAGS,
           "lc_vs_base_S": stats(lcr1, base, S_TAGS), "lc2k_vs_base_S": stats(lc2k, base, S_TAGS),
           "lc2k_vs_lc_S": stats(lc2k, lcr1, S_TAGS), "rows": rows, "secs": round(time.time() - t0)}
    p = HERE / f"lc3_curve_{args.tag}_{time.strftime('%m%d_%H%M')}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", p.name, f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "curve"])
    ap.add_argument("--set", default="S,VAL,POOL")
    ap.add_argument("--sizes", default="5,10,20,40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seeds-full", type=int, default=2)
    ap.add_argument("--pool-max", type=int, default=0)
    ap.add_argument("--model", default="r1")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    {"feats": cmd_feats, "curve": cmd_curve}[args.cmd](args)


if __name__ == "__main__":
    main()
