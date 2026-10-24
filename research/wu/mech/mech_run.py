"""Run a hooked copy of the submitted engine (research/wu/mech/est/mech_*) on public mini networks with one setting of
its 3 T^T entries, the way ablate.py meters a prediction (setup once, predict inside flopscope.BudgetContext), and
write one JSON line per network: final-layer and per-layer MSE against the dataset's Monte Carlo means, FLOPs, the
band sets per layer, and a sha of the prediction. Public mini rows 0-59 only (held-out rows 60-99 are refused).

The 3 T^T product M1 of layer t (rows i: the single index, columns j: the doubled index) is
    M1[i, j] = 3 kappa3(z_i, z_j, z_j) + F[i, j]  (i != j),   M1[i, i] = 3 t3_i,
F the K4 star (3,1) fold (r25a k4f). The band (when the file restricts) keeps rows i in Ia (the top-Pa neurons by
alpha) and columns j in Ib (the Pb smallest |alpha|); D = the off-diagonal entries outside Ia x Ib.

Settings (--spec, comma separated key=value; --override "<layer>:key=value;..." for one layer):
  diag = apa | exact | mc   the diagonal 3 t3: as the band-off product leaves it (one APA level), the exact sum, or
                            3 t3_mc (Monte Carlo kappa3(z_i, z_i, z_i))
  keep = apa | exact | mc   entries in Ia x Ib: as computed (APA), the exact product, or 3 TT_mc + F
  drop = apa | exact | zero | zrow | zcol | mc | mcnf | engnf | xnf | mcnoise
                            entries in D: as computed, the exact product, 0, 3 TT_mc + F, 3 TT_mc, M1 - F (APA),
                            the exact product - F, 3 (TT_mc + SE * N(0, 1)) + F (the Monte Carlo noise doubled);
                            zrow / zcol: 0 only on the dropped entries whose row / column is outside the kept set
  at = band | all           layers changed: those where the submitted rule restricts (D not empty), or 1..14
  tscale, kscale, dscale    multiply the off-diagonal entries (all, Ia x Ib, D) after the settings above
TT_mc[i, j] = kappa3(z_i, z_j, z_j) from research/wu/mech/mc/mc_mini<id>.npz (mc_truth.py). The file mech_prod / mech_nos7
restricts its own product (band on); there the hook records only (no setting may change entries).

    .venv-fs0123/bin/python research/wu/mech/mech_run.py --est research/wu/mech/est/mech_nott --ids 0,1 \
        --spec diag=exact,keep=exact,drop=zero --tag xz --out research/wu/mech/runs/xz.jsonl [--record research/wu/mech/rec]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # research/wu
import ablate as AB  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"
BUDGET = 2 ** 41
N, L = 1024, 16
PROD_TT = None  # the submitted file's _SAT_BINS["tt"] table (read from mech_prod's source at start)


def prod_tt_table(est_dir: Path) -> dict:
    import ast
    src = est_dir.parent / "mech_prod" / "estimator.py"
    if not src.exists():  # cuts without a band-on hooked copy (est_r37, est_r38x): opr30, r37 and r38 share this table
        src = HERE / "est" / "mech_prod" / "estimator.py"
    for ln in src.read_text(encoding="utf-8").split("\n"):
        if ln.startswith("_SAT_BINS = "):
            return ast.literal_eval(ln.partition("  #")[0][len("_SAT_BINS = "):])["tt"]
    raise RuntimeError("no _SAT_BINS line")


def parse_kv(s: str) -> dict:
    return dict(x.split("=", 1) for x in s.split(",") if x)


class Mech:
    def __init__(self, G, est, spec, overrides, mc, own_band, record, rec_layers=None, cov_layers=()):
        self.G, self.spec, self.over, self.mc, self.own, self.record = G, spec, overrides, mc, own_band, record
        self.rec_layers, self.cov_layers, self.covrec = rec_layers, set(cov_layers), {}
        c = est._cost
        self.d, self.ht = int(c["sw_depth"]), {int(x) for x in c["sw_hi_tt"]}
        self.rec = {}  # layer -> dict of arrays (record mode)
        self.info = {}  # layer -> small dict (always)
        self.Mf = None
        self.rng = np.random.default_rng(12345)

    # ---- the band sets the submitted file uses at this alpha ----
    def band(self, t, a, Pa, Pb, ord_t, ordb_t):
        n = a.shape[0]
        if self.own:  # the file restricts its own product: its sets
            Ia = np.asarray(ord_t)[n - Pa:] if Pa < n else None
            Ib = np.asarray(ordb_t)[:Pb] if Pb < n else None
            return Pa, Pb, Ia, Ib
        G = self.G
        S = G["_SAT"]
        q_ = S.get("q", 32)
        dtt = self.d + 1 if t in self.ht else self.d
        mult = max(1 << dtt, q_)
        af = np.asarray(a, dtype=np.float32)
        o = np.argsort(af)
        thr = np.array([-S["in"], -S["out"], -S["ta"], -S["tb"], S["tb"]], dtype=np.float32)
        cs = np.searchsorted(af[o], thr).tolist()
        c_ta, c_tb = n - cs[2], (cs[4] - cs[3]) if S["tb_band"] else n - cs[3]
        Pa = G["_sat_size"]("alive_" + str(S["ta"]), t, n, mult, c_ta)
        Pb = G["_sat_size"](("band_" if S["tb_band"] else "alive_") + str(S["tb"]), t, n, mult, c_tb)
        tt = {(G["_sat_pad"](fa, n, mult), G["_sat_pad"](fb, n, mult)) for fa, fb in PROD_TT.get(str(t), ())}
        tt = sorted((p for p in tt if p[0] < n or p[1] < n), key=lambda p: (p[0] * p[1], p))
        if Pa < n or Pb < n:
            Pa, Pb = next(((x, y) for x, y in tt if x >= Pa and y >= Pb), (n, n))
        Ia = o[n - Pa:] if Pa < n else None
        Ib = (np.argsort(np.abs(af))[:Pb] if S["tb_band"] else o[n - Pb:]) if Pb < n else None
        return Pa, Pb, Ia, Ib

    # ---- hook 1: the fold's share of 3 T^T, F = sum_s H_s^T (3 QF_s o G_s) + G_s^T (QF_s o H_s) ----
    def fold(self, t, h_, g_, QF, mc, ns):
        H, G, Q = np.asarray(h_), np.asarray(g_), np.asarray(QF)
        F = np.zeros((H.shape[2], H.shape[2]))
        Fc = np.zeros(H.shape[2])
        for s in range(H.shape[0]):
            h, g, q = H[s].astype(np.float64), G[s].astype(np.float64), Q[s].astype(np.float64)
            F += h.T @ (3.0 * q * g) + g.T @ (q * h)
            if s < mc:  # the compressed slots' share of DGF (their legs as they are at this point)
                Fc += np.einsum("mi,mi,mi->i", q, g, h)
        self.Mf, self.DGFc = F, Fc

    # ---- hook 2: M1 complete ----
    def tt(self, t, M1, a, Pa, Pb, ord_t, ordb_t, mc, ns, LA, TA, DGF):
        M = np.asarray(M1)
        n = M.shape[0]
        Pa, Pb, Ia, Ib = self.band(t, a, Pa, Pb, ord_t, ordb_t)
        kr = np.ones(n, bool) if Ia is None else np.zeros(n, bool)
        if Ia is not None:
            kr[Ia] = True
        kc = np.ones(n, bool) if Ib is None else np.zeros(n, bool)
        if Ib is not None:
            kc[Ib] = True
        off = ~np.eye(n, dtype=bool)
        K = np.outer(kr, kc) & off
        D = ~np.outer(kr, kc) & off
        self.info[t] = {"Pa": int(Pa), "Pb": int(Pb), "n_keep": int(K.sum()), "n_drop": int(D.sum()), "mc": int(mc), "ns": int(ns)}
        F = self.Mf
        rec_t = self.record and (self.rec_layers is None or t in self.rec_layers)
        need_x = (not self.own) and (rec_t or "exact" in self.setting(t).values())
        Mx = None
        if need_x:  # the exact product of the same operands (float64) and its diagonal as the band-on code makes it
            Lr, Ar = (np.asarray(x, dtype=np.float64) for x in LA)
            Lt, At = (np.asarray(x, dtype=np.float64) for x in TA)
            Mx = Lr.T @ Ar + Lt.T @ At
            dx = np.einsum("ri,ri->i", Lr, Ar) + np.einsum("ri,ri->i", Lt, At)
            if DGF is not None:
                dx = dx - 4.0 * np.asarray(DGF, dtype=np.float64)
            np.fill_diagonal(Mx, dx)
        if rec_t:
            r = {"a": np.asarray(a, dtype=np.float32).copy(), "kr": kr, "kc": kc, "M1": M.astype(np.float32),
                 "diag": np.diagonal(M).astype(np.float64).copy()}
            if F is not None:
                r["F"] = F.astype(np.float32)
            if Mx is not None:
                r["Mx"] = Mx.astype(np.float32)
            if DGF is not None:
                r["DGF"] = np.asarray(DGF, dtype=np.float64).copy()
                r["DGFc"] = self.DGFc.copy()
            self.rec[t] = r
        if self.own:
            return
        st = self.setting(t)
        ks = float(st.get("kscale", st.get("tscale", 1.0)))
        ds = float(st.get("dscale", st.get("tscale", 1.0)))
        if st["diag"] == "apa" and st["keep"] == "apa" and st["drop"] == "apa" and ks == 1.0 and ds == 1.0:
            return
        active = (st["at"] == "all") or bool(D.any())
        if not active or t not in self.mc_layers():
            return
        new = M.astype(np.float64)
        TT3 = None
        if (st["keep"] == "mc" or st["drop"] in ("mc", "mcnf", "mcnoise")) and f"TT_{t}" not in self.mc:
            st = dict(st, keep="apa" if st["keep"] == "mc" else st["keep"],
                      drop="apa" if st["drop"] in ("mc", "mcnf", "mcnoise") else st["drop"])  # no truth at this layer (t = 1)
        if st["keep"] == "mc" or st["drop"] in ("mc", "mcnf", "mcnoise"):
            TT3 = 3.0 * self.mc[f"TT_{t}"].astype(np.float64)
        if st["keep"] == "exact":
            new[K] = Mx[K]
        elif st["keep"] == "mc":
            new[K] = TT3[K] + F[K]
        dr = st["drop"]
        if dr == "exact":
            new[D] = Mx[D]
        elif dr == "zero":
            new[D] = 0.0
        elif dr == "zrow":  # only the entries dropped for their row (single index below the alive cut)
            new[D & ~kr[:, None]] = 0.0
        elif dr == "zcol":  # only the entries dropped for their column (doubled index outside the band)
            new[D & ~kc[None, :]] = 0.0
        elif dr == "mc":
            new[D] = TT3[D] + F[D]
        elif dr == "mcnf":
            new[D] = TT3[D]
        elif dr == "engnf":
            new[D] = M.astype(np.float64)[D] - F[D]
        elif dr == "xnf":
            new[D] = Mx[D] - F[D]
        elif dr == "mcnoise":
            se = 3.0 * self.mc[f"SE_{t}"].astype(np.float64)
            new[D] = TT3[D] + se[D] * self.rng.standard_normal(int(D.sum())) + F[D]
        if ks != 1.0:
            new[K] *= ks
        if ds != 1.0:
            new[D] *= ds
        if st["diag"] == "exact":
            np.fill_diagonal(new, np.diagonal(Mx))
        elif st["diag"] == "mc":
            np.fill_diagonal(new, 3.0 * np.asarray(self.mc["t3"][t], dtype=np.float64))
        M[...] = new.astype(np.float32)

    # ---- hook 3: this layer's final C' ----
    def cov(self, t, Cv):
        if t in self.cov_layers:
            self.covrec[t] = np.asarray(Cv).astype(np.float32)

    def setting(self, t):
        s = dict(self.spec)
        s.update(self.over.get(t, {}))
        return s

    def mc_layers(self):
        return range(1, 15) if self.spec.get("at") == "all" else range(2, 15)


def main() -> None:
    global PROD_TT
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--ids", default="0,1")
    ap.add_argument("--spec", default="diag=apa,keep=apa,drop=apa")
    ap.add_argument("--override", default="", help='"<layer>:key=value,...;<layer>:..."')
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default="", help="directory for the per-layer arrays (npz per network)")
    ap.add_argument("--mcdir", default=str(HERE / "mc"))
    ap.add_argument("--rec-layers", default="", help="record only these layers (comma separated; default all)")
    ap.add_argument("--cov-layers", default="", help="keep this layer's final C' (comma separated; saved with --record)")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    assert all(0 <= i < 60 for i in ids), "held-out mini rows 60-99 are not read"
    est_dir = Path(a.est).resolve()
    PROD_TT = prod_tt_table(est_dir)
    spec = {"diag": "apa", "keep": "apa", "drop": "apa", "at": "band"}
    spec.update(parse_kv(a.spec))
    over = {}
    for part in filter(None, a.override.split(";")):
        lt, _, kv = part.partition(":")
        over[int(lt)] = parse_kv(kv)
    own = est_dir.name.startswith(("mech_prod", "mech_nos7"))  # band-on files (mech_prod_noext, ...): record only
    if own:
        assert spec == {"diag": "apa", "keep": "apa", "drop": "apa", "at": "band"} and not over, "band-on files: record only"

    import flopscope as flops
    from whestbench import SetupContext
    from whestbench.domain import MLP
    import flopscope.numpy as fnp

    Est, entry = AB.load_estimator(est_dir)
    G = Est.__init__.__globals__  # the estimator module's globals: _MECH is set there
    assert "_MECH" in G, "not a hooked file"
    est = Est()
    est.setup(SetupContext(width=N, depth=L, flop_budget=BUDGET, api_version="1.0", submission_dir=str(entry.parent), seed=0))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for mlp_id, seed, name, W, gt in AB.load_rows(SNAP, "mini", ids):
        mcz = None
        if not own and (any(v in ("mc", "mcnf", "mcnoise") for v in spec.values()) or
                        any(v in ("mc", "mcnf", "mcnoise") for o in over.values() for v in o.values())):
            mcz = dict(np.load(Path(a.mcdir) / f"mc_mini{mlp_id}.npz"))
        hk = Mech(G, est, spec, over, mcz, own, bool(a.record),
                  {int(x) for x in a.rec_layers.split(",") if x} or None, [int(x) for x in a.cov_layers.split(",") if x])
        G["_MECH"] = hk
        mlp = MLP(width=N, depth=L, weights=[fnp.asarray(W[t]) for t in range(L)], seed=seed, name=name)
        mlp.validate()
        t0 = time.time()
        with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=10000.0, quiet=True) as ctx:
            pred = est.predict(mlp, BUDGET)
        G["_MECH"] = None
        pred = np.asarray(pred, dtype=np.float64)
        tr = gt.astype(np.float64)
        mse = ((pred - tr) ** 2).mean(axis=1)
        rec = {"tag": a.tag, "est": est_dir.name, "spec": spec, "override": a.override, "mlp_id": mlp_id,
               "final_mse": float(mse[-1]), "layer_mse": [float(v) for v in mse], "flops": int(ctx.flops_used),
               "finite": bool(np.isfinite(pred).all()), "sec": round(time.time() - t0, 1),
               "pred_sha16": hashlib.sha256(np.ascontiguousarray(np.asarray(pred[-1], dtype=np.float64)).tobytes()).hexdigest()[:16],
               "pred_sha16_f32": hashlib.sha256(np.ascontiguousarray(np.asarray(pred[-1], dtype=np.float32)).tobytes()).hexdigest()[:16],
               "band": {str(k): v for k, v in sorted(hk.info.items())}}
        with out.open("a", encoding="utf-8") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("tag", "mlp_id", "final_mse", "sec", "pred_sha16_f32")}), flush=True)
        if a.record:
            rd = Path(a.record)
            rd.mkdir(parents=True, exist_ok=True)
            arr = {"pred": pred}
            for t, r in hk.rec.items():
                for k, v in r.items():
                    arr[f"{k}_{t}"] = v
            for t, v in hk.covrec.items():
                arr[f"Cp_{t}"] = v
            np.savez(rd / f"{a.tag}_mini{mlp_id}.npz", **arr)
        del hk, mlp, W, gt, pred


if __name__ == "__main__":
    main()
