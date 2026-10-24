"""R25-A GRU retrain, step 1 (Nova 2026-10-14, coordinator 13:4x): run one folder estimator exactly as shipped (float32,
predict inside flops.BudgetContext(2**41), the corrector on) on public fit MLPs and keep what its W-message corrector
reads, by wrapping the module's _wmsg (research only, flopscope untouched): QB (depth, 9, n) [phi, Phi, 1, dA, lam'_t,
lam'_t+1, t3, a, var] after the corrector's own t3 fill, SVB, VX, the base mean of the last layer before the corrector
(pre15) and the shipped corrector's dp15. Writes, per MLP, a cache in the format research/lc3_curve.py load_lean reads
(m, s, t3, d4, kA, lam, a, pred, gt for layers 1..15; loc, dm_tf and R are placeholders the corrector does not read),
the neighbour features (lc3_curve.nb_feats, the same function as round 3) and the raw extras.

Refuses mini 60-99 and full 304-351 (held out), full 48-95 (the round-2 report set) and anything above full 999.

    (Mac) .venv/bin/python research/r25a/gru_dump.py --est <folder> --tags full_0096,full_0097 --threads 1 --out research/r25a/gru/wcf
"""
from __future__ import annotations

import os
import sys


def _early_threads() -> None:
    k = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = k


_early_threads()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research"))
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
import lc_chain as lc  # noqa: E402
import lc_fit as lf  # noqa: E402
import lc3_curve as C  # noqa: E402
from estsha import sha16  # noqa: E402
from whestbench import BaseEstimator, SetupContext  # noqa: E402
from whestbench.domain import MLP  # noqa: E402

B = 2 ** 41
V_W11, V_W11T, V_PK1 = 2, 8, 17


def allowed(tag: str) -> bool:
    src, i = tag.split("_")
    i = int(i)
    if src == "mini":
        return 0 <= i <= 59
    return src == "full" and 0 <= i <= 999 and not 48 <= i <= 95 and not 304 <= i <= 351


def load_est(folder: Path):
    path = folder / "estimator.py"
    spec = importlib.util.spec_from_file_location("est_gru_dump", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    spec.loader.exec_module(mod)
    c = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, BaseEstimator) and v is not BaseEstimator]
    assert len(c) == 1
    return mod, c[0]


def yield_reason():
    """Why this dump should give the Mac back now (exit 75, research/r25a/gru_stream.sh retries the batch later), or None:
    with R25_YIELD_LOCKS set (coordinator 14:0x: timing work comes first), a quiet-window file or any job lock other than
    these dump streams' own (arc_r25a_gru_*, or R25_OWN_PREFIX: the S reads that come before this lane's own work use
    arc_r25a_, research/r25a/gru_sread.sh; comma-separated, research/r25a/gru_sread3.sh adds harbor_, whose jobs the
    shared memory gate research/mac_budget.sh already counts)."""
    if os.environ.get("R25_YIELD_LOCKS"):
        P = Path.home() / "nova-adsb" / "prcm"
        for f in sorted(P.iterdir()):
            n = f.name
            own = tuple(os.environ.get("R25_OWN_PREFIX", "arc_r25a_gru_").split(","))  # comma-separated prefixes
            if n.startswith("arc_quiet") or (n.endswith(".lock") and not n.startswith(own)):
                return f"other work on the Mac: {n}"
    return None


def weights_gt(tag):
    if tag.startswith("full_"):
        z = np.load(ROOT / "research" / "r2_data" / f"{tag}.npz")
        return z["W"], z["gt"]
    return C.weights(tag, with_gt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="folder estimator (estimator.py and wmsg.npz)")
    ap.add_argument("--tags", default="", help="comma list of mini_NNN / full_NNNN")
    ap.add_argument("--tagfile", default="", help="json list of tags")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t] + (json.loads(Path(a.tagfile).read_text()) if a.tagfile else [])
    bad = [t for t in tags if not allowed(t)]
    if bad:
        sys.exit(f"refused: {bad}")
    folder = Path(a.est).resolve()
    out = Path(a.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "feat").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    mod, cls = load_est(folder)
    orig = mod._wmsg
    cap = {}

    def hook(weights, VX, QB, SVB, P, f32, parts=None, *rest):  # research only: keep the corrector's inputs
        L = len(weights) - 1
        cap["pre15"] = np.array(VX[L, V_PK1], dtype=np.float32, copy=True)
        orig(weights, VX, QB, SVB, P, f32, parts, *rest)  # rest: the v2 corrector's QX
        cap["QB"], cap["SVB"], cap["VX"] = (np.array(x, dtype=np.float32, copy=True) for x in (QB, SVB, VX))

    mod._wmsg = hook
    if os.environ.get("R25_YIELD_LOCKS"):  # yield at once, not only between MLPs
        import threading

        def watcher():
            while True:
                time.sleep(5)
                why = yield_reason()
                if why:
                    print(json.dumps({"yield_now": why}), flush=True)
                    os._exit(75)  # the MLP in flight is lost (its cache is written last), nothing else is
        threading.Thread(target=watcher, daemon=True).start()
    est = cls()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1.0", submission_dir=str(folder), seed=0))
    esha = sha16(folder)
    stop = os.environ.get("R25_STOP")
    log = out / "log.jsonl"
    for tag in tags:
        if stop and os.path.exists(stop):
            print(json.dumps({"stopped_before": tag}), flush=True)
            break
        dst = out / "cache" / f"{tag}.npz"
        if dst.exists():
            continue
        why = yield_reason()
        if why:
            print(json.dumps({"yield_before": tag, "why": why}), flush=True)
            sys.exit(75)
        t0 = time.time()
        W, gt = weights_gt(tag)
        cap.clear()
        mlp = MLP(width=1024, depth=16, weights=[fnp.asarray(W[l]) for l in range(16)], seed=0, name=tag)
        if os.environ.get("R29_ORC_T"):  # R29-A oracle: this MLP's true means for the research folder's hook (make_r29orc.py)
            mod._R29_ORC = {"t": int(os.environ["R29_ORC_T"]), "gt": fnp.asarray(np.asarray(gt, dtype=np.float32))}
        with flops.BudgetContext(flop_budget=B, wall_time_limit_s=600.0, quiet=True) as ctx:
            pred = est.predict(mlp, B)
        t_pred = time.time() - t0
        P = np.asarray(pred, dtype=np.float32)
        QB, VX = cap["QB"].astype(np.float64), cap["VX"]
        pre15 = cap["pre15"]
        dp15 = P[15].astype(np.float64) - pre15.astype(np.float64)
        base = P.astype(np.float64).copy()
        base[15] = pre15
        assert np.array_equal(P[1:15], VX[1:15, V_PK1]), "layers 1..14: the prediction is the engine's readout"
        g64 = gt.astype(np.float64)
        rec = {k: [] for k in ("m", "s", "t3", "d4", "kA", "lam", "a", "pred", "gt", "loc", "dm_tf")}
        lam_spread = 0.0
        for L in range(1, 16):
            a_, var = QB[L, 7], QB[L, 8]
            s = np.sqrt(var)
            dA, lamv, t3 = QB[L, 3], QB[L, 4], QB[L, 6]
            lam_spread = max(lam_spread, float(np.ptp(lamv)))
            lam = float(lamv[0])
            d4 = lf.G * dA + lf.G * lam * var
            for k, v in (("m", a_ * s), ("s", s), ("t3", t3), ("d4", d4), ("kA", dA / s ** 4), ("lam", np.full_like(s, lam)),
                         ("a", a_), ("pred", base[L]), ("gt", g64[L]), ("loc", g64[L] - base[L]), ("dm_tf", np.zeros_like(s))):
                rec[k].append(v)
        z = {k: np.array(v) for k, v in rec.items()}
        z["R"] = np.zeros((3, len(lc.RICH), 1024), np.float32)  # round 1's covariance features: lc only, not the corrector
        z["VX"] = VX.astype(np.float32)
        z["final_mse"] = np.array(float(((P[15].astype(np.float64) - g64[15]) ** 2).mean()))
        z["base_final_mse"] = np.array(float(((pre15.astype(np.float64) - g64[15]) ** 2).mean()))
        z["flops"], z["op_calls"] = np.array(int(ctx.flops_used)), np.array(len(ctx.op_log))
        qxb_ = getattr(mod, "_QX", {}).get("buf") if isinstance(getattr(mod, "_QX", None), dict) else None
        if qxb_ is not None:  # stage 2: the corrector v2 taps of layers 1..15 (research/r25a/make_r25a_qx.py)
            z["qx"] = np.asarray(qxb_, dtype=np.float32)[1:16].copy()
        NB = C.nb_feats(W, z)
        np.savez(out / "feat" / f"{tag}.npz", NB=NB)
        np.savez(out / "raw" / f"{tag}.npz", QB=cap["QB"], SVB=cap["SVB"], pre15=pre15, dp15=dp15.astype(np.float32),
                 pred=P)
        np.savez(dst, **z)  # last: its presence marks the MLP done
        line = {"tag": tag, "estimator_sha16": esha, "final_mse": float(z["final_mse"]), "base_final_mse": float(z["base_final_mse"]),
                "flops": int(z["flops"]), "op_calls": int(z["op_calls"]), "lam_spread": lam_spread, "secs_pred": round(t_pred, 1),
                "secs": round(time.time() - t0, 1), "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2),
                "pred_sha16": hashlib.sha256(P.tobytes()).hexdigest()[:16], "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        del W, gt, mlp, pred, P, z, NB
    print("done", flush=True)


if __name__ == "__main__":
    main()
