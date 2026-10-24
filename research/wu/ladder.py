"""Write-up ladder for submission 336955 (coordinator 2026-10-20; plan in ARC MISSION G2 before any number): what the
corrector's chain and its input groups are worth.

Each variant retrains the single-model recipe of the shipped corrector's teacher members (research/r25a/gru_lean.py
train, unchanged: one GRU cell, hidden 32, Adam lr 2e-3, weight decay 1e-4, early stopping on mini 44-49, the shipped
taps 0,1,2,4-11) with 30 epochs instead of their 60, on the same 400-network pool (research/r25a/gru/qx3a15) with one
change, for each of three seeds:

  ref            none
  nochain        dp_t = u_t: no carried term in the output (the messages W^T dp / s and (W o W)^T |dp| / s stay inputs)
  nochain_nomsg  no carried term and both dp messages 0: only the hidden-state carry W^T H_{t-1} / sqrt 2 links layers
  w1             the carried term uses D_t = w_1 = Phi(a) only, instead of w_1 + w_4 t3 / 6 + w_5 d4 / 24
  notaps         no structure taps (the inputs of the corrector before R25)
  nomsg          the two dp messages set to 0 (the hidden-state carry kept)
  nocarry        the hidden-state carry set to 0 (the two dp messages kept)

`eval` reads every trained variant on the 90 validation networks (research/r25a/gru/v90_c5) and reports, per seed and
averaged over seeds, final-layer MSE ratios against `ref` of the same seed on the 84 full networks (the 6 mini ones
also served early stopping) and on all 90, with a bootstrap over networks.

    (Mac) .venv-lc3/bin/python -u research/wu/ladder.py train --variant nochain --seed 2 [--epochs 30]
    (Mac) .venv-lc3/bin/python -u research/wu/ladder.py eval [--variants ref,nochain,...] [--seeds 2,3,4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE.parent
sys.path.insert(0, str(RES))
sys.path.insert(0, str(RES / "r25a"))
import gru_lean as G  # noqa: E402
import lc3_curve as C  # noqa: E402
import lc3_wmsg as M  # noqa: E402

POOL = RES / "r25a" / "gru" / "qx3a15"
V90 = RES / "r25a" / "gru" / "v90_c5"
V90_TAGS = RES / "r25a" / "gru" / "tags_v90.json"
TAPS = "0,1,2,4,5,6,7,8,9,10,11"
VARIANTS = ("ref", "nochain", "nochain_nomsg", "w1", "notaps", "nomsg", "nocarry")
VAR = "ref"
_BASE_LOAD = None


class WMsgV(M.WMsg):
    """M.WMsg with the ladder's changes (same parameters, so state dicts are interchangeable)."""

    def forward(self, d, W):
        X = (d["X"] - self.mu) / self.sd
        s, D = d["s"], d["D"]
        n = X.shape[1]
        dp = torch.zeros(n)
        H = torch.zeros(n, self.k)
        outs = []
        for i in range(15):
            if i == 0:
                m1, m2, Hm, push = torch.zeros(n), torch.zeros(n), H, None
            else:
                Wl = W[i]
                both = Wl.T @ torch.cat([dp[:, None], H], 1)
                push = both[:, 0]
                Hm = both[:, 1:] / math.sqrt(2.0)
                m1 = push / s[i]
                m2 = ((Wl * Wl).T @ dp.abs()) / s[i]
                if VAR in ("nomsg", "nochain_nomsg"):
                    m1, m2 = torch.zeros_like(m1), torch.zeros_like(m2)
                if VAR == "nocarry":
                    Hm = torch.zeros_like(Hm)
            inp = torch.cat([X[i], 1e3 * m1[:, None], 1e3 * m2[:, None]], 1)
            H = self.cell(inp, Hm)
            u = 1e-4 * s[i] * self.head(H)[:, 0]
            dp = u if (push is None or VAR in ("nochain", "nochain_nomsg")) else D[i] * push + u
            outs.append(dp)
        return torch.stack(outs, 0)


def install(variant: str) -> None:
    """Set the variant, the model factory and a loader that, for w1 only, replaces D by Phi(m / s) from the same cache.
    gru_lean's optional yield watcher (R25_YIELD_LOCKS) stays off; the shared job wrapper waits for room and for quiet
    requests before a run starts."""
    global VAR, _BASE_LOAD
    assert variant in VARIANTS, variant
    VAR = variant
    G.make_model = lambda arch, f, hidden, mu, sd, proj=64: WMsgV(f, hidden, mu, sd)
    G.yield_watch = lambda: None
    if _BASE_LOAD is None:
        _BASE_LOAD = M.load_mlp

        def load_var(tag, nb):
            d = _BASE_LOAD(tag, nb)
            if VAR == "w1":
                z = np.load(C.cache_path(tag))
                a = torch.as_tensor(z["m"].astype(np.float64) / z["s"].astype(np.float64))
                d["D"] = torch.special.ndtr(a).numpy().astype(np.float32)
            return d
        M.load_mlp = load_var


def cmd_train(a) -> None:
    install(a.variant)
    argv = ["train", "--dir", str(POOL), "--tag", f"lad_{a.variant}_s{a.seed}", "--wd", "1e-4", "--epochs", str(a.epochs),
            "--seed", str(a.seed), "--no-s"]
    if a.variant != "notaps":
        argv += ["--qx", TAPS]
    sys.argv = ["gru_lean.py"] + argv
    print(json.dumps({"ladder": a.variant, "argv": argv, "start": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
    G.main()


def load_model(path: Path) -> WMsgV:
    raw = dict(np.load(path).items())
    if "qxsel" in raw:
        G.qx_on([int(v) for v in raw.pop("qxsel")], raw.pop("qxc"))
    else:
        G.qx_on([], np.ones((15, 12)))
    sd = {k: torch.as_tensor(v) for k, v in raw.items()}
    m = WMsgV(sd["mu"].shape[0], sd["head.weight"].shape[1], sd["mu"].numpy(), sd["sd"].numpy())
    m.load_state_dict(sd)
    m.eval()
    return m


def cmd_eval(a) -> None:
    tags = json.loads(V90_TAGS.read_text())
    G.point(V90)
    seeds = [int(x) for x in a.seeds.split(",")]
    res = {}  # res[seed][variant][tag] = corrected final-layer MSE
    for sd_ in seeds:
        for v in a.variants.split(","):
            p = POOL / f"lc3_wmsg_lad_{v}_s{sd_}.npz"
            if not p.exists():
                print(json.dumps({"variant": v, "seed": sd_, "missing": str(p)}), flush=True)
                continue
            install(v)
            model = load_model(p)
            with torch.no_grad():
                mse = M.evaluate(model, tags, True)
            res.setdefault(sd_, {})[v] = {t: float(x) for t, x in mse.items()}
            print(json.dumps({"variant": v, "seed": sd_, "n": len(mse), "mean_final_mse": float(np.mean(list(mse.values())))}),
                  flush=True)
    base = {t: float(C.load_lean(t, chain=True, nb=False, vx=False)["base_mse"]) for t in tags}  # engine alone
    rng = np.random.default_rng(12345)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": seeds, "base": base, "mse": res, "ratios": {}}
    variants = [v for v in a.variants.split(",") if v != "ref"]
    for subset, keep in (("full84", lambda t: t.startswith("full_")), ("all90", lambda t: True)):
        ks = [t for t in tags if keep(t)]
        J = rng.integers(0, len(ks), (10000, len(ks)))
        b0 = np.array([base[t] for t in ks])
        for v in variants + ["ref"]:
            got = [sd_ for sd_ in seeds if sd_ in res and v in res[sd_] and "ref" in res[sd_]]
            if not got:
                continue
            per_seed, boot = {}, np.zeros(len(J))
            for sd_ in got:
                x1 = np.array([res[sd_][v][t] for t in ks])
                x0 = np.array([res[sd_]["ref"][t] for t in ks])
                per_seed[sd_] = {"vs_ref": float(x1.mean() / x0.mean()), "vs_engine": float(x1.mean() / b0.mean()),
                                 "better_than_ref": int((x1 < x0).sum())}
                boot += x1[J].mean(1) / x0[J].mean(1)
            boot /= len(got)
            out["ratios"][f"{v}/{subset}"] = {
                "seeds": got, "n": len(ks), "per_seed": per_seed,
                "vs_ref_mean": float(np.mean([per_seed[s_]["vs_ref"] for s_ in got])),
                "ci95": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
                "vs_engine_mean": float(np.mean([per_seed[s_]["vs_engine"] for s_ in got]))}
    dst = HERE / f"ladder_{time.strftime('%m%d_%H%M')}.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"ratios": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in out["ratios"].items()},
                      "saved": str(dst)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", required=True, choices=VARIANTS)
    t.add_argument("--seed", type=int, default=2)
    t.add_argument("--epochs", type=int, default=30)
    e = sub.add_parser("eval")
    e.add_argument("--variants", default=",".join(VARIANTS))
    e.add_argument("--seeds", default="2,3,4")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    main()
