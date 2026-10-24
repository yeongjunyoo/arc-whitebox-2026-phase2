"""lc3: W-message recurrent corrector (research/2026-10-05-r2-competitors.md H1; the idea is from mliston's public
Phase 1 write-up prose, discourse 18170: a GRU corrector whose input includes the previous layer's correction carried
through W and W o W, trained downstream-aware. No participant code was read; the design and code here are ours).

Per layer L = 1..15 and neuron i, with the base engine's own quantities (research/r2_dump.py caches):
  messages  m1 = W_L^T dp_{L-1} / s_L,  m2 = (W_L o W_L)^T |dp_{L-1}| / s_L,  Hm = W_L^T H_{L-1} / sqrt(2)
  state     H_L = GRUCell([x_L, 1e3 m1, 1e3 m2], Hm)        (x_L: base features (+ neighbour features), depth)
  output    u_L = 1e-4 s_L head(H_L)
  chain     dp_L = D_L * (W_L^T dp_{L-1}) + u_L,  D = w11 + w41 t3 / 6 + w51 d4 / 24 (the lin chain of lc)
The final mean is pred_15 + dp_15. Loss (downstream-aware, free-running): sum_L w_L MSE_L(corrected) / MSE_L(base),
w_15 = 5, else 1. One MLP per step (its own W), Adam, early stopping on mini 44-49 by the final-layer ratio.

    .venv-lc3/bin/python research/lc3_wmsg.py train --n 120 --hidden 16 --epochs 30 --tag h16_n120
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
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc3_curve as C  # noqa: E402

torch.set_num_threads(int(os.environ.get("LC3_THREADS", "1")))
N = 1024
WDIR = HERE / "lc3_w"  # float32 W[1:16] of mini MLPs (cached once; full MLPs read research/r2_data/)


def layer_weights(tag):
    if tag.startswith("mini_"):
        p = WDIR / f"{tag}.npy"
        if not p.exists():
            WDIR.mkdir(exist_ok=True)
            np.save(p, C.weights(tag)[1:16])
        return np.load(p)
    return np.load(HERE / "r2_data" / f"{tag}.npz")["W"][1:16]


DROP = []  # neighbour-feature names left out (--drop-soffd: the one that costs an n^3 product per layer)


def load_mlp(tag, nb):
    c = C.load_lean(tag, chain=True, nb=nb, vx=False)
    z = np.load(C.cache_path(tag))
    X = [c["FB"]]
    if nb:
        X.append(c["NB"][:, [i for i, k in enumerate(C.NB) if k not in DROP]])
    X.append(np.broadcast_to((np.arange(1, 16, dtype=np.float32) / 15.0)[:, None, None], (15, 1, N)))
    X = np.concatenate(X, 1).transpose(0, 2, 1).copy()  # (15, n, F)
    pred, gt = z["pred"].astype(np.float64), z["gt"].astype(np.float64)
    return {"tag": tag, "X": X, "s": z["s"].astype(np.float32), "D": c["D"].astype(np.float32),
            "res": (gt - pred).astype(np.float32), "base_mse": ((gt - pred) ** 2).mean(1)}


class WMsg(torch.nn.Module):
    def __init__(self, f, hidden, mu, sd, msg=True):
        super().__init__()
        self.msg = msg
        self.mu = torch.nn.Parameter(torch.as_tensor(mu), requires_grad=False)
        self.sd = torch.nn.Parameter(torch.as_tensor(sd), requires_grad=False)
        self.cell = torch.nn.GRUCell(f + 2, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.k = hidden

    def forward(self, d, W):
        """d: one MLP's tensors; W: (15, n, n). Returns dp per layer (15, n)."""
        X = (d["X"] - self.mu) / self.sd
        s, D = d["s"], d["D"]
        dp = torch.zeros(N)
        H = torch.zeros(N, self.k)
        outs = []
        for i in range(15):
            if i == 0:
                m1 = torch.zeros(N)
                m2 = torch.zeros(N)
                Hm = H
                push = None
            else:
                Wl = W[i]
                both = Wl.T @ torch.cat([dp[:, None], H], 1)
                push = both[:, 0]
                Hm = both[:, 1:] / math.sqrt(2.0)
                m1 = push / s[i]
                m2 = ((Wl * Wl).T @ dp.abs()) / s[i]
                if not self.msg:  # ablation: no message of any kind (the chain itself still carries dp)
                    m1, m2, Hm = torch.zeros_like(m1), torch.zeros_like(m2), torch.zeros_like(Hm)
            inp = torch.cat([X[i], 1e3 * m1[:, None], 1e3 * m2[:, None]], 1)
            H = self.cell(inp, Hm)
            u = 1e-4 * s[i] * self.head(H)[:, 0]
            dp = u if push is None else D[i] * push + u
            outs.append(dp)
        return torch.stack(outs, 0)


def to_t(d):
    return {k: torch.as_tensor(v) for k, v in d.items() if k in ("X", "s", "D", "res")}


def loss_of(out, d, base_mse_t, w):
    err = ((d["res"] - out) ** 2).mean(1)  # (15,) corrected MSE per layer
    return (w * err / base_mse_t).sum() / w.sum(), err


def evaluate(model, tags, nb, cache=None):
    """Final-layer MSE per MLP of the corrected chain."""
    out = {}
    model.eval()
    with torch.no_grad():
        for t in tags:
            d = cache[t] if cache is not None and t in cache else load_mlp(t, nb)
            W = torch.as_tensor(layer_weights(t))
            o = model(to_t(d), W)
            res = d["res"][14].astype(np.float64)
            out[t] = float(((res - o[14].numpy().astype(np.float64)) ** 2).mean())
            del W
    model.train()
    return out


def cmd_train(args):
    t0 = time.time()
    if args.drop_soffd:
        DROP.append("soffd")
    torch.manual_seed(args.seed)
    nb = not args.no_nb
    pool = C.POOL0 + C.pool1()
    if nb:
        pool = [t for t in pool if (C.FEAT / f"{t}.npz").exists()]
    order = list(np.random.default_rng(1000 + args.seed).permutation(pool))
    train_tags = order[: args.n]
    data = {t: load_mlp(t, nb) for t in train_tags}
    val = {t: load_mlp(t, nb) for t in C.VAL_TAGS}
    Xall = np.concatenate([d["X"].reshape(-1, d["X"].shape[-1]) for d in data.values()], 0)
    mu, sd = Xall.mean(0).astype(np.float32), (Xall.std(0) + 1e-6).astype(np.float32)
    del Xall
    f = next(iter(data.values()))["X"].shape[-1]
    model = WMsg(f, args.hidden, mu, sd, msg=not args.no_msg)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    w = torch.ones(15)
    w[14] = 5.0
    base_val = {t: float(v["base_mse"][14]) for t, v in val.items()}
    best, best_state, hist, bad = np.inf, None, [], 0
    print(f"train {len(train_tags)} features {f} hidden {args.hidden} params "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} loaded {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        tl = []
        for t in rng.permutation(train_tags):
            d = data[t]
            W = torch.as_tensor(layer_weights(t))
            out = model(to_t(d), W)
            loss, _ = loss_of(out, to_t(d), torch.as_tensor(d["base_mse"], dtype=torch.float32), w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
            del W
        vm = evaluate(model, C.VAL_TAGS, nb, val)
        vr = sum(vm.values()) / sum(base_val.values())
        hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_final_ratio_vs_base": vr,
                     "secs": round(time.time() - t0)})
        print(json.dumps(hist[-1]), flush=True)
        if vr < best - 1e-5:
            best, bad = vr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    del data
    base, lcr1, lc2k = C.yardsticks(C.S_TAGS)
    res = evaluate(model, C.S_TAGS, nb)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "train": train_tags, "val": C.VAL_TAGS,
           "hist": hist, "best_val_ratio_vs_base": best,
           "S_vs_base": C.stats(res, base, C.S_TAGS), "S_vs_lc": C.stats(res, lcr1, C.S_TAGS),
           "S_vs_lc2k": C.stats(res, lc2k, C.S_TAGS), "mse_S": res, "secs": round(time.time() - t0)}
    tag = args.tag or f"h{args.hidden}_n{args.n}"
    (HERE / f"lc3_wmsg_{tag}_{time.strftime('%m%d_%H%M')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez(HERE / f"lc3_wmsg_{tag}.npz", **{k: v.numpy() for k, v in model.state_dict().items()})
    print(f"S vs base {out['S_vs_base']['ratio_of_means']:.4f} vs lc {out['S_vs_lc']['ratio_of_means']:.4f} "
          f"{out['S_vs_lc']['ratio_of_means_ci95']} vs lc2k {out['S_vs_lc2k']['ratio_of_means']:.4f} "
          f"better than lc {out['S_vs_lc']['better_on']}/48  {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-nb", action="store_true")
    ap.add_argument("--no-msg", action="store_true")
    ap.add_argument("--drop-soffd", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    {"train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
