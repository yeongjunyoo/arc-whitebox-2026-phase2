"""R25-A GRU retrain, steps 2-3 (Nova 2026-10-14): the round-3 lean trainer (research/lc3_wmsg.py, its WMsg, load_mlp,
loss and evaluate, imported unchanged) pointed at the caches research/r25a/gru_dump.py wrote from one folder estimator.

  check  the shipped corrector (research/lc3_wmsg_h32_nb_nosoffd_n400_s1p12.npz, the arrays behind every build's wmsg.npz)
         run by the lean trainer on the new caches must give the dp15 the estimator itself added (raw/<tag>.npz): this
         is the proof that the cache and the features are the corrector's inputs.
  train  the same model and settings as h32_nb_nosoffd_n400_s1p12 (hidden 32, neighbour features without soffd, lr 2e-3,
         30 epochs, patience 12, seed 1, early stopping on mini 44-49) on a fixed tag list (default: the 400 tags of
         that run), then the final-layer MSE on S (full 0-47, out of sample) against the engine's base and against the
         shipped corrector on the same engine (the dump's own final mean).
  eval   a trained npz on any dumped tags (lean chain), against base and the shipped corrector.

    (Mac) .venv-lc3/bin/python research/r25a/gru_lean.py check --dir research/r25a/gru/wcf --tags full_0096
    (Mac) .venv-lc3/bin/python research/r25a/gru_lean.py train --dir research/r25a/gru/wcf --tag wcf_h32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent
sys.path.insert(0, str(RES))
import lc3_curve as C  # noqa: E402
import lc3_wmsg as M  # noqa: E402

OLD = RES / "lc3_wmsg_h32_nb_nosoffd_n400_s1p12.npz"
OLD_JSON = RES / "lc3_wmsg_h32_nb_nosoffd_n400_s1p12_1005_1638.json"
S_TAGS = [f"full_{i:04d}" for i in range(48)]
QXSEL = []  # corrector v2: tap indices appended to the corrector input (empty: the shipped inputs)
QXC = None  # (15, 4) per-layer scales of the taps
_LOAD = None


def qx_on(sel, qxc):
    """Append arcsinh(QX / c) for the taps in sel to every MLP the lean chain loads (M.load_mlp is wrapped once)."""
    global _LOAD, QXC
    QXSEL[:] = list(sel)
    QXC = np.asarray(qxc, dtype=np.float64)
    if _LOAD is None:
        _LOAD = M.load_mlp

        def load_mlp_qx(tag, nb):
            d = _LOAD(tag, nb)
            if QXSEL:
                qx = np.load(C.cache_path(tag))["qx"].astype(np.float64)  # (15, 4, n)
                q = np.arcsinh(qx[:, QXSEL, :] / QXC[:, QXSEL, None]).astype(np.float32)
                d["X"] = np.concatenate([d["X"], q.transpose(0, 2, 1)], 2)
            return d
        M.load_mlp = load_mlp_qx


def qx_scales(tags):
    """Per-layer median of |QX| over these dumps (zeros left out), for every tap row the dumps have."""
    K = np.load(C.cache_path(tags[0]))["qx"].shape[1]
    acc = [[[] for _ in range(K)] for _ in range(15)]
    for t in tags:
        qx = np.load(C.cache_path(t))["qx"].astype(np.float64)
        for L in range(15):
            for k in range(K):
                v = np.abs(qx[L, k])
                v = v[v > 0]
                if v.size:
                    acc[L][k].append(np.median(v))
    return np.array([[float(np.median(acc[L][k])) if acc[L][k] else 1.0 for k in range(K)] for L in range(15)])


def point(d: Path):
    """Point lc3_curve's cache and feature lookups (read at call time by load_lean and lc3_wmsg.load_mlp) at d."""
    def cache_path(tag, report=False):
        C.check_tag(tag, report)
        return d / "cache" / f"{tag}.npz"
    C.cache_path = cache_path
    C.FEAT = d / "feat"
    M.DROP[:] = ["soffd"]


class WMsg2(torch.nn.Module):
    """Corrector v3 (c): two stacked GRU cells (research/r25a/_patch_lean_v3.py)."""

    def __init__(self, f, hidden, mu, sd):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(mu), requires_grad=False)
        self.sd = torch.nn.Parameter(torch.as_tensor(sd), requires_grad=False)
        self.cell = torch.nn.GRUCell(f + 2, hidden)
        self.cell2 = torch.nn.GRUCell(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.k = hidden

    def forward(self, d, W):
        import math as _m
        X = (d["X"] - self.mu) / self.sd
        s, D = d["s"], d["D"]
        n = X.shape[1]
        dp = torch.zeros(n)
        H1 = torch.zeros(n, self.k)
        H2 = torch.zeros(n, self.k)
        outs = []
        for i in range(15):
            if i == 0:
                m1 = torch.zeros(n)
                m2 = torch.zeros(n)
                Hm1, Hm2, push = H1, H2, None
            else:
                Wl = W[i]
                both = Wl.T @ torch.cat([dp[:, None], H1, H2], 1)
                push = both[:, 0]
                Hm1 = both[:, 1:self.k + 1] / _m.sqrt(2.0)
                Hm2 = both[:, self.k + 1:] / _m.sqrt(2.0)
                m1 = push / s[i]
                m2 = ((Wl * Wl).T @ dp.abs()) / s[i]
            inp = torch.cat([X[i], 1e3 * m1[:, None], 1e3 * m2[:, None]], 1)
            H1 = self.cell(inp, Hm1)
            H2 = self.cell2(H1, Hm2)
            u = 1e-4 * s[i] * self.head(H2)[:, 0]
            dp = u if push is None else D[i] * push + u
            outs.append(dp)
        return torch.stack(outs, 0)


class WMsgP(torch.nn.Module):
    """Corrector v3 (d): a wider input projection tanh(proj([x, 1e3 m1, 1e3 m2])) before the GRU cell."""

    def __init__(self, f, hidden, mu, sd, proj=64):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(mu), requires_grad=False)
        self.sd = torch.nn.Parameter(torch.as_tensor(sd), requires_grad=False)
        self.proj = torch.nn.Linear(f + 2, proj)
        self.cell = torch.nn.GRUCell(proj, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.k = hidden

    def forward(self, d, W):
        import math as _m
        X = (d["X"] - self.mu) / self.sd
        s, D = d["s"], d["D"]
        n = X.shape[1]
        dp = torch.zeros(n)
        H = torch.zeros(n, self.k)
        outs = []
        for i in range(15):
            if i == 0:
                m1 = torch.zeros(n)
                m2 = torch.zeros(n)
                Hm, push = H, None
            else:
                Wl = W[i]
                both = Wl.T @ torch.cat([dp[:, None], H], 1)
                push = both[:, 0]
                Hm = both[:, 1:] / _m.sqrt(2.0)
                m1 = push / s[i]
                m2 = ((Wl * Wl).T @ dp.abs()) / s[i]
            inp = torch.tanh(self.proj(torch.cat([X[i], 1e3 * m1[:, None], 1e3 * m2[:, None]], 1)))
            H = self.cell(inp, Hm)
            u = 1e-4 * s[i] * self.head(H)[:, 0]
            dp = u if push is None else D[i] * push + u
            outs.append(dp)
        return torch.stack(outs, 0)


def make_model(arch, f, hidden, mu, sd, proj=64):
    if arch == "gru2":
        return WMsg2(f, hidden, mu, sd)
    if arch == "proj":
        return WMsgP(f, hidden, mu, sd, proj)
    return M.WMsg(f, hidden, mu, sd)


def model_from(npz: Path) -> M.WMsg:
    raw = dict(np.load(npz).items())
    if "qxsel" in raw:  # corrector v2: its taps and scales come with it
        qx_on([int(v) for v in raw.pop("qxsel")], raw.pop("qxc"))
    elif QXSEL:
        qx_on([], QXC if QXC is not None else np.ones((15, 4)))
    sd = {k: torch.as_tensor(v) for k, v in raw.items()}
    f = sd["mu"].shape[0]
    hidden = sd["head.weight"].shape[1]
    if "cell2.weight_ih" in sd:
        m = WMsg2(f, hidden, sd["mu"].numpy(), sd["sd"].numpy())
    elif "proj.weight" in sd:
        m = WMsgP(f, hidden, sd["mu"].numpy(), sd["sd"].numpy(), sd["proj.weight"].shape[0])
    else:
        m = M.WMsg(f, hidden, sd["mu"].numpy(), sd["sd"].numpy())
    m.load_state_dict(sd)
    m.eval()
    return m


def lean_final(model, tags):
    """Final-layer corrected mean per tag from the lean chain (pred + dp15)."""
    out = {}
    with torch.no_grad():
        for t in tags:
            d = M.load_mlp(t, True)
            W = torch.as_tensor(M.layer_weights(t))
            o = model(M.to_t(d), W)
            z = np.load(C.cache_path(t))
            out[t] = (z["pred"][14].astype(np.float64) + o[14].numpy().astype(np.float64), z)
    return out


def cmd_check(a):
    d = Path(a.dir)
    point(d)
    model = model_from(OLD)
    rows = []
    for t, (fin, z) in lean_final(model, a.tags.split(",")).items():
        raw = np.load(d / "raw" / f"{t}.npz")
        dp_est = raw["dp15"].astype(np.float64)
        dp_lean = fin - z["pred"][14].astype(np.float64)
        gt = z["gt"][14].astype(np.float64)
        mse_est = float(((z["pred"][14] + dp_est - gt) ** 2).mean())
        mse_lean = float(((fin - gt) ** 2).mean())
        rel = float(np.abs(dp_lean - dp_est).max() / (np.abs(dp_est).max() + 1e-30))
        rows.append({"tag": t, "dp_max_abs": float(np.abs(dp_est).max()), "dp_max_rel_diff": rel,
                     "dp_corr": float(np.corrcoef(dp_lean, dp_est)[0, 1]), "mse_est": mse_est, "mse_lean": mse_lean,
                     "mse_ratio_lean_over_est": mse_lean / mse_est})
        print(json.dumps(rows[-1]), flush=True)
    (d / f"check_{time.strftime('%m%d_%H%M')}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def stats(new, ref):
    ks = sorted(set(new) & set(ref))
    if len(ks) < 2:
        return {"n": len(ks)}
    r = np.array([new[k] / ref[k] for k in ks])
    rng = np.random.default_rng(0)
    nm, rm = np.array([new[k] for k in ks]), np.array([ref[k] for k in ks])
    bs = [nm[i].mean() / rm[i].mean() for i in (rng.integers(0, len(ks), len(ks)) for _ in range(20000))]
    return {"n": len(ks), "ratio_of_means": float(nm.mean() / rm.mean()), "boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "mean_ratio": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else None,
            "better_on": int((r < 1).sum())}


def evaluate_against(model, tags, d: Path):
    """Final-layer MSE per tag: new corrector (lean), engine base (pre15) and the shipped corrector (the dump's own)."""
    new, base, shipped = {}, {}, {}
    for t, (fin, z) in lean_final(model, tags).items():
        gt = z["gt"][14].astype(np.float64)
        new[t] = float(((fin - gt) ** 2).mean())
        base[t] = float(z["base_final_mse"])
        shipped[t] = float(z["final_mse"])
    return new, base, shipped


def cmd_train(a):
    t0 = time.time()
    d = Path(a.dir)
    point(d)
    torch.manual_seed(a.seed)
    train_tags = json.loads(Path(a.tagfile).read_text()) if a.tagfile else json.loads(OLD_JSON.read_text())["train"]
    train_tags = [t for t in train_tags if (d / "cache" / f"{t}.npz").exists() and (d / "feat" / f"{t}.npz").exists()]
    if a.exclude:
        ex = set(a.exclude.split(","))
        train_tags = [t for t in train_tags if t not in ex]
    if a.qx:  # corrector v2: per-layer tap scales from the training dumps only
        qx_on([int(v) for v in (a.qx.split(",") if "," in a.qx else a.qx)], qx_scales(train_tags))  # "0124" or "0,1,2,10"
        print(json.dumps({"qx": a.qx, "qxc": QXC.tolist()}), flush=True)
    data = {t: M.load_mlp(t, True) for t in train_tags}
    if a.teacher:  # R27-A distillation: the target mixes the truth with the teacher ensemble's dp (base_mse stays the truth's)
        td = Path(a.teacher)
        for t in train_tags:
            tdp = np.load(td / f"{t}.npz")["dp"].astype(np.float32)
            data[t]["res"] = ((1.0 - a.beta) * data[t]["res"] + a.beta * tdp).astype(np.float32)
        print(json.dumps({"teacher": str(td), "beta": a.beta}), flush=True)
    val_tags = [t for t in C.VAL_TAGS if (d / "cache" / f"{t}.npz").exists()]
    assert val_tags, "no validation dump"
    val = {t: M.load_mlp(t, True) for t in val_tags}
    Xall = np.concatenate([v["X"].reshape(-1, v["X"].shape[-1]) for v in data.values()], 0)
    mu, sd = Xall.mean(0).astype(np.float32), (Xall.std(0) + 1e-6).astype(np.float32)
    del Xall
    f = next(iter(data.values()))["X"].shape[-1]
    model = make_model(a.arch, f, a.hidden, mu, sd, a.proj)
    if a.init_old:  # warm start from the shipped corrector (same feature count), its mu/sd kept
        old = model_from(OLD)
        model.load_state_dict(old.state_dict())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    w = torch.ones(15)
    w[14] = a.w_last  # the last layer's weight in the loss (the build adds only its dp)
    b14 = {t: float(v["base_mse"][14]) for t, v in data.items()}  # R29-A --abs-weight: the score pools absolute MSE
    m14 = float(np.mean(list(b14.values())))
    tail = []  # R29-A --swa K: the last K epochs' weights, averaged at the end
    base_val = {t: float(v["base_mse"][14]) for t, v in val.items()}
    best, best_state, hist, bad = np.inf, None, [], 0
    print(f"train {len(train_tags)} features {f} hidden {a.hidden} loaded {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        if a.decay_every and ep and ep % a.decay_every == 0:  # step decay (the longer-trained variant): lr x 0.5
            for g in opt.param_groups:
                g["lr"] *= 0.5
        tl = []
        for t in rng.permutation(train_tags):
            dd = data[t]
            W = torch.as_tensor(M.layer_weights(t))
            out = model(M.to_t(dd), W)
            loss, _ = M.loss_of(out, M.to_t(dd), torch.as_tensor(dd["base_mse"], dtype=torch.float32), w)
            if a.abs_weight:  # each MLP weighed by its engine error, as the pooled score weighs it
                loss = loss * (b14[t] / m14)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
            del W
        vm = M.evaluate(model, val_tags, True, val)
        vr = sum(vm.values()) / sum(base_val.values())
        hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_final_ratio_vs_base": vr, "secs": round(time.time() - t0)})
        print(json.dumps(hist[-1]), flush=True)
        if a.swa:
            tail = (tail + [{k: v.detach().clone() for k, v in model.state_dict().items()}])[-a.swa:]
        if vr < best - 1e-5:
            best, bad = vr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    swa_val = None
    if a.swa and tail:  # the tail average replaces the best epoch; both validation ratios are kept in the record
        model.load_state_dict({k: sum(x[k] for x in tail) / len(tail) for k in tail[0]})
        sv = M.evaluate(model, val_tags, True, val)
        swa_val = sum(sv.values()) / sum(base_val.values())
        print(json.dumps({"swa_epochs": len(tail), "swa_val_ratio_vs_base": swa_val, "best_val_ratio_vs_base": best}), flush=True)
    model.eval()
    del data
    s_tags = [] if a.no_s else [t for t in S_TAGS if (d / "cache" / f"{t}.npz").exists()]  # --no-s: S is not touched
    new, base, shipped = evaluate_against(model, s_tags, d)
    vnew, vbase, vship = evaluate_against(model, val_tags, d)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "dir": str(d), "train": train_tags, "val": val_tags,
           "hist": hist, "best_val_ratio_vs_base": best, "swa_val_ratio_vs_base": swa_val,
           "S_new_vs_base": stats(new, base), "S_new_vs_shipped": stats(new, shipped), "S_shipped_vs_base": stats(shipped, base),
           "VAL_new_vs_shipped": stats(vnew, vship), "mse_S": {"new": new, "base": base, "shipped": shipped},
           "secs": round(time.time() - t0)}
    (d / f"train_{a.tag}_{time.strftime('%m%d_%H%M')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    extra = {"qxsel": np.array(QXSEL, dtype=np.int64), "qxc": QXC} if QXSEL else {}
    np.savez(d / f"lc3_wmsg_{a.tag}.npz", **{k: v.numpy() for k, v in model.state_dict().items()}, **extra)
    print(json.dumps({k: out[k] for k in ("best_val_ratio_vs_base", "S_new_vs_base", "S_new_vs_shipped", "S_shipped_vs_base",
                                         "VAL_new_vs_shipped", "secs")}), flush=True)


def cmd_eval(a):
    d = Path(a.dir)
    point(d)
    model = model_from(Path(a.model))
    new, base, shipped = evaluate_against(model, a.tags.split(","), d)
    print(json.dumps({"new_vs_base": stats(new, base), "new_vs_shipped": stats(new, shipped),
                      "per_tag": {t: [new[t], shipped[t], base[t]] for t in new}}), flush=True)


def cmd_ens(a):
    """Seed ensemble on dumped tags (default S): each member's final-layer MSE and the MSE of the members' mean dp15
    (each member runs its own recurrence; only dp15 is averaged), against the shipped corrector on the same engine (the
    dump's own final mean) and against the first member."""
    d = Path(a.dir)
    point(d)
    tags = a.tags.split(",") if a.tags else [t for t in S_TAGS if (d / "cache" / f"{t}.npz").exists()]
    paths = a.models.split(",")
    fins = [lean_final(model_from(Path(p)), tags) for p in paths]
    gt = {t: fins[0][t][1]["gt"][14].astype(np.float64) for t in tags}
    p14 = {t: fins[0][t][1]["pred"][14].astype(np.float64) for t in tags}
    shipped = {t: float(fins[0][t][1]["final_mse"]) for t in tags}
    mse = lambda x, t: float(((x - gt[t]) ** 2).mean())  # noqa: E731
    member = [{t: mse(f[t][0], t) for t in tags} for f in fins]
    out = {"tags": len(tags), "models": paths, "members_vs_shipped": [stats(m, shipped) for m in member]}
    for k in range(2, len(fins) + 1):
        ens = {t: mse(p14[t] + np.mean([f[t][0] - p14[t] for f in fins[:k]], 0), t) for t in tags}
        out[f"ens{k}_vs_shipped"] = stats(ens, shipped)
        out[f"ens{k}_vs_member1"] = stats(ens, member[0])
    (d / f"ens_{a.tag}_{time.strftime('%m%d_%H%M')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "models"}), flush=True)


def yield_watch():
    """R25_YIELD_LOCKS (coordinator: timing work on the Mac comes first): exit 75 within 5 s when a quiet-window file or a
    job lock other than this lane's own GRU locks (arc_r25a_gru_*) appears; the runner retries the job later."""
    if not os.environ.get("R25_YIELD_LOCKS"):
        return
    import threading
    P = Path.home() / "nova-adsb" / "prcm"

    def watcher():
        while True:
            time.sleep(5)
            for f in sorted(P.iterdir()):
                n = f.name
                if n.startswith("arc_quiet") or (n.endswith(".lock") and not n.startswith("arc_r25a_gru_")):
                    print(json.dumps({"yield_now": n}), flush=True)
                    os._exit(75)
    threading.Thread(target=watcher, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "train", "eval", "ens"])
    ap.add_argument("--models", default="")
    ap.add_argument("--decay-every", type=int, default=0)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--tagfile", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay (taps v4 training variants)")
    ap.add_argument("--w-last", type=float, default=5.0, help="loss weight of the last layer (the others 1)")
    ap.add_argument("--teacher", default="", help="R27-A: dir of <tag>.npz with the teacher's dp (15, n) per training MLP")
    ap.add_argument("--beta", type=float, default=1.0, help="R27-A: the teacher's share of the training target")
    ap.add_argument("--no-s", action="store_true", help="R27-A: skip the S evaluation at the end of training")
    ap.add_argument("--abs-weight", action="store_true", help="R29-A: weigh each MLP's loss by its final-layer engine MSE")
    ap.add_argument("--swa", type=int, default=0, help="R29-A: keep the average of the last K epochs' weights")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-old", action="store_true")
    ap.add_argument("--tag", default="wcf_h32")
    ap.add_argument("--qx", default="", help="corrector v2: taps to append (e.g. 012, 0123); needs dumps with qx")
    ap.add_argument("--arch", default="gru", choices=["gru", "gru2", "proj"], help="corrector v3 architecture")
    ap.add_argument("--proj", type=int, default=64, help="--arch proj: width of the input projection")
    a = ap.parse_args()
    yield_watch()
    {"check": cmd_check, "train": cmd_train, "eval": cmd_eval, "ens": cmd_ens}[a.cmd](a)


if __name__ == "__main__":
    main()
