"""Check estimators/k3_simple.py against ARC's reference (alignment-research-center/mlp_cumulant_propagation,
commit 93d091a, MIT) on random He-init MLPs (n = 8/32/64/128, depth 4-16) and on public mini MLPs at n = 1024.

Two halves, two interpreters (the reference needs torch, the port needs flopscope):

    # 1) reference side: writes research/k3_check_data/ref_<case>.npz
    cd third_party/mlp_cumulant_propagation
    set PYTHONPATH=src
    python <vault>/scripts/memcap.py --gb 4 -- ..\\.venv-ref\\Scripts\\python.exe ..\\..\\research\\k3_port_check.py ref --cases small
    python <vault>/scripts/memcap.py --gb 6 -- ..\\.venv-ref\\Scripts\\python.exe ..\\..\\research\\k3_port_check.py ref --cases mini0

    # 2) port side: reads those files, runs the estimator (float64 and float32), prints the differences
    cd whest-starterkit
    python <vault>/scripts/memcap.py --gb 4 -- .venv\\Scripts\\python.exe ..\\research\\k3_port_check.py compare --cases small,mini0

The reference side replays mlp_kprop's own layer loop (coerce_input, linear_kprop with set_metric=init_scale[l],
nonlin_kprop(factor=True, kind=SIMPLE)) but keeps only the current tower: mlp_kprop(output_all=True) clones every
layer's K3 factors, whose rank grows by 2n per layer (about 13 GB at n = 1024 in float64). On small cases it also
runs mlp_kprop(output_all=True) and asserts the replay is identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "k3_check_data"
MINI = Path(os.environ.get("ARC_MINI_DIR") or (Path.home() / ".cache" / "huggingface" / "hub" / "datasets--aicrowd--arc-whestbench-public-2026"
                                                / "snapshots" / "aa99830fdc09fad15407b10e8e3459d3e18bba0a" / "prepared" / "mini"))  # elsewhere: set ARC_MINI_DIR
# (name, n, depth, seed). tiny8 also compares the full third-cumulant tensor.
SMALL = [("tiny8", 8, 4, 11), ("n32d4", 32, 4, 1), ("n32d16", 32, 16, 2), ("n64d8", 64, 8, 3),
         ("n64d16", 64, 16, 4), ("n128d4", 128, 4, 5), ("n128d16", 128, 16, 6)]


def he_weights(n: int, depth: int, seed: int) -> np.ndarray:
    """(depth, n, n) float32, stored (input, output) like the competition, N(0, 2/n)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((depth, n, n)) * np.sqrt(2.0 / n)).astype(np.float32)


def load_mini(mlp_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights (16, 1024, 1024) float32 and ground-truth layer means (16, 1024) of one public mini MLP."""
    import pyarrow as pa

    def flat(cell, shape):
        if isinstance(cell, pa.ExtensionScalar):  # HF Array2D/3D types are registered once `datasets` is imported
            cell = cell.value
        arr = cell.values
        while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            arr = arr.flatten()
        return arr.to_numpy(zero_copy_only=False).astype(np.float32).reshape(shape)

    for f in sorted(MINI.glob("data-*.arrow")):
        with pa.memory_map(str(f)) as src:
            t = pa.ipc.open_stream(src).read_all()
        ids = t.column("mlp_id").to_pylist()
        if mlp_id in ids:
            i = ids.index(mlp_id)
            return flat(t.column("weights")[i], (16, 1024, 1024)), flat(t.column("all_layer_means")[i], (16, 1024))
    raise KeyError(mlp_id)


def cases(spec: str):
    out = []
    for c in spec.split(","):
        if c == "small":
            out += [(name, ("he", n, d, s)) for name, n, d, s in SMALL]
        elif c.startswith("mini"):
            out.append((c, ("mini", int(c[4:]))))
        else:
            out += [(name, ("he", n, d, s)) for name, n, d, s in SMALL if name == c]
    return out


def get_weights(src) -> tuple[np.ndarray, np.ndarray | None]:
    if src[0] == "he":
        return he_weights(*src[1:]), None
    return load_mini(src[1])


# ------------------------------------------------------------------ reference side (torch)
def ref_run(W: np.ndarray, internals: bool, full_k3: bool, save_path: Path | None = None):
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, coerce_input, linear_kprop, mlp_kprop, nonlin_kprop
    from mlp_kprop.mlp import MLP
    from mlp_kprop.wick import WICK_COEF_D

    torch.set_grad_enabled(False)
    dt = torch.float64
    depth, n, _ = W.shape
    mlp = MLP(hidden_dim=n, num_layers=depth + 1).to(dtype=dt)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=dt).contiguous()
    K_in = {1: torch.zeros(n, dtype=dt), 2: torch.eye(n, dtype=dt)}
    rec: dict[str, list] = {k: [] for k in ("means", "act_var", "c4")}
    if internals:
        for k in ("pre_mean", "pre_var", "t3", "T21", "c4_pre", "act_cov", "K3_3", "K3_21", "K3_full"):
            rec[k] = []
    K = coerce_input(K_in, k_max=3, kind=SIMPLE)
    t0 = time.time()
    for l in range(depth):
        Wl = mlp.Ws[l].weight
        K = linear_kprop(K, Wl, k_max=3, set_metric=mlp.init_scale[l], bias=mlp.Ws[l].bias)
        if internals:
            rec["pre_mean"].append(K[1].core.numpy().copy())
            rec["pre_var"].append(K[2].core.diag().numpy().copy())
            if 3 in K:
                rec["t3"].append(K[3].clone().get_dslice((3,)).numpy().copy())
                rec["T21"].append(K[3].clone().get_dslice((2, 1)).numpy().copy())
            else:
                rec["t3"].append(np.zeros(n))
                rec["T21"].append(np.zeros((n, n)))
            rec["c4_pre"].append(float(K[4].core) if 4 in K else 0.0)
        K = nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D[mlp.nonlin_names[l]], k_max=3, kind=SIMPLE, use_pK=True,
                         factor=True)
        rec["means"].append(K[1].to_tensor().numpy().copy())
        rec["act_var"].append(K[2].core.diag().numpy().copy())
        rec["c4"].append(float(K[4].core))
        if internals:
            rec["act_cov"].append(K[2].core.numpy().copy())
            rec["K3_3"].append(K[3].clone().get_dslice((3,)).numpy().copy())
            rec["K3_21"].append(K[3].clone().get_dslice((2, 1)).numpy().copy())
            if full_k3:
                rec["K3_full"].append(K[3].to_tensor().numpy().copy())
        print(f"  ref layer {l:2d}  {time.time() - t0:7.1f}s  K3 rank {K[3]._factors[0].shape[1]}", flush=True)
        if save_path is not None:  # partial results survive a memory-cap kill
            np.savez(save_path, **{k: np.array(v) for k, v in rec.items() if v}, layers_done=np.array(l + 1))
    out = {k: np.array(v) for k, v in rec.items() if v}
    if internals:  # the replay must equal the library entry point exactly
        Kl = mlp_kprop(mlp, K_in, k_max=3, kind=SIMPLE, factor=True, output_all=True, up_to_layer=f"act{depth - 1}")
        lib = np.stack([Kl[f"act{l}"][1].to_tensor().numpy() for l in range(depth)])
        out["replay_vs_mlp_kprop_maxabs"] = np.array(float(np.abs(lib - out["means"]).max()))
    return out


def cmd_ref(args):
    DATA.mkdir(exist_ok=True)
    for name, src in cases(args.cases):
        W, gt = get_weights(src)
        small = W.shape[1] <= 128
        print(f"[ref] {name}: n={W.shape[1]} depth={W.shape[0]}", flush=True)
        t0 = time.time()
        out = ref_run(W, internals=small, full_k3=W.shape[1] <= 16,
                      save_path=None if small else DATA / f"ref_{name}.partial.npz")
        out["secs"] = np.array(time.time() - t0)
        out["layers_done"] = np.array(W.shape[0])
        if gt is not None:
            out["gt"] = gt.astype(np.float64)
        np.savez(DATA / f"ref_{name}.npz", **out)
        if "replay_vs_mlp_kprop_maxabs" in out:
            print(f"[ref] {name}: replay vs mlp_kprop max|diff| = {float(out['replay_vs_mlp_kprop_maxabs']):.3e}")
        print(f"[ref] {name}: saved in {out['secs']:.1f}s", flush=True)


# ------------------------------------------------------------------ port side (flopscope)
def sym3(T: np.ndarray) -> np.ndarray:
    return (T + T.transpose(0, 2, 1) + T.transpose(1, 0, 2) + T.transpose(1, 2, 0) + T.transpose(2, 0, 1)
            + T.transpose(2, 1, 0)) / 6.0


def full_k3_from_sources(sources, n):
    T = np.zeros((n, n, n))
    for G, H, E, cv in sources:
        G = np.eye(n) if G is None else np.asarray(G, dtype=np.float64)
        H, E, cv = (np.asarray(a, dtype=np.float64) for a in (H, E, cv))
        T += np.einsum("im,jm,km,m->ijk", H, G, H, cv) + np.einsum("im,jm,km->ijk", E, G, G)
    return sym3(T)


def rel(a, b) -> tuple[float, float]:
    """(max |a-b|, max |a-b| / max |b|) over the whole array."""
    d = float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())
    s = float(np.abs(np.asarray(b, dtype=np.float64)).max())
    return d, d / s if s > 0 else (0.0 if d == 0 else float("inf"))


def cmd_compare(args):
    import flopscope as flops
    import flopscope.numpy as fnp

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT / "estimators"))
    from k3_simple import Estimator

    report = []
    for name, src in cases(args.cases):
        ref = np.load(DATA / f"ref_{name}.npz")
        W, gt = get_weights(src)
        depth, n, _ = W.shape
        weights = [fnp.asarray(W[l]) for l in range(depth)]
        line = {"case": name, "n": n, "depth": depth}
        for dtn in args.dtypes.split(","):
            dt = fnp.float64 if dtn == "float64" else fnp.float32
            rec = {"want_sources": n <= 16} if n <= 128 else None
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
                t0 = time.time()
                est = np.asarray(Estimator().propagate(weights, dt=dt, record=rec), dtype=np.float64)
                secs = time.time() - t0
            L = len(ref["means"])  # fewer than depth only if the reference run was cut short
            per_layer = [rel(est[l], ref["means"][l]) for l in range(L)]
            worst_rel = max(r for _, r in per_layer)
            worst_abs = max(a for a, _ in per_layer)
            line[dtn] = {"layers_compared": L, "means_maxabs": worst_abs, "means_maxrel_per_layer": worst_rel,
                         "final_maxabs": per_layer[-1][0], "final_rel": per_layer[-1][1],
                         "rel_by_layer": [float(f"{r:.2e}") for _, r in per_layer],
                         "flops": int(ctx.flops_used), "secs": round(secs, 2)}
            if gt is not None and L == depth:
                line[dtn]["final_mse_vs_gt"] = float(((est[-1] - gt[-1]) ** 2).mean())
                line["ref_final_mse_vs_gt"] = float(((ref["means"][-1] - gt[-1]) ** 2).mean())
            if rec is not None and dtn == "float64":
                checks = {}
                checks["pre_var"] = max(rel(rec["pre_var"][l], ref["pre_var"][l])[1] for l in range(depth))
                checks["t3"] = max(rel(rec["t3"][l], ref["t3"][l])[1] for l in range(1, depth))
                checks["T21"] = max(rel(rec["T21"][l], ref["T21"][l])[1] for l in range(1, depth - 1))
                checks["c4_pre"] = max(rel(rec["c4_pre"][l], ref["c4_pre"][l])[1] for l in range(1, depth))
                checks["act_cov"] = max(rel(rec["act_cov"][l], ref["act_cov"][l])[1] for l in range(depth - 1))
                checks["c4_act"] = max(rel(rec["c4"][l], ref["c4"][l])[1] for l in range(depth - 1))
                checks["K3_3"] = max(rel(rec["K3_3"][l], ref["K3_3"][l])[1] for l in range(depth - 1))
                checks["K3_21"] = max(rel(rec["K3_21"][l], ref["K3_21"][l])[1] for l in range(depth - 1))
                if "K3_full" in ref.files:
                    checks["K3_full"] = max(rel(full_k3_from_sources(rec["sources"][l], n), ref["K3_full"][l])[1]
                                            for l in range(depth - 1))
                line["internals_maxrel_float64"] = {k: float(f"{v:.3e}") for k, v in checks.items()}
        if "replay_vs_mlp_kprop_maxabs" in ref.files:
            line["ref_replay_vs_mlp_kprop_maxabs"] = float(ref["replay_vs_mlp_kprop_maxabs"])
        print(json.dumps(line), flush=True)
        report.append(line)
    out = HERE / f"k3_port_check_{time.strftime('%m%d_%H%M')}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", out.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "compare"])
    ap.add_argument("--cases", default="small")
    ap.add_argument("--dtypes", default="float64,float32")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "ref":
        import torch
        torch.set_num_threads(args.threads)
        cmd_ref(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
