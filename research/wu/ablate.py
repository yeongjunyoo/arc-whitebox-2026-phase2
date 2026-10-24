"""Write-up ablations for submission 336955 (Phase 2): run one estimator on public networks the way the grader meters a
prediction and write one JSON line per network.

Each network: setup() once, then predict(mlp, 2**41) inside flopscope.BudgetContext(flop_budget=2**41,
wall_time_limit_s=120), as the kit's harness does. Recorded: final-layer MSE against the dataset's Monte Carlo means,
the FLOPs flopscope counted, C/B, the adjusted score MSE x max(0.1, C/B), wall and residual time, op calls, and a
sha256 prefix of the prediction (so two runs of the same file can be checked bit for bit).

Data: the public dataset aicrowd/arc-whestbench-public-2026 at revision v2-phase2, from a local snapshot directory
(--snapshot, the huggingface_hub cache folder of that revision). "mini" rows are read from the snapshot's prepared
default split (prepared/mini/data-*.arrow), "full" rows from data/full-<shard>-of-00067.parquet (16 networks per
shard, ids 16 k .. 16 k + 15). "kp" rows are networks of Keenan Pepper's public corpus
keenanpepper/arc-whestbench-p2-d8b-corpus-14k (weights regenerated from its seed rule, truth from its 1e8-sample
means), one kp_NNNNN.npz per network in the --snapshot directory, as kp_make.py writes them.

    python ablate.py --estimator <folder or .py> --set mini --ids 60-99 --snapshot <dir> --out runs/x_mini.jsonl
    python ablate.py --estimator <folder or .py> --set full --ids 304-351 --snapshot <dir> --out runs/x_full.jsonl

Variants are separate estimator folders written by make_variants.py (each differs from the submitted file in the
lines it lists), so every row of the write-up's table is a file a reader can diff and rerun.
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


def _threads(k: str) -> None:
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = k


if "--threads" in sys.argv:
    _threads(sys.argv[sys.argv.index("--threads") + 1])

import numpy as np  # noqa: E402

B = 2 ** 41
N, L = 1024, 16


def ids(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _flat(cell, shape):
    import pyarrow as pa
    if isinstance(cell, pa.ExtensionScalar):
        cell = cell.value
    arr = cell.values
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        arr = arr.flatten()
    return arr.to_numpy(zero_copy_only=False).astype(np.float32).reshape(shape)


def load_rows(snapshot: Path, which: str, want: list[int]):
    """Yield (mlp_id, seed, name, weights (16, n, n) float32, truth (16, n) float32) for the requested ids."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    want_s = set(want)
    if which == "kp":  # Keenan Pepper's corpus: files written by kp_make.py into <snapshot> (W, gt, seed)
        for m in want:
            z = np.load(snapshot / f"kp_{m:05d}.npz")
            yield m, int(z["seed"]), f"kp_{m:05d}", z["W"].astype(np.float32), z["gt"].astype(np.float32)
        return
    if which == "mini":
        for f in sorted((snapshot / "prepared" / "mini").glob("data-*.arrow")):
            with pa.memory_map(str(f)) as src:
                t = pa.ipc.open_stream(src).read_all()
            col = t.column("mlp_id").to_pylist()
            for i, m in enumerate(col):
                if m in want_s:
                    yield (m, int(t.column("mlp_seed")[i].as_py()), str(t.column("mlp_name")[i].as_py()),
                           _flat(t.column("weights")[i], (L, N, N)), _flat(t.column("all_layer_means")[i], (L, N)))
        return
    for shard in sorted({m // 16 for m in want}):
        pf = pq.ParquetFile(str(snapshot / "data" / f"full-{shard:05d}-of-00067.parquet"))
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            col = t.column("mlp_id").to_pylist()
            for i, m in enumerate(col):
                if m in want_s:
                    yield (m, int(t.column("mlp_seed")[i].as_py()), str(t.column("mlp_name")[i].as_py()),
                           _flat(t.column("weights")[i], (L, N, N)), _flat(t.column("all_layer_means")[i], (L, N)))


def load_estimator(path: Path):
    from whestbench import BaseEstimator
    entry = path / "estimator.py" if path.is_dir() else path
    spec = importlib.util.spec_from_file_location(f"est_{entry.parent.name}_{entry.stem}", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(entry.parent))
    spec.loader.exec_module(mod)
    cands = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, BaseEstimator) and v is not BaseEstimator]
    assert len(cands) == 1, [c.__name__ for c in cands]
    return cands[0], entry


def file_sha(entry: Path) -> dict:
    out = {}
    for f in sorted(entry.parent.iterdir()) if entry.name == "estimator.py" else [entry]:
        if f.is_file() and f.suffix in (".py", ".npz"):
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--set", choices=["mini", "full", "kp"], required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--snapshot", required=True, help="mini, full: local snapshot directory of the dataset revision "
                    "v2-phase2; kp: the directory of kp_NNNNN.npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default="1")
    a = ap.parse_args()

    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    Est, entry = load_estimator(Path(a.estimator).resolve())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: networks already written are skipped
        done = {json.loads(x)["mlp_id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip().startswith("{")}
    meta = {"estimator": str(entry), "files_sha256": file_sha(entry), "set": a.set, "ids": a.ids, "threads": a.threads,
            "flopscope": getattr(flops, "__version__", "?"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps({"meta": meta}), flush=True)
    est = Est()
    est.setup(SetupContext(width=N, depth=L, flop_budget=B, api_version="1.0", submission_dir=str(entry.parent), seed=0))
    want = [m for m in ids(a.ids) if m not in done]
    with out.open("a", encoding="utf-8") as fo:
        for mlp_id, seed, name, W, gt in load_rows(Path(a.snapshot), a.set, want):
            mlp = MLP(width=N, depth=L, weights=[fnp.asarray(W[t]) for t in range(L)], seed=seed, name=name)
            mlp.validate()
            with flops.BudgetContext(flop_budget=B, wall_time_limit_s=120.0, quiet=True) as ctx:
                pred = est.predict(mlp, B)
            pred = np.asarray(pred, dtype=np.float64)
            C = int(ctx.flops_used)
            mse = float(((pred[-1] - gt[-1].astype(np.float64)) ** 2).mean())
            rec = {"set": a.set, "mlp_id": mlp_id, "name": name, "final_mse": mse, "flops": C, "C_over_B": C / B,
                   "adjusted": mse * max(0.1, C / B), "wall_s": round(ctx.wall_time_s, 2),
                   "residual_s": round(ctx.residual_wall_time_s, 4), "op_calls": len(ctx.op_log),
                   "finite": bool(np.isfinite(pred).all()),
                   "pred_sha16": hashlib.sha256(np.ascontiguousarray(pred[-1]).tobytes()).hexdigest()[:16]}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(json.dumps(rec), flush=True)
            del mlp, W, gt, pred
    print(json.dumps({"done": time.strftime("%Y-%m-%d %H:%M:%S"), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
