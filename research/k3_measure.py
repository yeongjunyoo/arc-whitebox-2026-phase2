"""Score estimators/k3_simple.py on public mini MLPs the way the harness meters it: predict(mlp, 2**41) inside
flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120) (as whest-starterkit/local_engine.py does), one MLP
at a time, refusing to start an MLP when less than --min-free-gb of RAM is free. Reports final-layer MSE against
the dataset's ground truth, the metered FLOPs C, C/B, the adjusted score MSE * max(0.1, C/B), residual wall time,
and where the FLOPs went (by flopscope namespace and by op).

    cd whest-starterkit
    python <vault>/scripts/memcap.py --gb 4 -- <abs>\\.venv\\Scripts\\python.exe ..\\research\\k3_measure.py --mlps 0,1,2
    (options: --dtype float64, --max-sources J for the truncation study, --threads 2 to mimic the grader's 2 vCPUs)
"""
from __future__ import annotations

import argparse
import os
import sys


def _early_threads() -> None:
    if "--threads" in sys.argv:
        k = sys.argv[sys.argv.index("--threads") + 1]
        for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = k


_early_threads()

import ctypes  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "estimators"))
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from k3_port_check import MINI  # noqa: E402
from k3_simple import Estimator  # noqa: E402
from whestbench import SetupContext  # noqa: E402
from whestbench.domain import MLP  # noqa: E402

B = 2**41


def free_gb() -> float:
    if sys.platform != "win32":  # macOS and others: available memory from vm_stat (macOS) or /proc/meminfo
        try:
            import subprocess
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(out.split("page size of ")[1].split()[0])
            pages = sum(int(ln.split(":")[1].strip().rstrip(".")) for ln in out.splitlines()
                        if ln.startswith(("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")))
            return pages * page / 2**30
        except Exception:  # noqa: BLE001
            return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 2**30

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def load_row(mlp_id: int):
    import pyarrow as pa

    def flat(cell, shape):
        if isinstance(cell, pa.ExtensionScalar):
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
            return (flat(t.column("weights")[i], (16, 1024, 1024)), flat(t.column("all_layer_means")[i], (16, 1024)),
                    int(t.column("mlp_seed")[i].as_py()), str(t.column("mlp_name")[i].as_py()))
    raise KeyError(mlp_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", default="0,1,2")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--threads", default=None)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--budget", type=float, default=B, help="BudgetContext cap (raise only for float64 studies)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dt = fnp.float64 if args.dtype == "float64" else fnp.float32
    results = []
    for mlp_id in [int(x) for x in args.mlps.split(",")]:
        fg = free_gb()
        if fg < args.min_free_gb:
            print(f"[skip] mlp {mlp_id}: only {fg:.2f} GB free (< {args.min_free_gb})", flush=True)
            continue
        W, gt, seed, name = load_row(mlp_id)
        mlp = MLP(width=1024, depth=16, weights=[fnp.asarray(W[l]) for l in range(16)], seed=seed, name=name)
        mlp.validate()
        est = Estimator(dtype=dt, max_sources=args.max_sources)
        est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1.0", submission_dir=None, seed=0))
        budget = int(args.budget)
        with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
            pred = est.predict(mlp, budget)
        pred = np.asarray(pred, dtype=np.float64)
        by_ns: dict[str, int] = defaultdict(int)
        by_op: dict[str, int] = defaultdict(int)
        for r in ctx.op_log:
            by_ns[str(getattr(r, "namespace", None))] += int(r.flop_cost)
            by_op[str(r.op_name)] += int(r.flop_cost)
        C = int(ctx.flops_used)
        mse_final = float(((pred[-1] - gt[-1].astype(np.float64)) ** 2).mean())
        mse_all = float(((pred - gt.astype(np.float64)) ** 2).mean())
        rec = {"mlp_id": mlp_id, "name": name, "dtype": args.dtype, "max_sources": args.max_sources,
               "threads": args.threads, "final_mse": mse_final, "all_layers_mse": mse_all, "flops": C,
               "C_over_B": C / B, "adjusted": mse_final * max(0.1, C / B),
               "mse_by_layer": [float(f"{v:.3e}") for v in ((pred - gt) ** 2).mean(axis=1)],
               "wall_s": round(ctx.wall_time_s, 2), "residual_s": round(ctx.residual_wall_time_s, 4),
               "backend_s": round(ctx.flopscope_backend_time_s, 2),
               "overhead_s": round(ctx.flopscope_overhead_time_s, 2),
               "op_log_sum_matches": sum(by_op.values()) == C,
               "flops_by_namespace": dict(sorted(by_ns.items(), key=lambda kv: -kv[1])),
               "flops_by_op": dict(sorted(by_op.items(), key=lambda kv: -kv[1])[:8]),
               "free_gb_before": round(fg, 2)}
        print(json.dumps({k: v for k, v in rec.items() if k not in ("mse_by_layer", "flops_by_op")}), flush=True)
        results.append(rec)
        del mlp, W, pred
    out = HERE / f"k3_measure_{args.dtype}{'_J' + str(args.max_sources) if args.max_sources is not None else ''}" \
                 f"{'_t' + args.threads if args.threads else ''}{args.tag}_{time.strftime('%m%d_%H%M')}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    if results:
        mean_adj = sum(r["adjusted"] for r in results) / len(results)
        print(f"mean adjusted over {len(results)} MLPs: {mean_adj:.4e}   saved {out.name}")


if __name__ == "__main__":
    main()
