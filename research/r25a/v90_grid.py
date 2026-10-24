"""R25-A R27-A (Nova 2026-10-17, coordinator 21:xx): the pre-registered V90 pick of the distillation axis. Every model runs as
a lean corrector on the V90 dumps of c5i175 (<dir>; the tags of --tagfile: the 84 public MLPs of tags_extra.json and mini
44-49). A model's score is its pooled ratio to v5 (33's corrector) on the same dumps: the sum over V90 of its final-layer
MSE over the same sum for v5. Reported: v5, each e60 seed, the teacher ensemble (the members' final-layer dp averaged, each
member its own recurrence; not shippable at one corrector's FLOPs, so never eligible) and each student, with the MLPs where
each beats v5, a paired bootstrap of the pooled ratio (20,000 resamples, seed 0) and the extras and mini parts apart. Only
students are eligible, at <= margin (0.995) x v5; the pick is the lowest eligible student, else NONE. S is not read.
Writes --out (json) and prints the pick on the last line.

    (Mac) .venv-lc3/bin/python research/r25a/v90_grid.py --dir research/r25a/gru/v90_c5 --tagfile research/r25a/gru/tags_v90.json \
        --ref research/r25a/gru/qx3a15/lc3_wmsg_wd4s2.npz --seeds <npz,...> --teacher <npz,...> --students <npz,...> \
        --out research/r25a/gru/v90_grid_r27a.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gru_lean as G  # noqa: E402

M = G.M
nm = lambda p: p.stem.replace("lc3_wmsg_", "", 1)  # noqa: E731  (the model tag: wd4s2, e60s2, kd05s1, ...)


def dp15(paths, tags):
    """{model tag: {tag: final-layer dp}} and {tag: (pred14, gt14, pre15 MSE)}; models with equal taps share one load."""
    groups = {}
    for p in paths:
        z = np.load(p)
        key = (z["qxsel"].tobytes(), z["qxc"].tobytes()) if "qxsel" in z else (b"", b"")
        groups.setdefault(key, []).append(p)
    out, base = {nm(p): {} for p in paths}, {}
    with torch.no_grad():
        for ps in groups.values():
            models = [(nm(p), G.model_from(p)) for p in ps]  # the group's taps are now the lean chain's
            for t in tags:
                d = M.load_mlp(t, True)
                W = torch.as_tensor(M.layer_weights(t))
                z = np.load(G.C.cache_path(t))
                base[t] = (z["pred"][14].astype(np.float64), z["gt"][14].astype(np.float64), float(z["base_final_mse"]))
                for s, m in models:
                    out[s][t] = m(M.to_t(d), W)[14].numpy().astype(np.float64)
    return out, base


def row(mse, ref, tags, extras):
    a, b = np.array([mse[t] for t in tags]), np.array([ref[t] for t in tags])
    rng = np.random.default_rng(0)
    bs = [a[i].sum() / b[i].sum() for i in (rng.integers(0, len(tags), len(tags)) for _ in range(20000))]
    ex = [i for i, t in enumerate(tags) if t in extras]
    mi = [i for i, t in enumerate(tags) if t not in extras]
    part = lambda ix: float(a[ix].sum() / b[ix].sum()) if ix else float("nan")  # noqa: E731
    return {"pooled_vs_v5": float(a.sum() / b.sum()), "boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "better": int((a < b).sum()), "n": len(tags), "extras_vs_v5": part(ex), "mini_vs_v5": part(mi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tagfile", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--students", default="")
    ap.add_argument("--margin", type=float, default=0.995)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    d = Path(a.dir)
    G.point(d)
    tags = json.loads(Path(a.tagfile).read_text())
    miss = [t for t in tags if not (d / "cache" / f"{t}.npz").exists()]
    assert not miss, f"V90 dumps missing: {miss}"
    extras = {t for t in tags if t.startswith("full_")}
    sp = lambda s: [Path(p) for p in s.split(",") if p]  # noqa: E731
    seeds, teacher, students = sp(a.seeds), sp(a.teacher), sp(a.students)
    for p in seeds + teacher + students + [Path(a.ref)]:
        assert p.exists(), p
    paths = list(dict.fromkeys([Path(a.ref)] + seeds + teacher + students))
    dps, base = dp15(paths, tags)
    mse = lambda dp, t: float(((base[t][0] + dp - base[t][1]) ** 2).mean())  # noqa: E731
    per = {s: {t: mse(v[t], t) for t in tags} for s, v in dps.items()}
    if teacher:
        per["teacher"] = {t: mse(np.mean([dps[nm(p)][t] for p in teacher], 0), t) for t in tags}
    ref = per[nm(Path(a.ref))]
    pre = {t: base[t][2] for t in tags}
    rows = {}
    for name, kind in ([(nm(Path(a.ref)), "v5")] + [(nm(p), "seed") for p in seeds] + ([("teacher", "teacher")] if teacher else [])
                       + [(nm(p), "student") for p in students]):
        rows[name] = {"kind": kind, **row(per[name], ref, tags, extras),
                      "pooled_vs_pre15": float(sum(per[name].values()) / sum(pre.values()))}
        r = rows[name]
        print(f"{kind:8s} {name:20s} x{r['pooled_vs_v5']:.5f} [{r['boot95'][0]:.5f}, {r['boot95'][1]:.5f}] {r['better']}/{r['n']}"
              f"  extras x{r['extras_vs_v5']:.5f}  mini x{r['mini_vs_v5']:.5f}  vs pre15 {r['pooled_vs_pre15']:.5f}", flush=True)
    cand = {nm(p): rows[nm(p)]["pooled_vs_v5"] for p in students}
    elig = {k: v for k, v in cand.items() if v <= a.margin}
    pick = min(elig, key=elig.get) if elig else "NONE"
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "dir": str(d), "tags": tags, "ref": a.ref, "margin": a.margin,
           "teacher_members": [str(p) for p in teacher], "rows": rows, "eligible": elig, "pick": pick,
           "per_mlp_final_mse": per, "pre15": pre, "secs": round(time.time() - t0)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"eligible (<= x{a.margin} of v5): {sorted(elig, key=elig.get) or 'none'}")
    print(pick)


if __name__ == "__main__":
    main()
