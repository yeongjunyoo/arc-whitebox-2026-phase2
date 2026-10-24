"""R25-A R27-A (Nova 2026-10-17, coordinator 21:xx): the distillation teacher. Every member (lean correctors of the e60
recipe trained on the same dump) runs on every training MLP of <dir>; the members' per-layer dp (15, n) are averaged and
written to <out>/<tag>.npz (key dp, float32). The members' tap scales (qxc) must be equal (same dump, same training tags),
so each MLP is loaded once. Training MLPs only (the tag list of the members' own training records); no S, no validation.

    (Mac) .venv-lc3/bin/python research/r25a/teacher_export.py --dir research/r25a/gru/qx3a15 \
        --models research/r25a/gru/qx3a15/lc3_wmsg_e60s1.npz,... --out research/r25a/gru/qx3a15/teacher_e60
"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first k training MLPs only")
    a = ap.parse_args()
    t0 = time.time()
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G.point(d)
    paths = [Path(p) for p in a.models.split(",")]
    raw = [dict(np.load(p).items()) for p in paths]
    for r in raw[1:]:
        assert np.array_equal(r["qxsel"], raw[0]["qxsel"]) and np.array_equal(r["qxc"], raw[0]["qxc"]), "members differ in taps"
    models = [G.model_from(p) for p in paths]  # the last model_from leaves the shared qxsel / qxc in place
    rec = json.load(open(sorted(glob.glob(str(d / f"train_{paths[0].stem.replace('lc3_wmsg_', '')}_*.json")))[-1]))
    tags = rec["train"]
    S = {f"full_{i:04d}" for i in range(48)}
    assert not set(tags) & S and not set(tags) & set(rec["val"]), "training tags overlap S or validation"
    tags = tags[:a.limit] if a.limit else tags
    with torch.no_grad():
        for i, t in enumerate(tags):
            dd = M.load_mlp(t, True)
            W = torch.as_tensor(M.layer_weights(t))
            dps = [m(M.to_t(dd), W).numpy().astype(np.float64) for m in models]
            np.savez(out / f"{t}.npz", dp=np.mean(dps, 0).astype(np.float32))
            if i % 50 == 0:
                print(f"{i}/{len(tags)} {t} {time.time() - t0:.0f}s", flush=True)
    (out / "teacher.json").write_text(json.dumps({"members": [str(p) for p in paths], "tags": len(tags),
                                                  "secs": round(time.time() - t0)}, indent=1), encoding="utf-8")
    print(f"teacher of {len(paths)} on {len(tags)} training MLPs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
