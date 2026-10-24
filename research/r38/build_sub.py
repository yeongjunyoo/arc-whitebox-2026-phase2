"""R38b: put correctors trained on the r38 engine's dumps into a copy of estimators/r38/r38_sub_x100 (wmsg.npz folded by
research/r25a/make_r25a_v4.fold with the shipped taps 0,1,2,4-11). One model: estimator.py unchanged. Several models
(comma list): wmsg.npz, wmsg2.npz, ... and an estimator.py whose final mean is the mean of the correctors' final means,
each corrector run on the same engine state (research only until a read picks it).

    (Mac) .venv/bin/python -X utf8 research/r38/build_sub.py <lc3_wmsg_<tag>.npz>[,<more>.npz] <name>
    (Mac) .venv/bin/python -X utf8 research/r38/build_sub.py --check     # fold the shipped kd05s3, compare md5
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v4", str(ROOT / "research" / "r25a" / "make_r25a_v4.py"))
V4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V4)
SHIPPED_TAPS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
SRC = ROOT / "estimators" / "r38" / "r38_sub_x100b"  # x100 with the two per-layer sums as 0-d arrays (same numbers)

LOAD = '            self._wm = {k: z[k] for k in _WM_KEYS + tuple(k for k in ("wqx", "qxc") if k in getattr(z, "files", z))}\n'
LOAD_ENS = LOAD + (
    '            wl_ = [self._wm]  # r38 ens: wmsg2.npz, wmsg3.npz, ... next to this file\n'
    '            for j_ in range(2, 9):\n'
    '                f_ = os.path.join(d, f"wmsg{j_}.npz")\n'
    '                if not os.path.exists(f_):\n'
    '                    break\n'
    '                z_ = fnp.load(str(f_))\n'
    '                wl_.append({k: z_[k] for k in _WM_KEYS + tuple(k for k in ("wqx", "qxc") if k in getattr(z_, "files", z_))})\n'
    '            if len(wl_) > 1:\n'
    '                self._wm = wl_\n')
CALL = '            _wmsg(weights, VX, QB, SVB, wm, f32, self._parts, QX if QXON else None)\n'
CALL_ENS = (
    '            if isinstance(wm, list):  # r38 ens: the mean of the correctors\' final means\n'
    '                pk0_ = fnp.copy(VX[depth - 1, V_PK1])\n'
    '                acc_ = None\n'
    '                for wm_ in wm:\n'
    '                    cp(VX[depth - 1, V_PK1], pk0_)\n'
    '                    _wmsg(weights, VX, QB, SVB, wm_, f32, self._parts, QX if QXON else None)\n'
    '                    acc_ = fnp.copy(VX[depth - 1, V_PK1]) if acc_ is None else fnp.add(acc_, VX[depth - 1, V_PK1])\n'
    '                mul(acc_, 1.0 / len(wm), out=VX[depth - 1, V_PK1])\n'
    '            else:\n'
    '    ' + CALL)


def tap_ids(model: Path, tmpdir: Path) -> Path:
    """A model trained on dumps of the shipped-form estimator reads its taps from compact QX rows 0..10 (qxsel 0..10,
    qxc (15, 11)); make_r25a_v4.fold wants tap ids (qxsel = the shipped taps, qxc indexed by tap id). Same weights."""
    z = dict(np.load(model).items())
    if [int(v) for v in z["qxsel"]] != list(range(len(SHIPPED_TAPS))):
        return model
    qxc = np.asarray(z["qxc"])
    full = np.zeros((qxc.shape[0], max(SHIPPED_TAPS) + 1), dtype=qxc.dtype)
    full[:, SHIPPED_TAPS] = qxc
    z["qxsel"], z["qxc"] = np.array(SHIPPED_TAPS, dtype=np.int64), full
    out = tmpdir / f"_tapids_{model.stem}.npz"
    np.savez(out, **z)
    return out


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    if sys.argv[1] == "--check":
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "wmsg.npz"
            sel = V4.fold(ROOT / "research/r25a/gru/qx3a15/lc3_wmsg_kd05s3.npz", out)
            print("taps", sel, "md5", md5(out), "shipped 5719884ab8444a3d0e5ec1f326799202")
        return
    models, name = sys.argv[1].split(","), sys.argv[2]
    dst = ROOT / "estimators" / "r38" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    src = (SRC / "estimator.py").read_text(encoding="utf-8")
    if len(models) > 1:
        for a_, b_ in ((LOAD, LOAD_ENS), (CALL, CALL_ENS)):
            assert src.count(a_) == 1, (a_[:80], src.count(a_))
            src = src.replace(a_, b_)
    (dst / "estimator.py").write_text(src, encoding="utf-8", newline="\n")
    for j, mp in enumerate(models):
        sel = V4.fold(tap_ids(Path(mp), dst), dst / ("wmsg.npz" if j == 0 else f"wmsg{j + 1}.npz"))
        assert sel == SHIPPED_TAPS, sel
    for p in dst.glob("_tapids_*.npz"):
        p.unlink()
    print(name, "estimator.py", hashlib.sha256((dst / "estimator.py").read_bytes()).hexdigest()[:16],
          "wmsg", [md5(p)[:12] for p in sorted(dst.glob("wmsg*.npz"))])


if __name__ == "__main__":
    main()
