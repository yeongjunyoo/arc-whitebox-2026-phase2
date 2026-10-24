"""R25-A build v4 (Nova 2026-10-15, ARC R25 lane A, taps v3): the corrector retrained with v2a's taps 0-2 plus kept new
taps (research/r25a/make_r25a_qx2.py: 4, 5 the 3T^T off-diagonal energies, 6 the compression's dropped leg energy, 7 the
pre-activation correlation energy), on a v2a-line folder (v2a itself or an R25-B cheaper engine built from it).
Copies estimators/<base> to estimators/<base>_r25av4<tag>: estimator.py gets the qx2 taps with the kept ones written to
QX rows 3, 4, ... in the model's order (the corrector reads rows 0..kq-1), wmsg.npz the folded model.

Folding as research/r25a/make_r25a_v2.py: the 31 shipped features through research/gru_build/make_wmsg_npz.py, the taps'
gate weights to wqx (kq, 96) with their z-score folded and their mean share moved to the bias row, qxc (15, kq) the
model's per-layer tap scales in its tap order.

    python -X utf8 research/r25a/make_r25a_v4.py --base <v2a-line folder> --model <lc3_wmsg_<tag>.npz> --tag a
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MK = _mod("mkw", ROOT / "research" / "gru_build" / "make_wmsg_npz.py")
Q2 = _mod("mkq2", HERE / "make_r25a_qx2.py")
Q3 = _mod("mkq3", HERE / "make_r25a_qx3.py")
Q4 = _mod("mkq4", HERE / "make_r25a_qx4.py")
Q5 = _mod("mkq5", HERE / "make_r25a_qx5.py")
H = MK.H


def fold(src: Path, dst: Path):
    z = dict(np.load(src).items())
    sel = [int(v) for v in z.pop("qxsel")]
    qxc = z.pop("qxc").astype(np.float64)
    kq = len(sel)
    assert sel[:3] == [0, 1, 2] and all(4 <= k <= 14 for k in sel[3:]) and sel[3:] == sorted(sel[3:]), sel
    mu, sd = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
    Wih = z["cell.weight_ih"].astype(np.float64)
    assert Wih.shape == (3 * H, 33 + kq) and mu.shape == (31 + kq,)
    base = dict(z)
    keep = list(range(31)) + [31 + kq, 32 + kq]
    base["cell.weight_ih"] = z["cell.weight_ih"][:, keep]
    base["mu"], base["sd"] = z["mu"][:31], z["sd"][:31]
    tmp = dst.parent / (dst.stem + "_base31.npz")
    np.savez(tmp, **base)
    MK.main(str(tmp), str(dst))
    tmp.unlink()
    out = dict(np.load(dst).items())
    r_, z_, n_ = slice(0, H), slice(H, 2 * H), slice(2 * H, 3 * H)
    Wq = Wih[:, 31:31 + kq] / sd[None, 31:31 + kq]
    G = np.zeros((3 * H, kq))
    G[0:H], G[H:2 * H], G[2 * H:] = 0.5 * Wq[z_], 0.5 * Wq[r_], Wq[n_]
    bq = -(G @ mu[31:31 + kq])
    wfb = out["wfb"].astype(np.float64)
    wfb[:, MK.FB_ROWS.index("one")] += bq[None, :]
    out["wfb"] = wfb.astype(np.float32)
    out["wqx"] = np.ascontiguousarray(G.T.astype(np.float32))  # (kq, 96): rows = QX rows 0..kq-1
    out["qxc"] = np.ascontiguousarray(qxc[:, sel].astype(np.float32))  # (15, kq) in the same order
    np.savez(dst, **{k: np.ascontiguousarray(v) for k, v in out.items()})
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    src = ROOT / "estimators" / a.base
    dst = ROOT / "estimators" / f"{a.base}_r25av4{a.tag}"
    s = (src / "estimator.py").read_text(encoding="utf-8")
    base_sha = hashlib.sha256(s.encode()).hexdigest()[:16]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    sel = fold(Path(a.model), dst / "wmsg.npz")
    new = sel[3:]
    xmap = {k: 3 + j for j, k in enumerate(new)}
    if any(k >= 8 for k in new):  # qx3 taps (research/r25a/make_r25a_qx3.py); tap 11 pushes tap 5, made in a spare row if unread
        eng = sorted(set(new) | ({5} if 11 in new else set()) | ({7} if 14 in new else set()))  # 14 pushes tap 7
        for j, k in enumerate(k for k in eng if k not in xmap):
            xmap[k] = 3 + len(new) + j
        s = Q3.apply(s, eng, rows=xmap, nrows=3 + len(xmap))
        if 12 in eng:  # research/r25a/make_r25a_qx4.py: what the restricted pushes drop
            s = Q4.add12(s)
        if 13 in eng or 14 in eng:  # research/r25a/make_r25a_qx5.py: the k31 closure sum, the carried (2,2) proxy
            s = Q5.add(s)
    else:
        s = Q2.apply(s, new)
        s = Q2.patch(s, f"_QX = {{'on': True, 'buf': None, 'k': 3, 'x': { {k: k for k in new}!r}, 'rows': 8}}",
                     f"_QX = {{'on': True, 'buf': None, 'k': 3, 'x': {xmap!r}, 'rows': {3 + len(new)}}}")
    head = (f"# R25-A build v4 (research/r25a/make_r25a_v4.py, Nova 2026-10-15, ARC R25 lane A): base {a.base} "
            f"(estimator.py sha16 {base_sha}), corrector {Path(a.model).name} with taps {sel}.\n"
            "#   v4: v2a's taps 0-2 and the kept qx2 taps (research/r25a/make_r25a_qx2.py) written to QX rows 0..kq-1, the\n"
            "#   corrector retrained by research/r25a/gru_lean.py on the gA pool (S and held-out excluded, mini 44-49 for early\n"
            "#   stopping) and folded here. Ours; no participant's code was read.\n")
    s = head + s
    (dst / "estimator.py").write_bytes(s.encode("utf-8"))
    print(f"{dst.name}: estimator.py {hashlib.sha256(s.encode()).hexdigest()[:16]} (base {base_sha}), wmsg.npz "
          f"{hashlib.sha256((dst / 'wmsg.npz').read_bytes()).hexdigest()[:16]}, taps {sel} -> rows {list(range(len(sel)))}")


if __name__ == "__main__":
    main()
