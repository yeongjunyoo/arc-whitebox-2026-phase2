"""R38b (coordinator 2026-10-21): the scale-mode extraction of il.miscusi's public Phase 2 write-up (AIcrowd forum 18248
and its PDF, prose only; no participant code read) in the production engine of shot 38 (r37), with the lam' law off.

The readable engine (research/r38/k3_acc_tc.py, grid research/r38/out/g3.jsonl and g4.jsonl) measured, engine alone on
mini 0-9: lam' = 0 with the extraction at f = 0.1 (all three parts) x0.89 against the shipped law, every part alone
x7 to x39 worse, a fixed per-layer eps table about as good as the online fit (eps varies 2-7 % across networks).
This generator writes the same change into a copy of the r37 estimator by asserted textual patches (the r37 file is
not touched):

  lam' = 0          the Estimator default lam=(0.0,) * 16 (as v_engine_nolaw)
  x38               after the birth window, with eps_t = scale * EPS[t] (EPS: mean online eps of the readable chain):
                    C' <- (C' - eps m m^T) / (1 + eps)                    (covariance, its diagonal included)
                    dA -= (W o W)^T Kmat [4 eps v sum v + 8 eps rowsum C'^2]  (K4 core rows: 12 eps v^2 on the
                                                                           (4) slice, 4 eps (v v^T + 2 C_off^2) on (2,2))
                    E  -= 6 eps v m^T (rows: birth index)                 (K3: the v_i m_j part of 2 eps S_ij)
                    the newborn's feed pair (u, c) := (-6 eps m, w11)     (K3: the 2 m_i C_ij part; the feed is 0
                                                                           when lam' = 0, so its slot is free)

    python research/r38/make_r38.py                      # engine-alone variants (gru, lc, law off)
    python research/r38/make_r38.py --sub --scale 1.0     # the submission form (corrector on), wmsg.npz copied
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R37 = ("k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces"
       "_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30_r36k256_opL4")
SRC = ROOT / "estimators" / R37 / "estimator.py"
SHA16 = "5cce4128344b6e4a"
OUT = ROOT / "estimators" / "r38"
# mean online eps per layer t = 0..14 of the readable chain with lam' = 0 and f = 0.1 (research/r38/out/g3.jsonl,
# config b_e0.1l0, mini 0-5)
EPS = [1.49e-4, 2.27e-4, 2.74e-4, 2.93e-4, 2.93e-4, 3.01e-4, 2.95e-4, 2.88e-4, 2.80e-4, 2.79e-4, 2.76e-4, 2.66e-4,
       2.58e-4, 2.48e-4, 2.34e-4]

ENGINE = [
    ("sw_depth_batched=5, kz=4, kq=32, lc: bool = True, law: bool = True, sw_inplace: bool = True,",
     "sw_depth_batched=5, kz=4, kq=32, lc: bool = False, law: bool = False, sw_inplace: bool = True,"),
    ("gru: bool = True, gru_law: bool = False, gru_stack: bool = False) -> None:",
     "gru: bool = False, gru_law: bool = False, gru_stack: bool = False) -> None:"),
]
NOLAW = [
    ("def __init__(self, dtype=None, force_aug: bool = False, lam=None, age=6,",
     "def __init__(self, dtype=None, force_aug: bool = False, lam=(0.0,) * 16, age=6,"),
]


def x38(scale: float, arr: bool = False) -> list[tuple[str, str]]:
    eps = "[" + ", ".join(f"{e:.3e}" for e in EPS) + "]"
    return [
        ("_K4F = {'scale': 1.0, 'live': True}  # r25a k4f: the K4 star (3,1) slice folded into 3 T^T (scale None: the base)\n",
         "_K4F = {'scale': 1.0, 'live': True}  # r25a k4f: the K4 star (3,1) slice folded into 3 T^T (scale None: the base)\n"
         f"_X38 = {{'on': True, 'scale': {scale!r}, 'eps': {eps}}}  # r38: scale-mode extraction, eps_t = scale eps[t]\n"),
        ("                Cv = BLK[2 * ns + 2]\n",
         "                Cv = BLK[2 * ns + 2]\n"
         "                x38_ = _X38.get('on') and t < len(_X38['eps'])\n"
         "                if x38_:  # r38: the scale mode out of C', the K4 core rows (dA) and the newborn's K3 (E; feed pair below)\n"
         "                    with _ns(\"x38\"):\n"
         "                        eps_ = float(_X38['eps'][t]) * float(_X38['scale'])\n"
         "                        m_ = VX[t, V_PK1]\n"
         "                        fnp.subtract(Cv, fnp.outer(m_, fnp.multiply(m_, eps_)), out=Cv)\n"
         "                        mul(Cv, 1.0 / (1.0 + eps_), out=Cv)\n"
         "                        v_ = es(\"ii->i\", Cv)\n"
         "                        # 12 eps v^2 + 4 eps v (sum v - v) + 8 eps (rowsum C'^2 - v^2) = 4 eps v sum v + 8 eps rowsum C'^2\n"
         + ("                        dr_ = fnp.add(mul(v_, mul(fnp.sum(v_), 4.0 * eps_)), mul(es(\"ij,ij->i\", Cv, Cv), 8.0 * eps_))\n"
            "                        kr_ = fnp.add(mul(dr_, kd), mul(fnp.sum(dr_), kt))\n" if arr else
            "                        dr_ = fnp.add(mul(v_, 4.0 * eps_ * float(fnp.sum(v_))), mul(es(\"ij,ij->i\", Cv, Cv), 8.0 * eps_))\n"
            "                        kr_ = fnp.add(mul(dr_, kd), kt * float(fnp.sum(dr_)))\n") +
         "                        dA = fnp.subtract(dA, es(\"ia,ia,i->a\", Wn, Wn, kr_))\n"
         "                        fnp.subtract(w1, fnp.outer(mul(v_, 6.0 * eps_), m_), out=w1)\n"),
        ("                cp(UPs[ns], VW21s[t])  # u of the new source (its g / 4 is in c)\n",
         "                cp(UPs[ns], VW21s[t])  # u of the new source (its g / 4 is in c)\n"
         "                if x38_:  # r38: the extraction's K3 pair (-6 eps m, w11) in the feed slot (the feed is 0 at lam' = 0)\n"
         "                    mul(VX[t, V_PK1], -6.0 * eps_, out=UPs[ns])\n"
         "                    cp(CSRC[ns], VW11s[t])\n"),
    ]


# lam' off in the engine only: the VEC rows are made from FF with its two lam' rows masked, so the corrector keeps
# reading the lam' rows it was trained with (QB rows 4, 5) while every engine term of the law is 0 (as lam=(0,)*16)
LAMMASK = [
    ("        LAMV = CR[_NP + 4:]\n",
     "        LAMV = CR[_NP + 4:]\n"
     "        LMK4 = fnp.array([[1.0], [1.0], [0.0], [0.0]], dtype=f32)  # r38: lam' rows out of the engine's VEC\n"
     "        LMK6 = fnp.array([[1.0], [1.0], [0.0], [0.0], [1.0], [1.0]], dtype=f32)\n"),
    ("                    es(\"rcdpe,ci,di,pi,ei->ri\", cVl, FF[0:3], FF[2:8], Mon, SV, out=VX[t, V_PK1:V_PK1 + 1])\n",
     "                    es(\"rcdpe,ci,di,pi,ei->ri\", cVl, FF[0:3], mul(FF[2:8], LMK6), Mon, SV, out=VX[t, V_PK1:V_PK1 + 1])\n"),
    ("                XA3 = es(\"kii,ci,di->kcdi\", MZ, FF[0:3], FF[2:6])\n",
     "                XA3 = es(\"kii,ci,di->kcdi\", MZ, FF[0:3], mul(FF[2:6], LMK4))\n"),
]


def patch(text: str, reps: list[tuple[str, str]]) -> str:
    for a, b in reps:
        assert text.count(a) == 1, (a[:90], text.count(a))
        text = text.replace(a, b)
    return text


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def write(name: str, reps, wmsg: Path | None = None) -> None:
    src = SRC.read_text(encoding="utf-8")
    assert sha16(SRC) == SHA16, sha16(SRC)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "estimator.py").write_text(patch(src, reps), encoding="utf-8", newline="\n")
    if wmsg is not None:
        shutil.copy2(wmsg, d / "wmsg.npz")
    print(name, sha16(d / "estimator.py"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--wmsg", default=str(SRC.parent / "wmsg.npz"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--arr", action="store_true", help="the two per-layer sums as 0-d arrays (no Python-float fetch)")
    a = ap.parse_args()
    if a.sub:  # lam' masked in the engine (kept in the corrector's rows), the extraction, the given corrector
        write(f"r38_sub_x{int(round(a.scale * 100)):03d}{a.tag}", LAMMASK + x38(a.scale, a.arr), Path(a.wmsg))
        return
    write("r38_eng_base", ENGINE)  # the r37 engine alone, lam' law on: the base of every r38_eng_* row
    write("r38_eng_nolaw", ENGINE + NOLAW)
    for s in (0.7, 0.85, 0.92, 1.0, 1.08, 1.17, 1.4):
        write(f"r38_eng_x{int(round(s * 100)):03d}", ENGINE + NOLAW + x38(s))
    write("r38_eng_mx100", ENGINE + LAMMASK + x38(1.0))  # must match r38_eng_x100 (the mask instead of lam=0)


if __name__ == "__main__":
    main()
