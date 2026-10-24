"""Write the ablation variants of the submitted estimator folder (submission 336955) for the write-up's held-out table.

Every variant is a copy of the submitted folder (estimator.py and wmsg.npz) with exact string replacements, each
checked to match exactly once, so a reader can diff any variant against the submitted file. The replacements of
each variant are written to <out>/VARIANTS.json with the sha256 of every file.

    python make_variants.py --base <submitted folder> --out <estimators dir>

  v_engine         the engine alone: Estimator defaults gru=False, lc=False, law=False
  v_engine_nostar  v_engine without the K4 star (3,1) fold and its G fold (_K4F['scale'] = None, _GF['scale'] = None)
  v_engine_nolaw   v_engine with the memoryless fourth-cumulant law at zero (Estimator default lam = 16 zeros)
  v_engine_nok4f   v_engine without the star (3,1) fold only (_K4F['scale'] = None)
  v_engine_nogf    v_engine without the G fold only (_GF['scale'] = None)
  v_nostar         the submitted file without the two folds (the corrector was trained with them on)
  v_noapa          the submitted file without the approximate bilinear level (_APA_ORI = {})
  v_engine_nosat   v_engine with every restricted product at full width (no allowed shapes, no dead-center gather, no
                   R25-B restriction steps): an experiment, kept only if it runs and checks out
  v_engine_nodc    v_engine without the dead-center gather and its birth rule (_DC theta None)
  v_engine_nokc    v_engine with the plain weighted compression SVD instead of the kept-subspace one (_R25B kc None)
  v_engine_noshape v_engine without the restricted shapes (_SAT_BINS empty, R25-B steps s1, s8, s2, s7 off)
  v_notaps         the submitted file with the folds on and the corrector's eleven structure taps off (QXON False)
  v_engine_notaps  v_engine with the taps off: what building them costs when nothing reads them
  v_tapsnofold     the submitted file with the taps built as usual and both folds left out of 3T^T and the G legs
                   (with v_nostar and v_notaps, the folds and the taps apart; review round 4, 10-21)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ENGINE = [
    ("sw_depth_batched=5, kz=4, kq=32, lc: bool = True, law: bool = True, sw_inplace: bool = True,",
     "sw_depth_batched=5, kz=4, kq=32, lc: bool = False, law: bool = False, sw_inplace: bool = True,"),
    ("gru: bool = True, gru_law: bool = False, gru_stack: bool = False) -> None:",
     "gru: bool = False, gru_law: bool = False, gru_stack: bool = False) -> None:"),
]
NOSTAR = [
    ("_K4F = {'scale': 1.0, 'live': True}", "_K4F = {'scale': None, 'live': True}"),
    ("_GF = {'scale': 0.5}", "_GF = {'scale': None}"),
]
NOLAW = [
    ("def __init__(self, dtype=None, force_aug: bool = False, lam=None, age=6,",
     "def __init__(self, dtype=None, force_aug: bool = False, lam=(0.0,) * 16, age=6,"),
]
NOAPA = [("_APA_ORI = {'tt': '232', 'push': '322'}", "_APA_ORI = {}")]
NOSAT = [  # every restricted product at full width: no allowed shapes, no dead-center gather, no R25-B restriction steps
    ("^_SAT_BINS = ", '_SAT_BINS = {"push": {}, "tt": {}, "skip_plain": []}  # write-up variant: no restricted shapes'),
    ("_DC = {'theta': '2.0', 'gather': True}", "_DC = {'theta': None, 'gather': True}"),
    ("_R25B['kc'] = 'A'", "_R25B['kc'] = None; [_R25B.pop(k_) for k_ in ('s1', 's8', 's2', 's7')]"),  # keys removed: the code reads them with .get
]

NODC = [NOSAT[1]]  # the dead-center gather and its birth rule off; restricted shapes and kc stay
NOKC = [("_R25B['kc'] = 'A'", "_R25B['kc'] = None")]  # plain weighted compression SVD; shapes and the gather stay
NOSHAPE = [NOSAT[0], ("_R25B['kc'] = 'A'", "_R25B['kc'] = 'A'; [_R25B.pop(k_) for k_ in ('s1', 's8', 's2', 's7')]")]  # shapes off
NOTAPS = [("QXON = _QX.get('on') and K4F is not None and GF is not None", "QXON = False  # write-up variant: no taps")]
TAPSNOFOLD = [  # the fold quantities are still made (taps 0-3 read them); only their additions are left out
    ('fnp.add(ah_, es("sji,sji,x->sji", QF, g_, c_three), out=ah_)  # + 3 s w31 lam G H^2',
     "pass  # write-up variant: made for the taps, not added"),
    ("fnp.add(ag_, mul(QF, h_), out=ag_)  # + s w31 lam H^3", "pass  # write-up variant: made for the taps, not added"),
    ('fnp.fill_diagonal(M1, fnp.subtract(es("ii->i", M1), mul(DGF, 4.0)))', "pass  # write-up variant: no fold to take out"),
    ("fnp.add(Gd_, Xf_, out=Gd_)", "pass  # write-up variant: the G increment made for the taps, not added"),
]

VARIANTS = {
    "v_engine": ENGINE,
    "v_engine_nostar": ENGINE + NOSTAR,
    "v_engine_nolaw": ENGINE + NOLAW,
    "v_engine_nok4f": ENGINE + NOSTAR[:1],
    "v_engine_nogf": ENGINE + NOSTAR[1:],
    "v_nostar": NOSTAR,
    "v_noapa": NOAPA,
    "v_engine_nosat": ENGINE + NOSAT,
    "v_engine_nodc": ENGINE + NODC,
    "v_engine_nokc": ENGINE + NOKC,
    "v_engine_noshape": ENGINE + NOSHAPE,
    "v_notaps": NOTAPS,
    "v_engine_notaps": ENGINE + NOTAPS,
    "v_tapsnofold": TAPSNOFOLD,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    src = (base / "estimator.py").read_text(encoding="utf-8")
    manifest = {"base": base.name, "base_sha256": {f.name: sha(f) for f in sorted(base.iterdir()) if f.is_file()}, "variants": {}}
    for name, reps in VARIANTS.items():
        s = src
        for old, new in reps:
            if old.startswith("^"):  # the one whole line that starts with this text
                lines = [ln for ln in s.split("\n") if ln.startswith(old[1:])]
                assert len(lines) == 1, (name, old, len(lines))
                old = lines[0]
            assert s.count(old) == 1, (name, old[:80], s.count(old))
            s = s.replace(old, new)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "estimator.py").write_bytes(s.encode("utf-8"))
        shutil.copy2(base / "wmsg.npz", d / "wmsg.npz")
        manifest["variants"][name] = {"replacements": [{"old": o, "new": n} for o, n in reps],
                                      "sha256": {f.name: sha(f) for f in sorted(d.iterdir()) if f.is_file()}}
        print(name, manifest["variants"][name]["sha256"]["estimator.py"][:16])
    (out / "VARIANTS.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
