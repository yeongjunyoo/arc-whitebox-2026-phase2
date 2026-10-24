"""Write the estimator copies of the band study (research/wu/mech): the submitted engine (estimators/r38/r38_sub_x100_s2,
submission 337488) as make_variants.py would cut it, and the same files with two read/replace hooks.

    .venv-fs0123/bin/python research/wu/mech/make_mech.py --base estimators/r38/r38_sub_x100_s2 --out research/wu/mech/est

  v_engine        make_variants.py ENGINE on the submitted file (gru, lc, law off): the band on (production)
  v_engine_nott   ENGINE + NOTT: the 3 T^T band off (_SAT_BINS tt tables emptied, R25-B step s7 popped)
  v_engine_nos7   ENGINE + only s7 popped: the band on, the compressed sources' legs and P made on every column
  mech_prod, mech_nott, mech_nos7   the three above plus HOOKS: a module global _MECH (None: the file computes as its
                  base, bit for bit) called after the K4 star fold made QF (the fold's share of 3 T^T is formed from the
                  live legs before P parks the bases), after 3 T^T (M1) is complete, before anything reads it, and
                  after the layer's C' (activation covariance) is final.

Every replacement must match exactly once; the replacements and the sha256 of each file go to <out>/MECH.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # research/wu
import make_variants as MV  # noqa: E402

NOS7 = [("_R25B['kc'] = 'A'", "_R25B['kc'] = 'A'; _R25B.pop('s7')")]
HOOKS = [
    ("    def _ns(name):\n        return contextlib.nullcontext()\n",
     "    def _ns(name):\n        return contextlib.nullcontext()\n\n"
     "_MECH = None  # research/wu/mech: hook object set by mech_run.py (None: this file computes as its base)\n"),
    ('                            DGF = es("sji,sji,sji->i", QF, g_, h_)  # 3T^T diagonal share / 4\n',
     '                            DGF = es("sji,sji,sji->i", QF, g_, h_)  # 3T^T diagonal share / 4\n'
     "                            if _MECH is not None:  # mech: the fold's share of 3 T^T from the live legs\n"
     "                                _MECH.fold(t, h_, g_, QF, mc, ns)\n"),
    ("                        if K4F is not None:  # r25a k4f: the diagonal stays the K3 read 3 t3\n"
     '                            fnp.fill_diagonal(M1, fnp.subtract(es("ii->i", M1), mul(DGF, 4.0)))\n',
     "                        if K4F is not None:  # r25a k4f: the diagonal stays the K3 read 3 t3\n"
     '                            fnp.fill_diagonal(M1, fnp.subtract(es("ii->i", M1), mul(DGF, 4.0)))\n'
     "                        if _MECH is not None:  # mech: record and (by mode) replace entries of 3 T^T\n"
     "                            _MECH.tt(t, M1, a, Pa, Pb, ord_t, ordb_t, mc, ns,\n"
     "                                     (Lr, Ar) if RIt_ is not None else (LEGt[lo:T0 + 2 * ns * n], ASt[lo:T0 + 2 * ns * n]),\n"
     "                                     (LEGt[:nth], ASt[:nth]), DGF if K4F is not None else None)\n"),
    ("                        fnp.subtract(w1, fnp.outer(mul(v_, 6.0 * eps_), m_), out=w1)\n"
     "                if ns:  # old legs (dense, and the bases of compressed ones), u and Z^T rows to the activation side\n",
     "                        fnp.subtract(w1, fnp.outer(mul(v_, 6.0 * eps_), m_), out=w1)\n"
     "                if _MECH is not None:  # mech: this layer's final C' (activation covariance)\n"
     "                    _MECH.cov(t, Cv)\n"
     "                if ns:  # old legs (dense, and the bases of compressed ones), u and Z^T rows to the activation side\n"),
]
BUILDS = {
    "v_engine": MV.ENGINE,
    "v_engine_nott": MV.ENGINE + MV.NOTT,
    "v_engine_nos7": MV.ENGINE + NOS7,
    "mech_prod": MV.ENGINE + HOOKS,
    "mech_nott": MV.ENGINE + MV.NOTT + HOOKS,
    "mech_nos7": MV.ENGINE + NOS7 + HOOKS,
}


def apply(src: str, reps: list) -> tuple[str, list]:
    """make_variants.main's replacement loop: '^...' selects the one whole line that starts with that text, a tuple
    ("satbins", kind) computes the new line from the matched one; every old text must occur exactly once."""
    s, rec = src, []
    for old, new in reps:
        pat = old
        if old.startswith("^"):
            lines = [ln for ln in s.split("\n") if ln.startswith(old[1:])]
            assert len(lines) == 1, (old, len(lines))
            old = lines[0]
        if isinstance(new, tuple):
            new = MV.satbins_without(old, new[1])
        assert s.count(old) == 1, (old[:80], s.count(old))
        s = s.replace(old, new)
        rec.append({"old": pat, "new": new})
    return s, rec


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    src = (base / "estimator.py").read_text(encoding="utf-8")
    man = {"base": str(base), "base_sha256": {f.name: sha(f) for f in sorted(base.iterdir()) if f.is_file()},
           "make_variants_sha256": sha(HERE.parent / "make_variants.py"), "builds": {}}
    for name, reps in BUILDS.items():
        s, rec = apply(src, reps)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "estimator.py").write_bytes(s.encode("utf-8"))
        shutil.copy2(base / "wmsg.npz", d / "wmsg.npz")
        man["builds"][name] = {"replacements": rec, "sha256": {f.name: sha(f) for f in sorted(d.iterdir()) if f.is_file()}}
        print(name, man["builds"][name]["sha256"]["estimator.py"][:16])
    (out / "MECH.json").write_text(json.dumps(man, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
