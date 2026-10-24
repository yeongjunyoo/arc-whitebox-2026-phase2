"""Write the estimator cuts of the band study's second round (research/wu/mech, coordinator 10-22):

  est_opr30/mech_prod, mech_nott   the opr30 cuts behind kp_table.json (estimators/wu2/v_engine, v_engine_nott, base
                                   ..._opr30, submission 336955) with make_mech.py's first three hooks (the C' hook's
                                   anchor is r38-only); _MECH None: the file computes as its cut
  est_r37/v_engine, v_engine_nott  make_variants.py ENGINE / ENGINE + NOTT on the r37 file (opr30 + R36 rank 256 + L4;
                                   lam' law on, no extraction). v_engine must equal estimators/r38/r38_eng_base.
  est_r38x/<cut>_<mode>            the submitted r38 file cut as v_engine / v_engine_nott with R38's two changes toggled:
                                   lawon (lam' rows back in the per-neuron algebra: LMK4, LMK6 all ones), noext (_X38 off),
                                   lawon_noext (both: r38 without R38, expected to reproduce r37); noxc, noxa, noxw,
                                   noxf: one of the extraction's four parts off (C', dA, w1, the feed pair); x05: the
                                   whole extraction at half strength
  est_r37/mech_nott, est_r38x/mech_nott_noext, mech_nott_lawon   hooked band-off copies (entry statistics)
  est_r37/mech_prod, est_r38x/mech_prod_noext, mech_prod_lawon     hooked band-on copies (final means, mech_overlap.py)

    .venv-fs0123/bin/python research/wu/mech/make_cuts.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import make_mech as MM  # noqa: E402
import make_variants as MV  # noqa: E402

REPO = HERE.parents[2]
OPR30 = REPO / "estimators" / "wu2"
R37 = REPO / "estimators" / ("k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L"
                             "_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30_r36k256_opL4")
R38 = REPO / "estimators" / "r38" / "r38_sub_x100_s2"
LAWON = [("LMK4 = fnp.array([[1.0], [1.0], [0.0], [0.0]], dtype=f32)  # r38: lam' rows out of the engine's VEC",
          "LMK4 = fnp.array([[1.0], [1.0], [1.0], [1.0]], dtype=f32)  # mech: lam' rows back in the engine's VEC"),
         ("LMK6 = fnp.array([[1.0], [1.0], [0.0], [0.0], [1.0], [1.0]], dtype=f32)",
          "LMK6 = fnp.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], dtype=f32)  # mech: lam' rows back in")]
NOEXT = [("_X38 = {'on': True,", "_X38 = {'on': False,")]
X05 = [("_X38 = {'on': True, 'scale': 1.0,", "_X38 = {'on': True, 'scale': 0.5,")]
# the extraction's four parts, one at a time off (r38 otherwise as submitted): C' (the scale mode out of the activation
# covariance and its rescale), dA (out of the K4 core rows), w1 (out of the newborn's K3 leg), the feed pair
XPART = {
    "noxc": [("                        fnp.subtract(Cv, fnp.outer(m_, fnp.multiply(m_, eps_)), out=Cv)\n"
              "                        mul(Cv, 1.0 / (1.0 + eps_), out=Cv)\n",
              "                        pass  # mech: the extraction's C' part off\n")],
    "noxa": [("                        dA = fnp.subtract(dA, es(\"ia,ia,i->a\", Wn, Wn, kr_))\n",
              "                        pass  # mech: the extraction's dA part off\n")],
    "noxw": [("                        fnp.subtract(w1, fnp.outer(mul(v_, 6.0 * eps_), m_), out=w1)\n",
              "                        pass  # mech: the extraction's w1 part off\n")],
    "noxf": [("                if x38_:  # r38: the extraction's K3 pair (-6 eps m, w11) in the feed slot",
              "                if False:  # mech: the extraction's feed pair off; r38: the extraction's K3 pair (-6 eps m, w11) in the feed slot")],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(out: Path, name: str, src_dir: Path, reps: list, man: dict) -> None:
    s, rec = MM.apply((src_dir / "estimator.py").read_text(encoding="utf-8"), reps)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    # identical files are left untouched: a rerun that adds cuts does not rewrite files a running job may be loading
    if not (d / "estimator.py").exists() or (d / "estimator.py").read_bytes() != s.encode("utf-8"):
        (d / "estimator.py").write_bytes(s.encode("utf-8"))
    if not (d / "wmsg.npz").exists() or sha(d / "wmsg.npz") != sha(src_dir / "wmsg.npz"):
        shutil.copy2(src_dir / "wmsg.npz", d / "wmsg.npz")
    man[f"{out.name}/{name}"] = {"from": str(src_dir.relative_to(REPO)), "from_sha256": sha(src_dir / "estimator.py"),
                                 "replacements": rec, "sha256": sha(d / "estimator.py")}
    print(f"{out.name}/{name}", man[f"{out.name}/{name}"]["sha256"][:16])


def main() -> None:
    man = {"make_variants_sha256": sha(HERE.parent / "make_variants.py"), "make_mech_sha256": sha(HERE / "make_mech.py")}
    hooks = MM.HOOKS[:3]
    eo = HERE / "est_opr30"
    write(eo, "mech_prod", OPR30 / "v_engine", hooks, man)
    write(eo, "mech_nott", OPR30 / "v_engine_nott", hooks, man)
    e7 = HERE / "est_r37"
    write(e7, "v_engine", R37, MV.ENGINE, man)
    write(e7, "v_engine_nott", R37, MV.ENGINE + MV.NOTT, man)
    base = REPO / "estimators" / "r38" / "r38_eng_base" / "estimator.py"
    man["est_r37/v_engine equals estimators/r38/r38_eng_base"] = sha(e7 / "v_engine" / "estimator.py") == sha(base)
    ex = HERE / "est_r38x"
    for mode, reps in (("lawon", LAWON), ("noext", NOEXT), ("lawon_noext", LAWON + NOEXT)):
        write(ex, f"v_engine_{mode}", R38, MV.ENGINE + reps, man)
        write(ex, f"v_engine_nott_{mode}", R38, MV.ENGINE + MV.NOTT + reps, man)
    for mode, reps in XPART.items():
        write(ex, f"v_engine_{mode}", R38, MV.ENGINE + reps, man)
        write(ex, f"v_engine_nott_{mode}", R38, MV.ENGINE + MV.NOTT + reps, man)
    # the whole extraction at half strength (all four parts, eps_t x 0.5; the law off as submitted)
    for mode, reps in (("x05", X05),):
        write(ex, f"v_engine_{mode}", R38, MV.ENGINE + reps, man)
        write(ex, f"v_engine_nott_{mode}", R38, MV.ENGINE + MV.NOTT + reps, man)
    # hooked band-off copies for the entry statistics of r37 and of r38 with one of R38's changes undone
    write(e7, "mech_nott", R37, MV.ENGINE + MV.NOTT + hooks, man)
    write(ex, "mech_nott_noext", R38, MV.ENGINE + MV.NOTT + NOEXT + MM.HOOKS, man)
    write(ex, "mech_nott_lawon", R38, MV.ENGINE + MV.NOTT + LAWON + MM.HOOKS, man)
    # hooked band-on copies (their final means for mech_overlap.py)
    write(e7, "mech_prod", R37, MV.ENGINE + hooks, man)
    write(ex, "mech_prod_noext", R38, MV.ENGINE + NOEXT + MM.HOOKS, man)
    write(ex, "mech_prod_lawon", R38, MV.ENGINE + LAWON + MM.HOOKS, man)
    # r38 without the extraction's dA part, band off and on (the only part-off engine where the band gains)
    write(ex, "mech_nott_noxa", R38, MV.ENGINE + MV.NOTT + XPART["noxa"] + MM.HOOKS, man)
    write(ex, "mech_prod_noxa", R38, MV.ENGINE + XPART["noxa"] + MM.HOOKS, man)
    (HERE / "CUTS.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("r37 v_engine == r38_eng_base:", man["est_r37/v_engine equals estimators/r38/r38_eng_base"])


if __name__ == "__main__":
    main()
