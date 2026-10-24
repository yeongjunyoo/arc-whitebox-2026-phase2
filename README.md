# ARC White-Box Estimation Challenge 2026: submissions 336955 and 337488

A learned corrector that reads the engine's cross-neuron terms, saturation-aware compute and a free fourth-cumulant
fold on factored cumulant propagation at width 1024. Phase 2 submission 336955 by Yeongjun Yoo (AIcrowd `yeongjunyoo`).

The write-up is `WRITEUP.md`, the same text as `WRITEUP.pdf` (the PDF submitted for the algorithmic-contribution prize and posted on the challenge forum). This repository holds the
graded files, the code that trained the corrector, and the harness and raw results behind every number in the
write-up's ablation tables. The paths are those of our working tree, so every script runs as it ran for us.
Our final designated submission, 337488, is 336955 plus three later changes (write-up sections 7.3 and 7.6); its
folder, the shot it was built on and the scripts that made it are here too.

## Contents

| Path | What it is |
|---|---|
| `estimators/k3_fast_aug_sc_2l_..._opr30/` | The graded folder: `estimator.py` (self-contained, every table inline) and `wmsg.npz` (the corrector's weights). `SHA256SUMS` matches the uploaded files. The folder name lists the steps of the trajectory (write-up section 7.5). |
| `estimators/k3_fast_aug_sc_2l_..._opb30a15_r25aqx3/` | The build that dumped the corrector's training features (engine of shots 30-32, sha16 fad376e68b85abff). |
| `estimators/k3_fast_aug_sc_2l_..._opr30_r36k256_opL4/` | Shot 38 (submission 337371): compression rank 256 and one more restriction on top of 336955 (write-up section 7.3); the base of 337488. |
| `estimators/r38/r38_sub_x100_s2/` | Submission 337488 (shot 39): shot 38's file with the memoryless law masked in the engine and the scale-mode extraction added (`research/r38/make_r38.py`), and the corrector retrained on the new engine folded in (`research/r38/build_sub.py`). `SHA256SUMS` matches the uploaded files. |
| `estimators/k3_fast.py`, `k3_simple.py` | Our ports of the reference engine, the starting point (shot 1) and the reference row of the ablation tables. |
| `research/r25a/` | Feature dumps from a folder estimator (`gru_dump.py`; `gru_dump_kp.py` for the public corpus), the trainer (`gru_lean.py`), the distillation teacher (`teacher_export.py`), the validation pick (`v90_grid.py`) and the fold into a folder (`make_r25a_v4.py`). |
| `research/` | What those import: the corrector model and loss (`lc3_wmsg.py`), the cache format and features (`lc3_curve.py`, `lc_chain.py`, `lc_fit.py`), data loading (`k3_measure.py`, `k3_port_check.py`) and the file hash (`estsha.py`). |
| `research/acc_aug_fit.py`, `research/acc_feed_scan.py` and their outputs | The fit of the memoryless law's table `LAM_REFIT` (write-up section 2.2): finite-difference runs over the per-layer scales on public mini 50-52 (`acc_feedscan_aug_foldlam2_refit.json`, `.log`) and the linearized least-squares refits (`acc_aug_fit_aug_foldlam2_jacgrp*_r0.json`). They ran against an earlier research engine that is not included. |
| `research/costfloor/satbins_table.py`, `satdyn_stack_cost.py`, `satbins_table.json`, `research/r25b/bins.py` | How the restricted shapes were sized (write-up section 5): the 99th-percentile kept counts over the 392 dumped public networks, binned per axis. They read the per-network engine dumps in `research/lc3_cache/s6_192` (1.4 GB, not included; the cache format is the one `research/lc3_curve.py` reads). |
| `FEATURES.md` | Every input column of the corrector, its formula and its code location (inference and training), the recurrence, the packed weights in `wmsg.npz`, and its FLOPs. |
| `submissions/shots.csv`, `figures/` | Every Phase 2 submission of this line with its grader values, and the trajectory figure made from it (`figures/make_trajectory.py`). |
| `research/r38/` | Write-up section 7.6: the readable research engine with the extraction options (`k3_acc_tc.py`), its grid runner and tabulator (`grid.py`, `tab.py`) with the raw grids (`out/`), the generator of the production files (`make_r38.py`), the fold of a trained corrector into them (`build_sub.py`) and the feature-dump driver (`r38_dump.sh`). |
| `research/wu/mech/` | Write-up section 5, what the 3T^T band's MSE gain is: Monte Carlo truth of the third cumulant on public mini 0-3 (`mc_truth.py`), hooked copies of the engines (`make_mech.py`, `make_cuts.py`), the oracle swaps and band on / off runs (`mech_run.py`, `run_*.sh`), the analysis (`mech_report.py`, `mech_decomp.py`, `mech_overlap.py`) and `README.md` with every table. The recorded arrays (several GB) are not included; the scripts rebuild them. |
| `research/wu/` | The write-up's ablations: `make_variants.py` (each variant is the submitted file with exact string replacements, listed with hashes in `estimators/wu2/VARIANTS.json`), `ablate.py` (one prediction per network under the kit's budget context), `paired.py` (paired ratios with a bootstrap over networks), `kp_floor.py` and `kp_floor.json` (the corpus's Monte Carlo floor), `report.py` (the tables), `ladder.py` (the corrector retrains), the shell drivers we ran, and the raw per-network results. |

## Requirements

Scoring and ablations: Python 3.11, numpy 2.4.4, flopscope 0.12.3 (the grader's version), whestbench 0.16.2, pyarrow
24.0.0. Corrector training: Python 3.11, torch 2.14.1, numpy 2.4.6. The feature dumps used flopscope 0.12.1, which
gives the same predictions. Everything ran on CPU (an Apple M3 Pro, one BLAS thread per job).

## Running the submitted estimator

`estimator.py` follows the starter kit's interface (`setup()`, then `predict(mlp, budget)`), so the kit's local tools
score it as they score any submission. To score public networks the way the write-up does:

    python research/wu/ablate.py --estimator estimators/<graded folder> --set mini --ids 0,1,2 \
        --snapshot <local snapshot of the HF dataset aicrowd/arc-whestbench-public-2026> --out mini.jsonl

Each output line has the network, final-layer MSE against the dataset's Monte Carlo means, FLOPs, the adjusted score
and the wall time.

## Reproducing the corrector

1. Dump the engine's features on the 400 training networks (write-up Appendix A) with the dump build:
   `research/r25a/gru_dump.py --est estimators/<dump build> --tags <tags> --threads 1 --out <dir>`.
2. Train twelve members, seeds 1-12: `research/r25a/gru_lean.py train --dir <dir> --tag e60s<seed>
   --qx 0,1,2,4,5,6,7,8,9,10,11 --wd 1e-4 --epochs 60 --seed <seed> --no-s`.
3. Average them on the training networks: `research/r25a/teacher_export.py --dir <dir> --models <the twelve npz>
   --out <dir>/teacher_e60`.
4. Train nine students (beta 0.5, 0.8, 1.0; seeds 1-3): the member command with `--teacher <dir>/teacher_e60
   --beta <beta>`. The shipped weights are beta 0.5, seed 3, picked on the 90 validation networks with
   `research/r25a/v90_grid.py`.
5. Fold the pick into a folder with `research/r25a/make_r25a_v4.py`.

For 337488 the corrector is a single model on the new engine: dump the 406-network pool with the r38 build
(`research/r38/r38_dump.sh`), train seeds 1-3 with the member command of step 2 but `--qx 0,1,2,3,4,5,6,7,8,9,10`
(the r38 dumps store the eleven taps in rows 0-10), keep the seed with the lowest validation ratio on mini 44-49, and
fold it with `research/r38/build_sub.py <weights npz> <folder name>`.

## Reproducing the ablation tables

`research/wu/run_kp.sh`, `run_nosat.sh`, `run_attr.sh`, `run_taps.sh`, `run_shape.sh`, `run_pub.sh` and `run_pub_shape.sh` are the drivers we ran for
the tables of sections 5, 6.3, 7.1 and 7.2; they wrap our job scheduler (not included), and each job is one
`ablate.py` call. `run_ladder.sh` ran the corrector retrains of section 4.5 (`ladder.py train`, then one
`ladder.py eval`; its output is `ladder_<date>.json`, per variant, seed and network). `research/wu/report.py --dir <results> [--floor
research/wu/kp_floor.json]` rebuilds a table from the raw `.jsonl` results.

## Lineage and AI use

The engine's skeleton is a port of ARC's reference implementation (`mlp_cumulant_propagation`, MIT). Borrowed ideas
are credited in the write-up's section 10. No code from another participant was read or used. The work was done with
AI assistance, as described in the write-up's section 10.

## License

MIT (see `LICENSE`, which also carries the notices of the reference implementation and the starter kit).
