# R25-B operating-point variant L4 (research/r25b/make_op.py, Nova 2026-10-15): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30_r36k256; corrector weights unchanged (wmsg.npz copied).
#   L4: the fold-layer push inner restriction (s1) from layer 4 (was 6), bins for the new layers
# R36 k256 (research/r36/make_r36.py, Nova 2026-10-20): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30; compression age 6 -> 6, rank 192 -> 256 (one of them), nothing else.
# R30-B cuts q0,q12,lr6,w4p (research/r30b/make_r30b.py, Nova 2026-10-18): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs
#   q0: tap 0 as k4f's DGF / (a SV3) (direct form at the last layer or a zero lam)
#   q12: the gf fold's increment reuses the taps' G o H (bit-equal order)
#   lr6: LR only on the 126 of 288 (y, x, r, k) rows the window boxes read
#   w4p: the corrector's W^4 neighbour sums without their three zero padding vectors
# R28-B cost cuts slcs (research/r28b/make_r28b.py, Nova 2026-10-18): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA
#   sl: k3_slices' e1c chunk views cut to the live slots (unborn slots computed zeros nothing read)
#   cs: covsym's diagonal quarter blocks from their upper and two diagonal eighth blocks (12 batched, depth 4)
# R25-B kc variant A (research/r25b/make_kc.py, Nova 2026-10-17): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175
#   kc: the compression SVD on the rows dc kept and the columns the next push reads; tap 6 on the kept columns
# R25-B operating-point variant i175 (research/r25b/make_op.py, Nova 2026-10-15): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3; corrector weights unchanged (wmsg.npz copied).
#   i175: push inner threshold alive 1.75 (was 2.5): s1 bins, s8 pairs, s2 pairs recomputed (bins.py k = 2)
# R25-A build v4 (research/r25a/make_r25a_v4.py, Nova 2026-10-15, ARC R25 lane A): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15 (estimator.py sha16 d2dcad2c26e9d16e), corrector lc3_wmsg_kd05s3.npz with taps [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11].
#   v4: v2a's taps 0-2 and the kept qx2 taps (research/r25a/make_r25a_qx2.py) written to QX rows 0..kq-1, the
#   corrector retrained by research/r25a/gru_lean.py on the gA pool (S and held-out excluded, mini 44-49 for early
#   stopping) and folded here. Ours; no participant's code was read.
# R25-B operating-point variant b30a15 (research/r25b/make_op.py, Nova 2026-10-15): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a; corrector v2 weights unchanged.
#   b30: 3 T^T column band and the last layer's band |a| < 3.0 (was 3.5), bins and pairs recomputed
#   a15: 3 T^T rows alive 1.5 (was 2.5), bins recomputed
# R25-A build v2 (research/r25a/make_r25a_v2.py, Nova 2026-10-15, ARC R25 lane A): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx (estimator.py sha16 1d07b10775521b59), corrector v2 lc3_wmsg_v2a.npz with 3 taps.
#   v2: the corrector also reads arcsinh(QX / c) of the engine's fold taps (research/r25a/make_r25a_qx.py),
#   trained by research/r25a/gru_lean.py on the gA pool (S and held-out excluded, mini 44-49 for early
#   stopping) and folded here. Ours; no participant's code was read.
# R25-A build qx (research/r25a/make_r25a_qx.py, Nova 2026-10-15, ARC R25 lane A): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05 (estimator.py sha16 aee608671f9ad120).
#   qx: corrector v2 input taps QX (star K4 diagonal, the G fold's two diagonal shapes, the rank-1 norm of its
#   increment), filled per layer; the corrector here is still gA. _QX['on'] False is the base, bit for bit. Ours;
#   no participant's code was read.
# R25-A build gf (research/r25a/make_r25a_gf.py, Nova 2026-10-15, ARC R25 lane A): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x (estimator.py sha16 6f2e8e4f0140d197).
#   gf: the G-slot part of the K4 star's supply to K3 (w31 Sym3(U, h, h), U = w21 o G o H) folded into the dense
#   G legs at every layer before the w11 step: G += scale lam (w21 / w11) G o H, lam = -alpha / sigma of the
#   source's birth layer. Elementwise only. _GF['scale'] None is the base, bit for bit. Ours; no participant's
#   code was read.
# R25-B build r25bc3e7x (research/r25b/make_r25b.py --final, Nova 2026-10-14, ARC R25 cost lane): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces.
#   s1: the fold layers' (6-10) leg rows run over layer t-1's alive top-Pin inner columns (alpha > -2.5, padded to the
#       depth-6 multiple and rounded up to the bins), C's n rows in full (the covariance sandwich stays exact)
#   s8: layers 11-14: the push's inner size (alive 2.5 at t-1) and output size (alive 3.5 at t) apart instead of
#       one square max of the two (bins of pairs)
#   s2: the last layer: C' rows pushed alone in full (var of every neuron), then a, then the leg rows over the
#       alive 2.5 inner columns of layer 14 and only the band |a| < 3.5 of layer 15 (plain depth 6); t3 outside the band is 0
#   rb: the last layer's compressed legs rebuilt on the band columns only
#   s3: the last compression (the source born at depth - 2 - age) is skipped; it stays dense for the last layer
#   s7: layers whose 3 T^T is band-restricted: the compressed sources' legs rebuilt and their P made on the band
#       columns only (P and their thin A rows zero outside the band; exact frames per band size)
#   esc: lane C's six per-box matrices einsums take the explicit path [(1, 3), (1, 2), (0, 1, 2)] (L and R rows
#        selected first, then one three-operand step): same bill (flopscope's accumulation total is path-
#        independent), no box-sized pairwise intermediates (16 to 216 MB under 'auto')
#   ax: every APA product shape gets its own Strassen-Winograd frames carved from the arena (no cut leaf
#       blocks): the same ops, shapes, bills and arithmetic per element
# Not bit for bit the base. Base header follows.
# Lane C R25 (research/r25c/resid/make_r25c_es.py, Nova 2026-10-14): the matrices einsum split by the E2 boxes
# (research/costfloor/make_esplit_estimator.py, 2026-10-07): 63 of the 144 (x, r, k) slices billed instead of all.
# Base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA (folder sha16 c5c9cd415344a501). Not bit for bit the base (the sum over k and r is regrouped).
# R25-A build (research/r25a/make_r25a.py, Nova 2026-10-14, ARC R25 lane A): base
# k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv (sha16 e463936ca5fba2c0).
#   k4f: the K4 star (3,1) slice that every K3 source carries on its own two legs (born with weight w31 of its birth
#   layer, sum_m w31_m Sym(G_m (x) H_m^3)) is folded into the 3 T^T product: its A rows get 3 s w31 lam G H^2 (AH) and
#   s w31 lam H^3 (AG), lam = w31 / w21 = -alpha / sigma of the layer, so C' gains (1/6)(w31_a w11_b K31_ab +
#   w11_a w31_b K31_ba) through its T diagrams; 3T^T's diagonal is put back to 3 t3. _K4F['scale'] None is the base,
#   bit for bit. Ours; no participant's code was read.
# R24 build wc10 (research/r23x/make_r24.py, Nova 2026-10-13, slot B last bet): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv.
#   wc: compression SVD rows weighted by (birth atom T mass / row norm)^1.0, floor 0.02
# R143-B build bcxa8tv (research/r143/b/make_b.py, Nova 2026-10-12, slot B upside mandate): base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x.
#   norecarve: predict no longer re-carves the setup plan's frames to bill their reshapes
#   covsym: S0 from its upper half block and six batched quarter blocks, mirrored (depths 5, 5)
#   covsw: the non-fold layers' S0 = W^T (C W) by Strassen-Winograd C W and covsym; the first layer's W^T W by covsym
#   apa: one Bini <3,2,2;10> level (eps 2^-8) on top of tt, push products without covariance rows
#   depth: sw_depth 5, sw_depth_batched 5, sw_hi_push (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15), sw_hi_tt (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
# Not bit for bit the base.
# Dead-center compact build (research/r10_structural/make_dc_sb_estimator.py, Nova 2026-10-08; make_dc_cf_estimator.py
# with the sat-bins plan hook): every source born
# at layer t >= 1 keeps its top-R centers by that layer's alpha (R: fixed p99 frame of alpha > -theta, a multiple
# of max(2^sw_depth, 32)); the dropped centers carry w11, w21 ~ 0, so their atoms are dropped; the push (fold
# layers too) and the 3 T^T run on the gathered kept rows. _DC['theta'] = None is
# k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2 bit for bit. Base header follows.
# Per-MLP saturation frames, sat mode bins (research/costfloor/make_satbins_estimator.py, Nova 2026-10-08, k = 2); base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p. Not bit for bit the base.
# Depth variant d6p (research/costfloor/make_depth_variant.py, Nova 2026-10-07): push at depth 6 on layers 1-10 and 15;
# base k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp. Not bit for bit the base.
# Setup plan (research/costfloor/make_setupplan_estimator.py, Nova 2026-10-07): the Strassen-Winograd arena, frames
# and every view the calls derive are made once in setup() and reused by every predict; predict re-carves the
# frames (billed as before), so ops and FLOPs in predict are unchanged.
# Round-trip trims (research/costfloor/make_vcache_estimator.py, Nova 2026-10-07): views of uncut Strassen frames
# memoized per plan, VX row views from unstacks made once; same ops and FLOPs as before this step.
# Covariance fold (research/costfloor/make_covfold_estimator.py, Nova 2026-10-07): on layers with an unrestricted
# push, C rides the push (T = C W from the same Strassen product) and S0 = W^T T is one more Strassen-Winograd
# product, symmetrized, instead of the 3 n^3 einsum; other layers unchanged. Not bit for bit the base. Base follows.
# Recursive-layout Strassen-Winograd (research/costfloor/make_rsw_estimator.py, Nova 2026-10-07): each product
# kind's work frames are one array per side whose axes hold the 7 children of every level and the 4 quadrants
# below them, so a level's Winograd sums write the children in place: no per-level copies of A's quadrants (B keeps
# one), no per-level frame cuts or quadrant slices, the same products, sums and combination order (predictions
# expected bit for bit the base's), fewer billed FLOPs and client round trips. Base header follows.
# Saturation-aware restriction on the exact twin k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35 (research/costfloor/make_sat_estimator.py, Nova
# 2026-10-06; idea ARC R7): {"in": 2.5, "out": 4.0, "ta": 2.5, "tb": 3.5, "tb_band": true, "mode": "fixed", "q": 32, "fbuf": true, "e1c": 0, "e1cv": 8}
# Exact-saving package (research/costfloor/make_ex_estimator.py, Nova 2026-10-06): shot 11
# (k3_fast_aug_sc_2l_b_lc2k_rd_wm, da43671b0f6c157a) computing the same quantities with fewer billed FLOPs:
#   E3 the last layer's variance from C' pushed with the legs (float-equal)
#   E5 the VEC rows over the 49 monomials of their table (float-equal)
# The learned corrector was retrained on this engine (R25-A, Nova 2026-10-14, research/r25a/gru_lean.py, model A: the same model and settings as shot 11's h32_nb_nosoffd_n400_s1p12, trained on the 400 MLPs of shot 11's pool (full shards 6-18 and 22-35, mini 0-43 and 51-58), early stopping on mini 44-49; inputs dumped from this folder's engine by research/r25a/gru_dump.py), wmsg.npz folded by research/gru_build/make_wmsg_npz.py. The corrector code is shot 11's, unchanged.
# Attribution: this estimator re-implements, in flopscope, the factored k_max=3 cumulant propagation of ARC's
# reference implementation
#   alignment-research-center/mlp_cumulant_propagation, commit 93d091a
#   (src/mlp_kprop/kprop_harmonic.py, factor_k3.py, wick.py, harmonic.py, diagslice.py, cumulants.py),
#   MIT License, Copyright (c) 2026 Alignment Research Center,
# and the companion paper arXiv:2605.05179: the "SIMPLE" term set (our port estimators/k3_simple.py, regrouped into
# few operations in estimators/k3_fast.py, whose engine is reused verbatim below for other shapes) and, at the scored
# shape, the "AUGMENT" term set with the activation fourth-cumulant core's off-diagonal replaced by a one-scalar-per-
# layer law (our approximation, estimators/k3_acc_aug.py, research/acc_report_2026-10-04.md). The idea of a
# one-scalar-per-layer law for that off-diagonal follows the text of AIcrowd forum post 18218 ("memoryless kappa4
# regeneration"); no code from that post or its repository was read. The activation-side form, the exact diagonal,
# the heavy-leg fold, the lam' table and this implementation are ours. Each source is carried in a 2-leg form (its E leg
# written as G diag(xv) + H diag(e) + a rank-16 residual Z Lz^T, so only H and G are dense); the idea of writing the
# third leg through the other two plus a thin residual is credited to team 504aldo, AIcrowd forum post 18218, section
# 2.1 ("M = P d(s) + 3 A d(e) + Z L^T"), taken from the post's prose only; no code from that post or its repository
# was read, copied or adapted, and the form in our leg algebra, the least-squares e and the randomized residual factor
# are ours. No other participant's code was read or used.
#
# Learned mean correction (this file only): a second, corrected mean is carried next to the estimator. At every
# layer t >= 1, a'_t = a_t + W_t^T (p'_{t-1} - p_{t-1}) / sigma_t and p'_t = readout(a'_t) + net_t(features), where
# readout is this estimator's own pK1 and net_t a small ReLU network on per-neuron quantities the estimator already
# has (a, sigma, the t3 and K4 terms of the readout and their higher Edgeworth companions; at layers 13-15 also
# covariance statistics). All cumulants stay the base estimator's; the last layer returns p'_15. The networks were
# fitted offline on public mini MLPs 0-39 (each layer's local error against the dataset's Monte Carlo means, the
# incoming mean set to the truth), chosen on 40-49 and reported once on 50-59 (research/lc_chain.py,
# research/lc_fit.py). The idea of learned networks that consume cumulant estimates as features is credited to Jacob
# Hilton's public LessWrong comment on ARC's challenge announcement (warm-up leaders' estimators "combined with
# learned networks that consume the cumulant estimates as features"); the targets, features, chain, training and
# weights are ours, and no participant's code was read.
#
# Per-neuron K4 law (this file only): at every layer t >= 1 the K4 law rows lam'_t, lam'_{t+1} and the dA row enter the
# per-neuron algebra scaled per neuron by 1 + c1 a + c2 a^2 (a: the layer's standardized mean; 6 constants fitted
# offline by finite differences of the final-layer error on public mini MLPs 20-23, research/r2_law.py); ours, no
# participant's code was read.
#
# MIT License text of the reference (applies to the portions derived from it):
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#   associated documentation files (the "Software"), to deal in the Software without restriction,
#   including without limitation the rights to use, copy, modify, merge, publish, distribute,
#   sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions: The above copyright notice and this
#   permission notice shall be included in all copies or substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#   NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
#   OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Generated by research/acc_make_fast_aug.py (the SIMPLE engine is copied from estimators/k3_fast.py), then
# research/make_aug_sc.py (compression, Strassen-Winograd) and research/make_aug_sc_2l.py (2-leg sources).
# Settings variant written by research/make_2l_variant.py: age=6,rank=192,sw_depth=5,sw_depth_batched=2,kz=4 (code identical to k3_fast_aug_sc_2l.py).
# Learned correction added by research/make_lc.py (pick split_mlp_base_1_12__mlp_rich_h32_13_15, weights from
# research/lc_chain_models/; this file is k3_fast_aug_sc_2l_b.py when Estimator(lc=False)).
# research/make_lc1fix.py (round 2): the correction's constant tables and W^2..W^4 buffer are created in setup();
# predictions are bit-identical to k3_fast_aug_sc_2l_b_lc.py (sha16 024f5caca49bae58).
# research/make_lc2k.py (round 2, direction 3): the per-neuron K4 law above; this file is k3_fast_aug_sc_2l_b_lc2.py
# (sha16 31c96324a0e5df00) when Estimator(law=False).
# Strassen-Winograd memory and per-layer depth written by research/make_swp.py (on research/make_sw6.py's in-place
# deepest level, sw_inplace: the deepest Winograd sums overwrite the quadrants they come from between the leaf
# products, same op calls and FLOPs, smaller work arena); with sw_depth_batched=2, sw_hi_tt=(4, 5, 6, 7, 8) and
# sw_hi_batched=(10, ..., 15) this file computes k3_fast_aug_sc_2l_b_lc2k_swp.py (sha16 c297a5668f70d53d).
# Residual diet (research/2026-10-05-residual-diet note, Nova 2026-10-05): the plumbing of the Strassen-Winograd
# calls is rewritten for fewer client round trips per predict (at the source's settings 10,790 to 8,284): the work
# frames carry one axis of 7 per level, so the 7 children of a level are one fnp.unstack (0 FLOPs) instead of 7
# slices; each call makes its views side by side before its arithmetic (_SwPlan; or once per worker in setup() with
# sw_setup_plan=True, which keeps the work arena between MLPs); per-source views of the engine come from one unstack
# each. The ops and their operands are unchanged: at the same settings the predictions are bit for bit and the FLOPs
# equal to k3_fast_aug_sc_2l_b_lc2k.py and k3_fast_aug_sc_2l_b_lc2k_swp.py (public mini MLP 0, n = 1024).
# Defaults here: the 3 T^T at depth sw_depth + 1 = 6 at layers (4, 5, 6, 7, 8) (sw_hi_tt), rebuild and P at
# sw_depth_batched = 4 at every layer, the push at sw_depth = 5, in place. Estimator(sw_depth=5,
# sw_depth_batched=2, sw_inplace=False, sw_hi_push=(), sw_hi_tt=(), sw_hi_batched=()) is k3_fast_aug_sc_2l_b_lc2k.py.
# W-message recurrent corrector (this file only, Nova 2026-10-05): the final mean is the base estimator's p_15 plus
# dp_15 of the chain dp_t = D_t (W_t^T dp_{t-1}) + u_t, with D_t = w11 + w41 t3 / 6 + w51 d4 / 24 (the derivative of this
# estimator's own readout) and u_t = 1e-4 sigma_t head(H_t), H_t = GRUCell(x_t, W_t^T H_{t-1} / sqrt 2). x_t holds 14
# per-neuron features of the estimator's own quantities, 16 neighbour features (the previous layer's ReLU cumulants and
# readout terms summed through W o W and W^3 .. W^6, column norms), the depth, and the messages W_t^T dp_{t-1} / sigma_t
# and (W_t o W_t)^T |dp_{t-1}| / sigma_t. Hidden size 32, trained offline (research/lc3_wmsg.py, model
# h32_nb_nosoffd_n400_s1p12) on 400 public MLPs (full shards 6-18 and 22-35, mini 0-43 and 51-58; early stopping on mini
# 44-49), the trained arrays folded into wmsg.npz next to this file by research/gru_build/make_wmsg_npz.py. It was
# trained on the engine without the per-neuron K4 law and replaces both the learned mean correction and the law (on
# public full MLPs 0-47 the law under the corrector does not help, nor does lc's mean next to it; Nova's
# research/2026-10-05-gru-build note). Estimator(gru=False) computes k3_fast_aug_sc_2l_b_lc2k_rd.py, sha16
# 74144c1f68d775a1, bit for bit. The idea of a GRU corrector whose input carries the previous layer's correction through
# W and W o W, trained downstream-aware, is credited to mliston's public Phase 1 write-up (AIcrowd discourse 18170), taken
# from its prose only; no code from that post or from any participant's repository was read. The features, the chain,
# the training and this implementation are ours.
"""Factored K=3 cumulant propagation, augmented (law-based) at the scored shape, in few remote op calls, 2-leg sources.

At (n, depth) = (1024, 16) this computes the estimate of estimators/k3_fast_aug_sc.py (k3_acc_aug preset
"recommended" with old sources compressed to rank 320 at age 4 and Strassen-Winograd products) with one approximation:
every third-cumulant source keeps only two dense legs. Its E leg is written at birth as

    E = G diag(xv) + H diag(e) + Z Lz^T

(xv: E's birth diagonal; e: per birth neuron, the least-squares coefficient of E's off-diagonal part on H; Z Lz^T:
rank-kz factor of the residual, kz = 16, from a randomized range finder whose test matrix is the next layer's first kq
weight columns). All legs are transported by the same left products, so the identity holds at every later layer: the
push, the 3 T^T contraction, the rebuild and the blocks P of compressed sources and the compression SVD run over
[H, G] (two n x n legs per source instead of three), and Z travels as kz thin rows next to the source's feed row u.

Layout (k3_fast_aug_sc's transposed source slab, 2-leg):
* LEGt: per source s, TH = 1 + kz thin rows [u_s; Z_s^T] (rows [0, S TH)); then blocks (2 s + x) n + j for legs
  x = [H, G], two more blocks (the last birth window) and a block of ones. ASt pairs with LEGt row by row: thin rows
  [r_s; Lz_s^T (G*G)] and blocks [AH, AG] = [3 A1 + diag(e) 3 A3, 3 A2 + diag(xv) 3 A3] with the feed added.
* The birth window [H'^T, E, C', R4] sits in [H_s, G_s, H_{s+1}, G_{s+1}]: E is split into (xv, e, Lz, Z) right away
  and its slot later receives G = W; C' waits for the next covariance step in the next source's H slot.
* Per-neuron algebra, the K4 law and the matrices einsum are those of k3_fast_aug_sc (see its docstring).
"""

from __future__ import annotations

import contextlib
import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

try:  # analysis-only API: the remote flopscope client used by the grader raises on the attribute access itself
    _ns = flops.namespace
except AttributeError:
    def _ns(name):
        return contextlib.nullcontext()

_G2 = 4.0  # g^2 with g = mlp.init_scale = 2.0, the reference's average metric E[W W^T] for He-init ReLU layers

# ----------------------------------------------------------------------------------------------------------------
# Coefficient tables (pure Python; the grader has no numpy)
# ----------------------------------------------------------------------------------------------------------------
_EXP = (-3, -2, -1, 0, 1, 2, 3, 4)  # sigma exponents in SV
_NC, _NP, _NE = 3, 5, len(_EXP)  # F = [phi, Phi, 1]; a^0..a^4; sigma powers
_PH, _PHI, _ONE = 0, 1, 2


def _add(*parts):
    out = {}
    for s, d in parts:
        for key, v in d.items():
            out[key] = out.get(key, 0.0) + s * v
    return out


def _poly(*monos):  # monos: (coef, F row, power of a, sigma exponent); keys (z, w, c, p, e)
    out = {}
    for coef, c, p, e in monos:
        key = (0, 0, c, p, _EXP.index(e))
        out[key] = out.get(key, 0.0) + float(coef)
    return out


def _t3(d, s):  # s * t3 * d, t3 = (tA + tB) / 3
    out = {}
    for (z, w, c, p, e), v in d.items():
        for zz in (1, 2):
            out[(zz, w, c, p, e)] = out.get((zz, w, c, p, e), 0.0) + s * v / 3.0
    return out


def _c4(d, s):  # s * c4 * d (w = 1 marks "times c4"; expanded with c4 = cc (lin - 2 sp11) when the table is built)
    return {(z, 1, c, p, e): s * v for (z, w, c, p, e), v in d.items()}


# Wick coefficients w_kp = E[d^k ReLU(Z)^p], Z ~ N(m, sigma^2), a = m / sigma (as in k3_simple)
_W01 = _poly((1, _PH, 0, 1), (1, _PHI, 1, 1))
_W11 = _poly((1, _PHI, 0, 0))
_W21 = _poly((1, _PH, 0, -1))
_W02 = _poly((1, _PH, 1, 2), (1, _PHI, 2, 2), (1, _PHI, 0, 2))
_W31 = _poly((-1, _PH, 1, -2))
_W41 = _poly((1, _PH, 2, -3), (-1, _PH, 0, -3))
_W03 = _poly((1, _PH, 2, 3), (2, _PH, 0, 3), (1, _PHI, 3, 3), (3, _PHI, 1, 3))
_W04 = _poly((1, _PH, 3, 4), (5, _PH, 1, 4), (1, _PHI, 4, 4), (6, _PHI, 2, 4), (3, _PHI, 0, 4))
_ONE_T = _poly((1, _ONE, 0, 0))

# Pext rows: 1, pK1..pK4, u = w11, v = w21, r = w12, t3, k4o (k4d = G2 c4, k4o = G2 c4 / 3)
_PEXT = [
    _ONE_T,
    _add((1, _W01), (1, _t3(_W31, 1 / 6)), (1, _c4(_W41, _G2 / 24))),
    _add((1, _W02), (1, _t3(_W21, 1 / 3)), (1, _c4(_W31, _G2 / 12))),
    _add((1, _W03), (1, _t3(_W11, 1.0)), (1, _c4(_W21, _G2 / 4))),
    _add((1, _W04), (1, _t3(_W01, 4.0)), (1, _c4(_W11, _G2))),
    _W11,
    _W21,
    _add((2.0, _W01)),
    _t3(_ONE_T, 1.0),
    _c4(_ONE_T, _G2 / 3),
]
_NPX = len(_PEXT)
_P_U, _P_V = 5, 6
# Cum2: quadratic in Pext (pairs of Pext rows; Pext row 0 is 1)
_C2 = [{(0, k): 1.0} for k in range(_NPX)] + [
    {(1, 1): 1.0},  # 10 pK1^2
    {(1, 2): 1.0},  # 11 pK1 pK2
    {(1, 3): 1.0},  # 12 pK1 pK3
    {(2, 2): 1.0},  # 13 pK2^2
    {(5, 5): 1.0},  # 14 u^2
    {(5, 8): 1.0},  # 15 u t3
    {(5, 1): 1.0},  # 16 u pK1
    {(6, 1): 1.0},  # 17 v pK1
    {(9, 5): 1.0},  # 18 k4o u
    {(9, 6): 1.0},  # 19 k4o v
]
# The vectors the matrix step needs, each quadratic in Cum2 (so of degree <= 4 in Pext)
_C4 = [
    {(0, 0): 1.0},  # 0 one
    {(0, 5): 1.0},  # 1 u
    {(0, 6): 1.0},  # 2 v
    {(0, 7): 1.0},  # 3 r
    {(0, 1): 1.0},  # 4 pK1
    {(0, 16): 1.0},  # 5 u pK1
    {(0, 17): 1.0},  # 6 v pK1
    {(0, 14): 1.0},  # 7 u^2
    {(0, 18): 1.0},  # 8 k4o u
    {(0, 19): 1.0},  # 9 k4o v
    {(19, 1): 1.0},  # 10 k4o v pK1
    {(0, 2): 1.0, (0, 10): -1.0},  # 11 var_out = pK2 - pK1^2
    {(0, 3): 1.0, (0, 11): -3.0, (1, 10): 2.0, (14, 15): -1.0},  # 12 x = K3_3 - u^3 t3
    {(0, 4): 1.0, (0, 12): -4.0, (0, 13): -1.0, (10, 2): 8.0, (10, 10): -4.0},  # 13 K4_4 + 2 var_out^2
]
_ONE4, _U, _V, _R, _PK1, _UPK, _VPK, _UU, _K4U, _K4V, _K4VPK, _VAR, _X, _K4ROW = range(14)
# L/R vectors over the stack M = [S0, 3T^T, 3T, Q, J, I] (slots k = 0..5).
# y = 0, 1: L, R of the main einsum, outputs x = [H, E, C']; y = 2, 3: L, R of the c4 sums, terms
# z = [sum pK22, -4 sum pK21_ij pK1_j, +4 pK1' pK11 pK1] (+ sum_i (K4_4 + 2 var_out^2) in the I slot).
_LR = {
    0: {(0, 0): {_U: 1.0},
        (1, 0): {_U: 3.0}, (1, 1): {_U: 1.0}, (1, 2): {_V: 1.0}, (1, 3): {_V: 3.0}, (1, 4): {_V: 3.0}, (1, 5): {_X: 1.0},
        (2, 0): {_U: 1.0}, (2, 1): {_U: 1 / 6}, (2, 2): {_V: 1 / 6}, (2, 3): {_V: 0.5}, (2, 4): {_K4V: 0.25},
        (2, 5): {_VAR: 1.0}},
    1: {(0, 0): {_ONE4: 1.0},
        (1, 0): {_R: 1.0, _UPK: -2.0}, (1, 1): {_U: 1.0, _VPK: -1.0, _UU: -1.0}, (1, 2): {_R: 0.5, _UPK: -1.0},
        (1, 3): {_U: 1.0, _VPK: -1.0, _UU: -1.0}, (1, 4): {_K4U: 0.5, _K4VPK: -0.5}, (1, 5): {_ONE4: 1.0},
        (2, 0): {_U: 1.0}, (2, 1): {_V: 1.0}, (2, 2): {_U: 1.0}, (2, 3): {_V: 1.0}, (2, 4): {_V: 1.0},
        (2, 5): {_ONE4: 1.0}},
    2: {(0, 0): {_R: 1.0}, (0, 1): {_R: 1 / 6}, (0, 2): {_U: 1 / 3}, (0, 3): {_U: 1.0}, (0, 4): {_K4U: 0.5},
        (0, 5): {_K4ROW: 1.0},
        (1, 0): {_R: -4.0}, (1, 1): {_R: -2 / 3}, (1, 2): {_U: -4 / 3}, (1, 3): {_U: -4.0}, (1, 4): {_K4U: -2.0},
        (2, 0): {_UPK: 4.0}, (2, 1): {_UPK: 2 / 3}, (2, 2): {_VPK: 2 / 3}, (2, 3): {_VPK: 2.0}, (2, 4): {_K4VPK: 1.0}},
    3: {(0, 0): {_R: 1.0}, (0, 1): {_U: 2.0}, (0, 2): {_R: 1.0}, (0, 3): {_U: 2.0}, (0, 4): {_U: 2.0},
        (0, 5): {_ONE4: 1.0},
        (1, 0): {_UPK: 1.0}, (1, 1): {_VPK: 1.0}, (1, 2): {_UPK: 1.0}, (1, 3): {_VPK: 1.0}, (1, 4): {_VPK: 1.0},
        (2, 0): {_UPK: 1.0}, (2, 1): {_VPK: 1.0}, (2, 2): {_UPK: 1.0}, (2, 3): {_VPK: 1.0}, (2, 4): {_VPK: 1.0}},
}


def _dtname(dt) -> str:
    s = str(dt)
    return "float64" if "64" in s else ("float32" if "32" in s else s)


def _zeros(*shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros(*shape[1:]) for _ in range(shape[0])]


def _ptable(n: int):
    cc = 3.0 / (n * (n + 2))  # c4 = 3 (sum K4_4 + sum_{i != j} K4_22) / (n (n + 2))
    P = _zeros(_NPX, 3, 3, _NC, _NP, _NE)
    for r, d in enumerate(_PEXT):
        for (z, w, c, p, e), v in d.items():
            if w == 1:
                P[r][z][1][c][p][e] += v * cc
                P[r][z][2][c][p][e] += -2.0 * v * cc
            else:
                P[r][z][0][c][p][e] += v
    return P


def _tables():
    C2 = _zeros(len(_C2), _NPX, _NPX)
    for r, d in enumerate(_C2):
        for (a, b), v in d.items():
            C2[r][a][b] += v
    LR = _zeros(4, 3, 6, len(_C2), len(_C2))  # L/R vectors as quadratic forms of Cum2
    for y, entries in _LR.items():
        for (x, k), d in entries.items():
            for row, v in d.items():
                for (a, b), w in _C4[row].items():
                    LR[y][x][k][a][b] += v * w
    return C2, LR


_TABLES = _tables()


# ----------------------------------------------------------------------------------------------------------------
# Strassen-Winograd products (breadth first, on preallocated work slabs)
# ----------------------------------------------------------------------------------------------------------------
def _layout(d: int, batch: tuple, a_dims: tuple, b_dims: tuple, inplace: bool = True, leaf_split: bool = False):
    """Work frames of one d-level product kind in the recursive layout (research/costfloor/make_rsw_estimator.py):
    every side is ONE array whose axis l (l < d - 1) holds the 7 children of the level-l subproblems and whose last
    lead axis holds the 4 quadrants of the level d - 1 subproblems; a child's own quadrants are entries 0..3 of the
    next axis (A in Z order [11, 12, 21, 22], B in N order [11, 21, 12, 22], the products in Z order), so a level's
    Winograd sums write its children in place. Frames: A (7,) * (d - 1) + (4,) + batch + (m, k) >> d, B the same with
    (k, n) >> d, M with (m, n) >> d, the three leaf product buffers L (3,) + (7,) * (d - 1) + batch + (m, n) >> d and
    the leaf's spare B quadrant T (7,) * (d - 1) + batch + (k, n) >> d. inplace and leaf_split are the earlier form's
    switches (this form is always in place)."""
    (m, k), (_, nn) = a_dims, b_dims
    s7, bt = (7,) * (d - 1), tuple(batch)
    md, kd, nd = m >> d, k >> d, nn >> d
    return {"d": d, "A": s7 + (4,) + bt + (md, kd), "B": s7 + (4,) + bt + (kd, nd), "M": s7 + (4,) + bt + (md, nd),
            "L": (3,) + s7 + bt + (md, nd), "T": s7 + bt + (kd, nd)}


_RSW_FRAMES = ("A", "B", "M", "L", "T")


def _size(shape) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _regions(lay) -> dict:
    """Frame sizes (elements) of a layout; its frames sit side by side from offset 0."""
    return {name: _size(lay[name]) for name in _RSW_FRAMES}


def _rsplit(F, d):
    """A recursive frame's level-1 children as the root writes and reads them: (7,) + (2, 2) * (d - 1) + batch + leaf,
    entries 0..3 of every lower axis split into their (row, col) pair (the B side: (col, row)); one reshape, billed."""
    R = F[(slice(None),) + (slice(0, 4),) * (d - 2)] if d > 2 else F
    return R.reshape((7,) + (2, 2) * (d - 1) + tuple(R.shape[d:]))


def _carve(arena, lay):
    """Frames of a layout as views of one flat arena (product kinds run one after another, so every kind starts at
    0), and the root views of the A, B and product frames (reshapes, billed once per carve)."""
    off, fr = 0, {}
    for name in _RSW_FRAMES:
        s = _size(lay[name])
        fr[name] = arena[off:off + s].reshape(lay[name])
        off += s
    for name in ("A", "B", "M"):
        fr[name + "s"] = _rsplit(fr[name], lay["d"])
    fr["_vc"] = {"_orig": {id(v) for v in fr.values()}}  # vcache: views of these uncut frames, made once per plan
    return fr


def _rt(X, d, cm=False):
    """X (*lead, r, c) in the recursive layout: (2, 2) * d + lead + (r >> d, c >> d), pair l = (row bit, col bit) of
    level l, or (col bit, row bit) for the B side (cm); one reshape (billed) and one transpose (free)."""
    sh = tuple(X.shape)
    nl = len(sh) - 2
    r, c = sh[-2], sh[-1]
    Y = X.reshape(sh[:nl] + (2,) * d + (r >> d,) + (2,) * d + (c >> d,))
    ax = []
    for i_ in range(d):
        ax += [nl + d + 1 + i_, nl + i_] if cm else [nl + i_, nl + d + 1 + i_]
    return fnp.transpose(Y, tuple(ax) + tuple(range(nl)) + (nl + d, nl + 2 * d + 1))


def _rq(R):
    """The four root quadrants of a recursive view, in its pair order."""
    return R[0, 0], R[0, 1], R[1, 0], R[1, 1]


def _rqc(Q, mc, full):
    """Root quadrants of a batched recursive view, cut to the first mc batch entries."""
    return Q if mc == full else tuple(q[..., :mc, :, :] for q in Q)


def _rcall(fr, bc, md, kd, nd):
    """One call's frames, each cut to the call's batch bc and leaf sizes (a frame that already fits is itself)."""
    bc = tuple(bc)
    nb = len(bc)

    def cut(F, r, c):
        sh = tuple(F.shape)
        if sh[len(sh) - 2 - nb:] == bc + (r, c):
            return F
        return F[(Ellipsis,) + tuple(slice(0, b) for b in bc) + (slice(0, r), slice(0, c))]

    return (cut(fr["A"], md, kd), cut(fr["B"], kd, nd), cut(fr["M"], md, nd), cut(fr["As"], md, kd),
            cut(fr["Bs"], kd, nd), cut(fr["Ms"], md, nd), cut(fr["L"], md, nd), cut(fr["T"], kd, nd), fr["_vc"])


def _sw(A, B, out, va, vb, vm, d: int, ip: bool = True, qa0=None, qb0=None, qp0=None):
    """out = A @ B by d >= 2 levels of Strassen-Winograd in the recursive layout (_layout): the root copies A's
    quadrants 11, 12, 22 and B's 11, 21, 22 into the level-1 children and makes the 8 sums from the operands, each
    deeper level makes its 4 + 4 sums in place (one copy on the B side), the deepest level runs the in-place leaf of
    research/make_sw6.py (7 products and 15 additions over three leaf buffers and a spare B quadrant), and the products
    combine upwards in place (entries 0..3 of each axis) and at the root into out's quadrants. The products, sums and
    combination order are the earlier frame-per-level form's.
    A (*batch, m, K), B (*batch, K, N), out (*batch, m, N), m, K, N divisible by 2^d; va: the call's frames
    (_SwPlan.view, view_shape); vb, vm unused. qa0, qb0, qp0: the root quadrants of A (Z order), B (N order) and out
    (Z order) when made once per predict; otherwise each is made here (one reshape), out reusing A's when out is A
    (every read of A happens at the root, every write to out in the last combination)."""
    add, sub, cp, mm, us = fnp.add, fnp.subtract, fnp.copyto, fnp.matmul, fnp.unstack
    FA, FB, FM, FAs, FBs, FMs, L3, T2, VC = va
    orig = VC["_orig"]

    def uv(F, ax, tag, rs=None):  # vcache: an uncut frame's derived views are made once per plan
        if id(F) in orig:
            key = (id(F), ax, tag)
            v = VC.get(key)
            if v is None:
                v = VC[key] = us(F[rs] if rs else F, axis=ax)
            return v
        return us(F[rs] if rs else F, axis=ax)

    if qa0 is None:
        qa0 = _rq(_rt(A, d))
    if qb0 is None:
        qb0 = _rq(_rt(B, d, True))
    if qp0 is None:
        qp0 = qa0 if out is A else _rq(_rt(out, d))
    a11, a12, a21, a22 = qa0
    b11, b21, b12, b22 = qb0
    ca, cb = uv(FAs, 0, "r"), uv(FBs, 0, "r")
    cp(ca[0], a11)
    cp(ca[1], a12)
    cp(ca[3], a22)
    cp(cb[0], b11)
    cp(cb[1], b21)
    cp(cb[2], b22)
    add(a21, a22, out=ca[4])  # S1 = A21 + A22
    sub(ca[4], a11, out=ca[5])  # S2 = S1 - A11
    sub(a11, a21, out=ca[6])  # S3 = A11 - A21
    sub(a12, ca[5], out=ca[2])  # S4 = A12 - S2
    sub(b12, b11, out=cb[4])  # T1 = B12 - B11
    sub(b22, cb[4], out=cb[5])  # T2 = B22 - T1
    sub(b22, b12, out=cb[6])  # T3 = B22 - B12
    sub(cb[5], b21, out=cb[3])  # T4 = T2 - B21
    for lev in range(1, d - 1):  # in place: A [11, 12, 21, 22] -> [11, 12, S4, 22, S1, S2, S3]; B [11, 21, 12, 22] -> [11, 21, 22, T4, T1, T2, T3]
        r_ = d - 2 - lev
        rs = (slice(None),) * (lev + 1) + (slice(0, 4),) * r_
        e = uv(FA, lev, "s", rs if r_ else None)
        add(e[2], e[3], out=e[4])  # S1 = A21 + A22
        sub(e[4], e[0], out=e[5])  # S2 = S1 - A11
        sub(e[0], e[2], out=e[6])  # S3 = A11 - A21
        sub(e[1], e[5], out=e[2])  # S4 = A12 - S2 (over A21)
        f = uv(FB, lev, "s", rs if r_ else None)
        sub(f[2], f[0], out=f[4])  # T1 = B12 - B11
        sub(f[3], f[4], out=f[5])  # T2 = B22 - T1
        sub(f[3], f[2], out=f[6])  # T3 = B22 - B12
        cp(f[2], f[3])  # child 2 = B22 (over B12)
        sub(f[5], f[1], out=f[3])  # T4 = T2 - B21 (over B22)
    # the deepest level's quadrants (entries of the last lead axis) and the leaf, in place as before
    a11, a12, a21, a22 = uv(FA, d - 1, "q")
    b11, b21, b12, b22 = uv(FB, d - 1, "q")
    c11, c12, c21, c22 = uv(FM, d - 1, "q")
    X, Y, Z = uv(L3, 0, "l")
    mm(a11, b11, out=X)  # P1
    mm(a12, b21, out=Y)  # P2
    add(X, Y, out=c11)  # C11 = P1 + P2
    sub(b12, b11, out=b11)  # T1 = B12 - B11 (over B11)
    sub(b22, b12, out=b12)  # T3 = B22 - B12 (over B12)
    sub(b22, b11, out=T2)  # T2 = B22 - T1 (spare)
    sub(T2, b21, out=b21)  # T4 = T2 - B21 (over B21)
    mm(a22, b21, out=Y)  # P4 = A22 T4
    add(a21, a22, out=a22)  # S1 = A21 + A22 (over A22)
    sub(a11, a21, out=a21)  # S3 = A11 - A21 (over A21)
    sub(a22, a11, out=a11)  # S2 = S1 - A11 (over A11)
    sub(a12, a11, out=a12)  # S4 = A12 - S2 (over A12)
    mm(a11, T2, out=Z)  # P6 = S2 T2
    add(X, Z, out=c12)  # U2 = P1 + P6
    mm(a21, b12, out=X)  # P7 = S3 T3
    add(c12, X, out=c21)  # U3 = U2 + P7
    mm(a22, b11, out=Z)  # P5 = S1 T1
    add(c21, Z, out=c22)  # C22 = U3 + P5
    add(c12, Z, out=c12)  # U4 = U2 + P5
    sub(c21, Y, out=c21)  # C21 = U3 - P4
    mm(a12, b22, out=X)  # P3 = S4 B22
    add(c12, X, out=c12)  # C12 = U4 + P3
    for lev in range(d - 2, 0, -1):  # level lev + 1 products (axis lev) into level lev's quadrants, in place (Z order)
        r_ = d - 2 - lev
        rs = (slice(None),) * (lev + 1) + (slice(0, 4),) * r_
        e = uv(FM, lev, "m", rs if r_ else None)
        add(e[0], e[5], out=e[5])  # U2 = M1 + M6
        add(e[0], e[1], out=e[0])  # C11 = M1 + M2
        add(e[5], e[6], out=e[6])  # U3 = U2 + M7
        add(e[5], e[4], out=e[5])  # U4 = U2 + M5
        add(e[5], e[2], out=e[1])  # C12 = U4 + M3
        sub(e[6], e[3], out=e[2])  # C21 = U3 - M4
        add(e[6], e[4], out=e[3])  # C22 = U3 + M5
    M1, M2, M3, M4, M5, M6, M7 = uv(FMs, 0, "r")
    c11, c12, c21, c22 = qp0
    add(M1, M2, out=c11)
    add(M1, M6, out=c12)
    add(c12, M7, out=c21)
    add(c21, M5, out=c22)
    add(c12, M5, out=c12)
    add(c12, M3, out=c12)
    sub(c21, M4, out=c21)
    return out


def _sw_warm(va, d: int):
    """setup plan: make (and cache) every view _sw derives from a call's frames, without any op."""
    us = fnp.unstack
    FA, FB, FM, FAs, FBs, FMs, L3, T2, VC = va
    orig = VC["_orig"]

    def uv(F, ax, tag, rs=None):  # the same cache and keys as _sw
        if id(F) in orig:
            key = (id(F), ax, tag)
            v = VC.get(key)
            if v is None:
                v = VC[key] = us(F[rs] if rs else F, axis=ax)
            return v
        return us(F[rs] if rs else F, axis=ax)

    uv(FAs, 0, "r")
    uv(FBs, 0, "r")
    for lev in range(1, d - 1):
        r_ = d - 2 - lev
        rs = (slice(None),) * (lev + 1) + (slice(0, 4),) * r_
        uv(FA, lev, "s", rs if r_ else None)
        uv(FB, lev, "s", rs if r_ else None)
    uv(FA, d - 1, "q")
    uv(FB, d - 1, "q")
    uv(FM, d - 1, "q")
    uv(L3, 0, "l")
    for lev in range(d - 2, 0, -1):
        r_ = d - 2 - lev
        rs = (slice(None),) * (lev + 1) + (slice(0, 4),) * r_
        uv(FM, lev, "m", rs if r_ else None)
    uv(FMs, 0, "r")


_COVSYM = (5, 5)  # r143-B covsym: Strassen-Winograd depths of S0's half block and of its six quarter blocks
_QPAIRS = ((0, 1), (2, 3), (0, 0), (1, 1), (2, 2), (3, 3))  # (row quarter of W^T, column quarter of T) per batched block
_R28B = {'sl': True, 'cs': 4}  # r28b: sl live-slot chunks; cs eighth-block depth (None: off)
_EPAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7))  # r28b cs: (row, column) eighth
_R30B = {'q0': True, 'q12': True, 'lr6': True, 'w4p': True}  # r30b: work billed but not needed, cut


def _r28_sl(ch, ns):
    """r28b sl: the e1c chunk views cut to the live slots (an unborn slot computes zeros that nothing reads)."""
    out = []
    for c_, v_ in enumerate(ch[:(ns + 7) // 8]):
        nl_ = ns - 8 * c_
        out.append(v_ if nl_ >= v_[0].shape[0] else tuple(x_[:nl_] for x_ in v_))
    return out


def _covsym(W, T, M0, swp, ip):
    """S0 = W^T T with T = C W (C symmetric), so S0 is symmetric: the upper half block W_1^T T_2 (n/2 x n x n/2), then
    batched the upper quarter blocks of the two diagonal halves and the four diagonal quarter blocks (n/4 x n x n/4),
    written to M0 with their mirror images; each diagonal quarter block is symmetrized (D + D^T) / 2."""
    n = W.shape[0]
    h, q = n // 2, n // 4
    cp, add, mul = fnp.copyto, fnp.add, fnp.multiply
    b = swp.covbuf
    OFF, A6, B6, O6, SQ = b["OFF"], b["A6"], b["B6"], b["O6"], b["SQ"]
    WT = W.T
    dh, vah, _, _ = swp.view(("covh", 0))
    _sw(WT[:h], T[:, h:], OFF, vah, None, None, dh, ip)
    nq_ = A6.shape[0]  # r28b cs: 2 (the diagonal quarter blocks come from eighth blocks below) or 6
    fnp.stack([WT[r_ * q:(r_ + 1) * q] for r_, _ in _QPAIRS[:nq_]], out=A6)
    fnp.stack([T[:, c_ * q:(c_ + 1) * q] for _, c_ in _QPAIRS[:nq_]], out=B6)
    dq, vaq, _, _ = swp.view(("covq", 0))
    _sw(A6, B6, O6, vaq, None, None, dq, ip)
    cp(M0[:h, h:], OFF)
    cp(M0[h:, :h], OFF.T)
    cp(M0[:q, q:h], O6[0])
    cp(M0[q:h, :q], O6[0].T)
    cp(M0[h:h + q, h + q:], O6[1])
    cp(M0[h + q:, h:h + q], O6[1].T)
    if nq_ == 2:  # r28b cs: a diagonal quarter block is symmetric, so its upper eighth block and two diagonal ones
        e = n // 8
        A12, B12, O12, S8 = b["A12"], b["B12"], b["O12"], b["S8"]
        fnp.stack([WT[r_ * e:(r_ + 1) * e] for r_, _ in _EPAIRS], out=A12)
        fnp.stack([T[:, c_ * e:(c_ + 1) * e] for _, c_ in _EPAIRS], out=B12)
        de, vae, _, _ = swp.view(("cove", 0))
        _sw(A12, B12, O12, vae, None, None, de, ip)
        for i_ in range(4):
            r0 = 2 * i_ * e
            cp(M0[r0:r0 + e, r0 + e:r0 + 2 * e], O12[i_])
            cp(M0[r0 + e:r0 + 2 * e, r0:r0 + e], O12[i_].T)
        add(O12[4:], fnp.swapaxes(O12[4:], 1, 2), out=S8)  # the eight diagonal eighth blocks, symmetrized at once
        mul(S8, 0.5, out=S8)
        for r_ in range(8):
            cp(M0[r_ * e:(r_ + 1) * e, r_ * e:(r_ + 1) * e], S8[r_])
        return
    for i_ in range(4):
        D = O6[2 + i_]
        add(D, D.T, out=SQ)
        mul(SQ, 0.5, out=M0[i_ * q:(i_ + 1) * q, i_ * q:(i_ + 1) * q])


# r143-B apa: one approximate (APA) level at the top of the 3 T^T and of the pushes that carry no covariance rows:
# Bini, Capovani, Lotti and Romani's <3,2,2;10> (Inf. Proc. Lett. 8(5), 1979; coefficients from github.com/arbenson/
# fast-matmul codegen/algorithms, BSD-2-Clause, verified exactly in research/r143/apa/apa_schemes.py), 3-split on the
# long dimension, eps = 2^-8, its ten products batched through the engine's Strassen-Winograd one level shallower.
_APA_ORI = {'tt': '232', 'push': '322'}
_APA_SPLIT = False  # the pushes that carry covariance rows: leg rows by APA, covariance rows exactly
_BINI = {'322': {'m0': 3, 'k0': 2, 'n0': 2, 't': 10, 'U': [[(0, 0, 1.0), (1, 1, 1.0)], [(1, 1, 1.0)], [(0, 0, 1.0)], [(0, 1, 0.00390625), (1, 1, 1.0)], [(0, 0, 1.0), (0, 1, 0.00390625)], [(1, 0, 1.0), (2, 1, 1.0)], [(2, 1, 1.0)], [(1, 0, 1.0)], [(2, 0, 0.00390625), (2, 1, 1.0)], [(1, 0, 1.0), (2, 0, 0.00390625)]], 'V': [[(0, 0, 0.00390625), (1, 1, 1.0)], [(1, 0, -1.0), (1, 1, -1.0)], [(1, 1, 1.0)], [(0, 0, -0.00390625), (1, 0, 1.0)], [(0, 1, 0.00390625), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 0.00390625)], [(0, 0, 1.0)], [(0, 0, -1.0), (0, 1, -1.0)], [(0, 0, 1.0), (1, 0, 0.00390625)], [(0, 1, 1.0), (1, 1, -0.00390625)]], 'W': [((0, 0), [(0, 256.0), (1, 256.0), (2, -256.0), (3, 256.0)]), ((0, 1), [(2, -256.0), (4, 256.0)]), ((1, 0), [(3, 1.0), (5, 1.0), (8, -1.0)]), ((1, 1), [(0, 1.0), (4, -1.0), (9, 1.0)]), ((2, 0), [(6, -256.0), (8, 256.0)]), ((2, 1), [(5, 256.0), (6, -256.0), (7, 256.0), (9, 256.0)])]}, '232': {'m0': 2, 'k0': 3, 'n0': 2, 't': 10, 'U': [[(0, 0, 1.0), (1, 1, 1.0)], [(1, 1, 1.0)], [(0, 0, 1.0)], [(1, 0, 0.00390625), (1, 1, 1.0)], [(0, 0, 1.0), (1, 0, 0.00390625)], [(0, 1, 1.0), (1, 2, 1.0)], [(1, 2, 1.0)], [(0, 1, 1.0)], [(0, 2, 0.00390625), (1, 2, 1.0)], [(0, 1, 1.0), (0, 2, 0.00390625)]], 'V': [[(0, 0, 256.0), (1, 1, 1.0)], [(0, 0, 256.0)], [(0, 0, -256.0), (0, 1, -256.0)], [(0, 0, 256.0), (1, 0, 1.0)], [(0, 1, 256.0), (1, 1, -1.0)], [(1, 0, 1.0), (2, 1, 256.0)], [(2, 0, -256.0), (2, 1, -256.0)], [(2, 1, 256.0)], [(1, 0, -1.0), (2, 0, 256.0)], [(1, 1, 1.0), (2, 1, 256.0)]], 'W': [((0, 0), [(0, 0.00390625), (3, -0.00390625), (5, 1.0), (6, 1.0), (7, -1.0), (8, 1.0)]), ((0, 1), [(4, 0.00390625), (7, -1.0), (9, 1.0)]), ((1, 0), [(1, -1.0), (3, 1.0), (8, 0.00390625)]), ((1, 1), [(0, 1.0), (1, -1.0), (2, 1.0), (4, 1.0), (5, 0.00390625), (9, -0.00390625)])]}}


def _apa_enc(dst, terms, T):
    """dst = sum of coef * block over terms; blocks cut short by the zero padding are added into dst's corner."""
    cp, add, sub, mul = fnp.copyto, fnp.add, fnp.subtract, fnp.multiply
    R, C = dst.shape
    terms = [(x, c) for x, c in terms if x.shape[0] and x.shape[1]]
    i0 = next((i_ for i_, (x, c) in enumerate(terms) if tuple(x.shape) == (R, C)), None)
    if i0 is None:
        cp(dst, 0.0)
        rest = terms
    else:
        x, c = terms[i0]
        if c == 1.0:
            cp(dst, x)
        else:
            mul(x, c, out=dst)
        rest = terms[:i0] + terms[i0 + 1:]
    for x, c in rest:
        rs, cs = x.shape
        d_ = dst if (rs, cs) == (R, C) else dst[:rs, :cs]
        if c == 1.0:
            add(d_, x, out=d_)
        elif c == -1.0:
            sub(d_, x, out=d_)
        else:
            t_ = T[:rs, :cs]
            mul(x, c, out=t_)
            add(d_, t_, out=d_)


def _apa(kind, swp, A, B, out, dp, ip):
    """out = A @ B approximately: one Bini level (orientation _APA_ORI[kind]) over Strassen-Winograd at depth dp."""
    o = _BINI[_APA_ORI[kind]]
    m0, k0, n0 = o["m0"], o["k0"], o["n0"]
    m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]
    g = 1 << dp
    mb, kb, nb = -(-m // (m0 * g)) * g, -(-k // (k0 * g)) * g, -(-n // (n0 * g)) * g
    va, AS, BS, OS, TA, TB, TC = swp.apa_view(kind, dp, mb, kb, nb)

    def blk(X, i, l, bi, bl, R_, C_):
        r0, c0 = i * bi, l * bl
        return X[r0:r0 + max(0, min(bi, R_ - r0)), c0:c0 + max(0, min(bl, C_ - c0))]

    for r in range(o["t"]):
        _apa_enc(AS[r], [(blk(A, i, l, mb, kb, m, k), c) for i, l, c in o["U"][r]], TA)
        _apa_enc(BS[r], [(blk(B, l, j, kb, nb, k, n), c) for l, j, c in o["V"][r]], TB)
    _sw(AS, BS, OS, va, None, None, dp, ip)
    for (i, j), terms in o["W"]:
        dst = blk(out, i, j, mb, nb, m, n)
        rs, cs = dst.shape
        if rs and cs:
            _apa_enc(dst, [(OS[r][:rs, :cs], c) for r, c in terms], TC)
    return out


def _swk(kind, swp, A, B, out, va, vb, vm, d, ip, qp0=None):
    """A product of kind 'tt' or 'push' (no covariance rows): one APA level when the plan has its frames, else _sw.
    Kind 'pushc' (the fold layers' and the last layer's push, whose last n rows carry the covariance): with _APA_SPLIT
    the covariance rows exactly by Strassen-Winograd, the leg rows with one APA level."""
    if kind == "pushc":
        nC = B.shape[-2]
        if _APA_SPLIT and "push" in _APA_ORI and swp is not None and ("apa_push", d - 1) in swp.frames and A.shape[-2] > nC:
            c0 = A.shape[-2] - nC
            _apa("push", swp, A[:c0], B, out[:c0], d - 1, ip)
            vaC = swp.plain_view("push", d, nC, nC, B.shape[-1])
            _sw(A[c0:], B, out[c0:], vaC, None, None, d, ip)
            return out
        if va is None:
            va = swp.plain_view("push", d, A.shape[-2], A.shape[-1], B.shape[-1])
        return _sw(A, B, out, va, vb, vm, d, ip, qp0=qp0)
    if kind in _APA_ORI and swp is not None and ("apa_" + kind, d - 1) in swp.frames:
        return _apa(kind, swp, A, B, out, d - 1, ip)
    if va is None:
        va = swp.plain_view(kind, d, A.shape[-2], A.shape[-1], B.shape[-1])
    return _sw(A, B, out, va, vb, vm, d, ip, qp0=qp0)


class _SwPlan:
    """The Strassen-Winograd products of one predict at shape (n, depth) and cost setting (age, rank k, depths d, d2,
    the deeper layers hp / ht / hb, in place ip): the recursive work layouts (sized for the longest operand of each
    product kind at each depth), one arena with its frames, and per call (kind, layer t) the frames cut to the call.
    Estimator.setup() builds one with views=True when sw_setup_plan is set; otherwise predict builds one with
    views=False and cuts each call's frames when the call comes."""

    def __init__(self, f32, n, depth, age, k, d, d2, hp, ht, hb, ip, views=True, cf=frozenset()):
        Smc = max(0, depth - 1 - age - _R25B.get('s3', 0)) if (age is not None and age >= 1 and 0 < k < n) else 0
        d2 = d2 if Smc else 0
        hb = hb if d2 else frozenset()
        self.key = (n, depth, age, k, d, d2, hp, ht, hb, ip, cf)
        self.n, self.k, self.d, self.d2, self.ip = n, k, d, d2, ip
        self.persist = bool(views)  # setup plan: the views live across predicts and their derived views are cached
        self._r25_plan = True  # r25b ax: the plan is being made (setup)
        sizes = {}  # setup plan: (t) -> (push rows, 3 T^T rows) for the saturation-restricted calls
        Xr, mc_ = 0, 0
        rows = {}  # (kind, depth): longest operand among the layers that run that product kind at that depth
        self.calls = {}  # (kind, t) -> (depth, batch, m, K, N)
        for t_ in range(depth):
            off_ = 2 * mc_ * n - mc_ * k
            if t_:
                rp, rt = ((2 * t_ - (0 if (t_ == depth - 1 or t_ in cf) else 1)) * n - off_,  # E3, covfold: C' or C rides
                          ((2 * t_ * n - off_) if t_ < depth - 1 else 0))
                sizes[t_] = (rp, rt)
                Xr = max(Xr, rp, rt)
                kp = ("push", d + 1 if t_ in hp else d)
                rows[kp] = max(rows.get(kp, 0), rp)
                if _R25B.get('s2') and n == 1024 and _R25B['s2']['apa'] and t_ == depth - 1:  # r25b s2: the last legs by APA at depth d
                    rows[("push", d)] = max(rows.get(("push", d), 0), rp - n)
                if d:
                    self.calls[("push", t_)] = (kp[1], (), rp, n, n)
                    if t_ in cf and not _COVSYM:
                        self.calls[("cov", t_)] = (d, (), n, n, n)
                if rt:
                    kt = ("tt", d + 1 if t_ in ht else d)
                    rows[kt] = max(rows.get(kt, 0), rt)
                    if d:
                        self.calls[("tt", t_)] = (kt[1], (), n, rt, n)
                if mc_ and d2:
                    db = d2 + 1 if t_ in hb else d2
                    self.calls[("rebuild", t_)] = (db, (mc_,), 2 * n, k, n)
                    if rt:
                        self.calls[("p", t_)] = (db, (mc_,), k, 2 * n, n)
            while Smc and mc_ < Smc and mc_ <= t_ - age:
                mc_ += 1
        lays = {}
        self.apabuf, self.apaviews = {}, {}  # r143-B apa: the ten products' frames (batched layouts), buffers, views
        self.apa_only = set()  # r143-B apa: (kind, depth) whose every call runs through APA (no plain frames made)
        if d:
            need_ = {}
            cpd_ = {d + 1 if t_ in hp else d for t_ in range(1, depth) if t_ in cf or t_ == depth - 1}  # C-carrying pushes
            for (kind_, dd_), R_ in sorted(rows.items()):
                if kind_ not in _APA_ORI or dd_ - 1 < 2 or (kind_ == "push" and dd_ != d and not _APA_SPLIT):
                    continue
                o_ = _BINI[_APA_ORI[kind_]]
                m_, k_, n_ = (n, R_, n) if kind_ == "tt" else (R_, n, n)
                g_ = 1 << (dd_ - 1)
                mb_, kb_, nb_ = (-(-m_ // (o_["m0"] * g_)) * g_, -(-k_ // (o_["k0"] * g_)) * g_, -(-n_ // (o_["n0"] * g_)) * g_)
                lays[("apa_" + kind_, dd_ - 1)] = _layout(dd_ - 1, (10,), (mb_, kb_), (kb_, nb_))
                if not (kind_ == "push" and dd_ in cpd_):
                    self.apa_only.add((kind_, dd_))
                e_ = need_.setdefault(kind_, [0, 0, 0, 0])
                e_[0] = max(e_[0], 10 * mb_ * kb_)
                e_[1] = max(e_[1], 10 * kb_ * nb_)
                e_[2] = max(e_[2], 10 * mb_ * nb_)
                e_[3] = max(e_[3], mb_ * kb_, kb_ * nb_, mb_ * nb_)
            for kind_, (a_, b_, c_, t_) in need_.items():
                self.apabuf[kind_] = {"AS": fnp.empty((a_,), dtype=f32), "BS": fnp.empty((b_,), dtype=f32),
                                      "OS": fnp.empty((c_,), dtype=f32), "T": fnp.empty((t_,), dtype=f32)}
        if d:
            for dd_ in (d, d + 1):  # legs^T @ W
                if ("push", dd_) in rows:
                    lays[("push", dd_)] = _layout(dd_, (), ((rows[("push", dd_)] if ip else Xr), n), (n, n))
            if _R25B.get('s7') and n == 1024 and d2:  # r25b s7: exact frames for the band rebuild and P
                for P_ in _R25B['s7_pb']:
                    lays[("rebuild_k%d" % P_, d2)] = _layout(d2, (Smc,), (2 * n, k), (k, P_))
                    lays[("p_k%d" % P_, d2)] = _layout(d2, (Smc,), (k, 2 * n), (2 * n, P_))
            if _R25B and n == 1024 and ("push", d + 1) in rows:  # r25b: exact frames (contiguous leaf blocks)
                lays[("pushC", d + 1)] = _layout(d + 1, (), (n, n), (n, n))
                for P_ in sorted({p_ for v_ in _R25B.get('s1_bins', {}).values() for p_ in v_}):
                    lays[("push_k%d" % P_, d + 1)] = _layout(d + 1, (), (rows[("push", d + 1)], P_), (P_, n))
                if _R25B.get('s2') and not _R25B['s2']['apa']:  # r25b s2 (plain): exact frames per (Pin, Pb) pair
                    for x_, y_ in _R25B['s2']['pairs']:
                        lays[("push_s2_%d_%d" % (x_, y_), d + 1)] = _layout(d + 1, (), (rows[("push", d + 1)], x_), (x_, y_))
                if _R25B.get('s2') and _R25B['s2'].get('rb') and d2:  # r25b rb: exact frames per band size
                    for y_ in sorted({q_ for _, q_ in _R25B['s2']['pairs']}):
                        lays[("rebuild_k%d" % y_, d2)] = _layout(d2, (Smc,), (2 * n, k), (k, y_))
            for dd_ in (d, d + 1):  # L^T A
                if ("tt", dd_) in rows:
                    R_ = rows[("tt", dd_)] if ip else Xr
                    lays[("tt", dd_)] = _layout(dd_, (), (n, R_), (R_, n))
        if d and cf and not _COVSYM:
            lays[("cov", d)] = _layout(d, (), (n, n), (n, n))  # covfold: S0 = W^T (C W)
        self.covbuf = None
        if d and _COVSYM:  # r143-B covsw: T = C W on the non-fold layers (n x n x n)
            lays[("covT", _COVSYM[0])] = _layout(_COVSYM[0], (), (n, n), (n, n))
            self.calls[("covT", 0)] = (_COVSYM[0], (), n, n, n)
        if d and _COVSYM:  # r143-B covsym: S0's half block and six quarter blocks (batched), their buffers
            dh_, dq_ = _COVSYM
            de_ = _R28B.get('cs')  # r28b cs: the eighth blocks' depth (None: the six quarter blocks)
            if de_ and (n // 8) % (1 << de_):
                de_ = None
            nq_ = 2 if de_ else 6
            lays[("covh", dh_)] = _layout(dh_, (), (n // 2, n), (n, n // 2))
            lays[("covq", dq_)] = _layout(dq_, (nq_,), (n // 4, n), (n, n // 4))
            self.calls[("covh", 0)] = (dh_, (), n // 2, n, n // 2)
            self.calls[("covq", 0)] = (dq_, (nq_,), n // 4, n, n // 4)
            self.covbuf = {"OFF": fnp.empty((n // 2, n // 2), dtype=f32), "A6": fnp.empty((nq_, n // 4, n), dtype=f32),
                           "B6": fnp.empty((nq_, n, n // 4), dtype=f32), "O6": fnp.empty((nq_, n // 4, n // 4), dtype=f32),
                           "SQ": fnp.empty((n // 4, n // 4), dtype=f32)}
            if de_:  # r28b cs: the twelve eighth blocks (4 upper, 8 diagonal), batched, and their buffers
                lays[("cove", de_)] = _layout(de_, (12,), (n // 8, n), (n, n // 8))
                self.calls[("cove", 0)] = (de_, (12,), n // 8, n, n // 8)
                self.covbuf.update({"A12": fnp.empty((12, n // 8, n), dtype=f32), "B12": fnp.empty((12, n, n // 8), dtype=f32),
                                    "O12": fnp.empty((12, n // 8, n // 8), dtype=f32), "S8": fnp.empty((8, n // 8, n // 8), dtype=f32)})
        if d2:
            for dd_ in (d2, d2 + 1) if hb else (d2,):
                lays[("rebuild", dd_)] = _layout(dd_, (Smc,), (2 * n, k), (k, n))  # Ct @ Qt
            for dd_ in (d2, d2 + 1) if hb else (d2,):
                lays[("p", dd_)] = _layout(dd_, (Smc,), (k, 2 * n), (2 * n, n))  # Ct^T @ A
        self.lays = lays
        self.arena = None
        self.frames = {}
        self.views = {}
        if lays:
            self.arena = fnp.empty((max(sum(_regions(v).values()) for v in lays.values()),), dtype=f32)
            self.frames = {kind: _carve(self.arena, v) for kind, v in lays.items()}
            if views:
                skip = set()  # fixed mode: the plain views of the calls the sat step restricts (predict never reads them)
                if d and _SAT.get("mode") == "fixed":  # the restricted calls' frames, from the sat step's fixed rule
                    q_ = _SAT.get("q", 32)
                    rb = ("band_" if _SAT["tb_band"] else "alive_") + str(_SAT["tb"])
                    ord_p = False
                    for t_ in range(1, depth - 1):
                        dtt = d + 1 if t_ in ht else d
                        Pp = n
                        if ord_p:
                            Pp = max(_sat_size("alive_" + str(_SAT["in"]), t_ - 1, n, max(1 << d, q_)),
                                     _sat_size("alive_" + str(_SAT["out"]), t_, n, max(1 << d, q_)))
                        Pa = _sat_size("alive_" + str(_SAT["ta"]), t_, n, max(1 << dtt, q_))
                        Pb = _sat_size(rb, t_, n, max(1 << dtt, q_))
                        nxt = _sat_size("alive_" + str(_SAT["in"]), t_, n, max(1 << d, q_)) < n
                        rp, rt = sizes[t_]
                        dcs_ = _dc_sizes(t_, n, depth, age, k, Smc, d, cf, hp)
                        if dcs_ is not None:  # dc: the gathered rows
                            rp, rt = dcs_
                            if Pp >= n:
                                self.view_shape("push", d + 1 if t_ in hp else d, rp, n, n)
                                skip.add(("push", t_))
                            if not (Pa < n or Pb < n):
                                self.view_shape("tt", dtt, n, rt, n)
                                skip.add(("tt", t_))
                        if Pp < n:
                            self.view_shape("push", d, rp, Pp, Pp)
                            skip.add(("push", t_))
                        if Pa < n or Pb < n:
                            self.view_shape("tt", dtt, Pa, rt, Pb)
                            skip.add(("tt", t_))
                        ord_p = Pp < n or Pa < n or Pb < n or nxt
                if d and _SAT.get("mode") == "bins":  # mode bins: every bin's restricted frames (a predict rounds up to one)
                    q_ = _SAT.get("q", 32)
                    for t_ in range(1, depth - 1):
                        dtt = d + 1 if t_ in ht else d
                        rp, rt = sizes[t_]
                        dcs_ = _dc_sizes(t_, n, depth, age, k, Smc, d, cf, hp)
                        if dcs_ is not None:  # dc: every bin's frames on the gathered rows, and the unrestricted calls
                            rp, rt = dcs_
                            self.view_shape("push", d + 1 if t_ in hp else d, rp, n, n)
                            skip.add(("push", t_))
                            self.view_shape("tt", dtt, n, rt, n)
                            skip.add(("tt", t_))
                        pu_, tt_ = _sat_bins(t_, n, d, dtt, q_, cf)
                        for P_ in (pu_ if not (_R25B.get('s8') and n == 1024 and t_ in _R25B['s8']['pairs']) else ()):  # r25b s8
                            self.view_shape("push", d, rp, P_, P_)
                        for Pa_, Pb_ in tt_:
                            self.view_shape("tt", dtt, Pa_, rt, Pb_)
                    skip |= {(k_, int(t_)) for k_, t_ in _SAT_BINS["skip_plain"]}  # calls every mini MLP restricts
                    dcs_ = _dc_sizes(depth - 1, n, depth, age, k, Smc, d, cf, hp)
                    if dcs_ is not None:  # dc: the last push (no 3 T^T)
                        self.view_shape("push", d + 1 if (depth - 1) in hp else d, dcs_[0], n, n)
                        skip.add(("push", depth - 1))
                    for t_ in range(1, depth if n == 1024 else 1):  # r25b: the restricted shapes s1, s8, s2 reach (scored width only)
                        dcs_ = _dc_sizes(t_, n, depth, age, k, Smc, d, cf, hp)
                        if dcs_ is None:
                            continue
                        ddf_ = d + 1 if t_ in hp else d
                        if _R25B.get('s1') and t_ >= _R25B['s1'] and t_ in cf:
                            for P_ in _R25B['s1_bins'].get(t_, []):
                                _r25_bview(self, "push_k%d" % P_, ddf_, (), dcs_[0] - n, P_, n)
                            _r25_bview(self, "pushC", ddf_, (), n, n, n)
                        if _R25B.get('s8') and t_ in _R25B['s8']['pairs']:
                            for x_, y_ in _R25B['s8']['pairs'][t_]:
                                self.view_shape("push", d, dcs_[0], x_, y_)
                        if _R25B.get('s7') and d2 and Smc and 0 < t_ < depth - 1 and t_ - age >= 1:
                            mc7_ = min(Smc, t_ - age)
                            for P_ in _R25B['s7_pb']:
                                _r25_bview(self, "rebuild_k%d" % P_, d2, (mc7_,), 2 * n, k, P_)
                                _r25_bview(self, "p_k%d" % P_, d2, (mc7_,), k, 2 * n, P_)
                        if _R25B.get('s2') and t_ == depth - 1:
                            _r25_bview(self, "pushC", ddf_, (), n, n, n)
                            if _R25B['s2'].get('rb') and d2 and Smc:
                                for P_ in sorted({q_ for _, q_ in _R25B['s2']['pairs']}):
                                    _r25_bview(self, "rebuild_k%d" % P_ if ("rebuild_k%d" % P_, d2) in self.frames else "rebuild", d2, (Smc,), 2 * n, k, P_)
                            for x_, y_ in _R25B['s2']['pairs']:
                                if not _R25B['s2']['apa'] and ("push_s2_%d_%d" % (x_, y_), ddf_) in self.frames:
                                    _r25_bview(self, "push_s2_%d_%d" % (x_, y_), ddf_, (), dcs_[0] - n, x_, y_)
                                else:
                                    self.view_shape("push", d if _R25B['s2']['apa'] else ddf_, dcs_[0] - n, x_, y_)
                for key in self.calls:  # a call predict reaches unplanned still gets its view there (lazily)
                    if key not in skip:
                        self.view(key)
                for v in list(self.views.values()):  # 3. every call's derived views, made now
                    if v[1] is not None:  # r143-B apa: APA shapes have no plain frames
                        _sw_warm(v[1], v[0])
        self._r25_plan = False  # r25b ax: a later apa_view call is a predict's

    def bill(self):
        """Carve the frames again from the arena: the reshapes a predict without a setup plan makes (billed as such)."""
        for v in self.lays.values():
            _carve(self.arena, v)

    def _apa_on(self, kind, dd):
        """r143-B apa: does every product of this kind at depth dd run through one APA level (no
        covariance-carrying push shares the depth), so its shapes need no plain frames."""
        return (kind, dd) in self.apa_only

    def plain_view(self, kind, dd, m, K, N):
        """r143-B apa: plain frames of a shape made on demand (a fallback the planned calls do not reach)."""
        key = ("plain", kind, dd, m, K, N)
        v = self.views.get(key)
        if v is None:
            v = self.views[key] = (dd, _rcall(self.frames[(kind, dd)], (), m >> dd, K >> dd, N >> dd), None, None)
        return v[1]

    def _apa_pre(self, kind, dd, m, K, N):
        """r143-B apa: with the setup plan, the APA views of a plain shape are made when its plain view is."""
        if self.persist and kind in _APA_ORI and ("apa_" + kind, dd - 1) in self.frames:
            o_ = _BINI[_APA_ORI[kind]]
            g_ = 1 << (dd - 1)
            for m_ in ((m, m - K) if (kind == "push" and _APA_SPLIT and m > K) else (m,)):
                self.apa_view(kind, dd - 1, -(-m_ // (o_["m0"] * g_)) * g_, -(-K // (o_["k0"] * g_)) * g_,
                              -(-N // (o_["n0"] * g_)) * g_)
            if kind == "push" and _APA_SPLIT and m > K:
                self.view_shape("push", dd, K, K, N)

    def apa_view(self, kind, dp, mb, kb, nb):
        """r143-B apa: the batched frames and operand views of one APA product shape, cached per shape."""
        key = (kind, dp, mb, kb, nb)
        v = self.apaviews.get(key)
        if v is None:
            lay_ = self.lays[("apa_" + kind, dp)]  # r25b ax: this APA shape's own frames while the setup plan is made,
            fit_ = lay_["A"][-2:] == (mb >> dp, kb >> dp) and lay_["B"][-2:] == (kb >> dp, nb >> dp)  # unless it fits
            if _R25B.get('ax') and getattr(self, '_r25_plan', False) and not fit_:
                fo_ = self.frames[("apa_" + kind, dp)]
                vo_ = _rcall(fo_, (10,), mb >> dp, kb >> dp, nb >> dp)  # the base's cut frames of this shape
                if any(x is fo_[k_] for x, k_ in zip(vo_[:8], ("A", "B", "M", "As", "Bs", "Ms", "L", "T"))):
                    _sw_warm(vo_, dp)  # a full frame among them: its derived views are made now, as the base made them
                va = _rcall(_carve(self.arena, _layout(dp, (10,), (mb, kb), (kb, nb))), (10,), mb >> dp, kb >> dp, nb >> dp)
            else:
                va = _rcall(self.frames[("apa_" + kind, dp)], (10,), mb >> dp, kb >> dp, nb >> dp)
            if self.persist:
                va[8]["_orig"].update(id(x) for x in va[:8])
            bf = self.apabuf[kind]
            v = self.apaviews[key] = (va, bf["AS"][:10 * mb * kb].reshape((10, mb, kb)), bf["BS"][:10 * kb * nb].reshape((10, kb, nb)),
                                   bf["OS"][:10 * mb * nb].reshape((10, mb, nb)), bf["T"][:mb * kb].reshape((mb, kb)),
                                   bf["T"][:kb * nb].reshape((kb, nb)), bf["T"][:mb * nb].reshape((mb, nb)))
            if self.persist:
                _sw_warm(va, dp)
        return v

    def view_shape(self, kind, dd, m, K, N):
        """(depth, the call's frames, None, None) of a product (m, K) @ (K, N) on this kind's frames of depth dd (the
        saturation-restricted push and 3 T^T), cached per shape."""
        key = ("shape", kind, dd, m, K, N)
        v = self.views.get(key)
        if v is None:
            if self._apa_on(kind, dd):  # r143-B apa: this shape runs through one APA level, no plain frames
                self._apa_pre(kind, dd, m, K, N)
                v = self.views[key] = (dd, None, None, None)
                return v
            v = self.views[key] = (dd, _rcall(self.frames[(kind, dd)], (), m >> dd, K >> dd, N >> dd), None, None)
            if self.persist:  # its cut frames live as long as the plan: cache their derived views
                v[1][8]["_orig"].update(id(x) for x in v[1][:8])
            self._apa_pre(kind, dd, m, K, N)
        return v

    def view(self, key):
        """(depth, the call's frames, None, None) of call key = (kind, t)."""
        v = self.views.get(key)
        if v is None:
            dd, bc, m, K, N = self.calls[key]
            if not bc and self._apa_on(key[0], dd):  # r143-B apa: no plain frames for an APA shape
                self._apa_pre(key[0], dd, m, K, N)
                v = self.views[key] = (dd, None, None, None)
                return v
            v = self.views[key] = (dd, _rcall(self.frames[(key[0], dd)], bc, m >> dd, K >> dd, N >> dd), None, None)
            if self.persist:  # its cut frames live as long as the plan: cache their derived views
                v[1][8]["_orig"].update(id(x) for x in v[1][:8])
            if not bc:
                self._apa_pre(key[0], dd, m, K, N)
        return v


class _SimpleEngine:
    """estimators/k3_fast.py's SIMPLE propagation, verbatim: the fallback for shapes the AUG table does not cover."""

    def __init__(self, dtype=None) -> None:
        self._dtype = dtype

    def propagate(self, weights):
        """Per-layer means (depth, n)."""
        f32 = fnp.float32 if self._dtype is None else self._dtype
        depth = len(weights)
        n = weights[0].shape[0]
        S = max(depth - 1, 1)  # sources ever created (one per layer but the last)
        mul, cp, es = fnp.multiply, fnp.copyto, fnp.einsum

        # ---- constants and slabs (a fixed number of calls per predict) ----
        tC2, tLR = _TABLES
        P = _ptable(n)
        cP = fnp.array(P, dtype=f32)  # Z0 = [1, tA, tB] (last layer)
        cPz = fnp.array([[P[r][1], P[r][0]] for r in range(_NPX)], dtype=f32)  # over diag([3T^T, I]) = [tA, 1]
        cC2 = fnp.array(tC2, dtype=f32)
        cLR = fnp.array(tLR, dtype=f32)
        HE = fnp.array([[e / 2.0] for e in _EXP], dtype=f32)  # sigma^e = var^(e/2)
        PEXP = fnp.array([[float(p)] for p in range(_NP)], dtype=f32)
        e_sig = fnp.array([1.0 if e == -1 else 0.0 for e in _EXP], dtype=f32)  # picks sigma^-1
        e_pk1 = fnp.array([1.0 if r == 1 else 0.0 for r in range(_NPX)], dtype=f32)  # picks pK1 from Pext
        e_l, e_r, e_ll, e_rl = (fnp.array([1.0 if j == i else 0.0 for j in range(4)], dtype=f32) for i in range(4))
        c_a13 = fnp.array([6.0, 1.0], dtype=f32)  # [3 A1, 3 A3] = [6 v HG, 1 GG]  (cv = 3 v)
        c_a2 = fnp.array([2.0, 3.0], dtype=f32)  # 3 A2 = 2 * 1 * EG + 3 * v * HH
        c_t = fnp.array([9.0, 3.0], dtype=f32)  # tA + tB = 3 t3 = 9 v HHG + 3 EGG (last layer)
        one0 = fnp.array(1.0, dtype=f32)
        onen = fnp.ones(n, dtype=f32)
        zn = fnp.zeros(n, dtype=f32)
        Zs = fnp.array([1.0, 0.0, 0.0], dtype=f32)  # [1, lin, sp11] before c4 exists

        M = fnp.empty((6, n, n), dtype=f32)  # [S0, 3T^T, 3T, Q, J, I]; empty, not zeros: no inferred symmetry tag
        M0, M1, M2, M3, M4, M5 = (M[k] for k in range(6))
        MZ = M[1::4]  # [3T^T, I]: their diagonals are [tA = 3 t3, 1], read by the Pext einsum before M1's is cleared
        M1T = M1.T
        cp(M5, fnp.eye(n, dtype=f32))
        fnp.subtract(1.0, M5, out=M4)
        PX = fnp.zeros((depth, _NPX, n), dtype=f32)  # Pext of every layer; row 1 is the mean, row _P_V is v
        if depth > 1:
            # legs stored transposed: rows (3s + x) n + j, columns i (the current layer), x = [H, E, G]
            LEGt = fnp.zeros((3 * S * n, n), dtype=f32)
            LEGt3 = LEGt.reshape((3 * S, n, n))
            LEGt4 = LEGt.reshape((S, 3, n, n))
            LEGtT = LEGt.T
            ASt = fnp.zeros((3 * S * n, n), dtype=f32)  # [A1, A3, A2] per source, pairing with [H, E, G]
            ASt4 = ASt.reshape((S, 3, n, n))
            Gt = LEGt3[2::3]
            A13t, A2t = ASt4[:, 0:2], ASt4[:, 2]
            HE_, HG, EH, GH = LEGt4[:, 0:2], LEGt4[:, 0::2], LEGt4[:, 1::-1], LEGt4[:, 2::-2]  # leg pairs per source
            V1 = PX[:S, 0:_P_V + 1:_P_V]  # Pext rows (1, v) of every source's layer
            Vv1 = PX[:S, _P_V::-_P_V]  # Pext rows (v, 1)

        Pprev = None  # Pext of the previous layer (its row 1 is the mean)
        Cv = None  # previous layer's C' (a view of the newest source's G slot)
        ns = 0
        for t in range(depth):
            W = weights[t]
            if _dtname(W.dtype) != _dtname(f32):  # compare names: client dtypes are not numpy dtypes
                W = W.astype(f32)
            last = t == depth - 1

            # ---- linear step ----
            with _ns("cov"):
                if t == 0:
                    if last:
                        var = es("ia,ia->a", W, W)
                    else:
                        es("ia,ib->ab", W, W, out=M0)  # W^T W, symmetric: billed at half
                        cp(M[1:3], 0.0)  # no T before the first source
                else:
                    Cs = flops.as_symmetric(Cv, symmetry=(0, 1))
                    if last:
                        var = es("ij,ia,ja->a", Cs, W, W)
                    else:
                        es("ij,ia,jb->ab", Cs, W, W, out=M0)
                if not last:
                    var = fnp.diagonal(M0)
                var = fnp.maximum(var, 1e-10)  # the reference's relu_wick_coef clamp (copies the diagonal)
                if not last:
                    fnp.fill_diagonal(M0, 0.0)  # S0
                    mul(M0, M0, out=M3)  # Q

            # ---- third cumulant of the preactivation: push the legs through W^T (rows of the transposed slab) ----
            with _ns("k3_push"):
                if ns:
                    blk = LEGt[:(3 * ns - 1) * n]  # all legs but the newest source's G (= I before the push)
                    fnp.matmul(blk, W, out=blk)
                    cp(Cv, W)  # newest G^T = W (C' sat in this slot until the covariance step above)

            # ---- slices: t3 = K3_iii (as tA + tB = 3 t3) and 3T^T = L A^T ----
            Z0 = None
            with _ns("k3_slices"):
                if ns and last:
                    Z0 = fnp.stack([onen, es("sxji,sxji,sji,sxj,x->i", HE_, HG, Gt, Vv1, c_t), zn])  # [1, tA + tB, 0]
                elif ns:
                    es("sxji,sji,sxj,x->sxji", HG, Gt, Vv1, c_a13, out=A13t)  # [3 A1, 3 A3] = [2 cv HG, GG]
                    es("sxji,sxji,sxj,x->sji", EH, GH, V1, c_a2, out=A2t)  # 3 A2 = cv HH + 2 EG
                    fnp.matmul(LEGtT[:, :3 * ns * n], ASt[:3 * ns * n], out=M1)  # 3 T^T; its diagonal is 3 t3

            # ---- per-neuron algebra: Wick coefficients, power cumulants, cumulants ----
            with _ns("wick_pk"):
                SV = fnp.power(var, HE)  # sigma^-3..sigma^4
                a = zn if t == 0 else es("ia,ri,r,ea,e->a", W, Pprev, e_pk1, SV, e_sig)  # m / sigma
                Mon = fnp.power(a, PEXP)
                FF = fnp.stack([flops.stats.norm.pdf(a), flops.stats.norm.cdf(a), onen], dtype=f32)
                Pext = PX[t]
                if Z0 is None and last:  # depth 1: no source, no M1
                    Z0 = fnp.stack([onen, zn, zn])
                if Z0 is None:
                    es("rkwcpe,kii,w,ci,pi,ei->ri", cPz, MZ, Zs, FF, Mon, SV, out=Pext)
                else:
                    es("rzwcpe,zi,w,ci,pi,ei->ri", cP, Z0, Zs, FF, Mon, SV, out=Pext)
                if last:
                    break
                Cum2 = es("rab,ai,bi->ri", cC2, Pext, Pext)
                LR = es("yxkab,ai,bi->yxki", cLR, Cum2, Cum2)

            # ---- the new source (H, E), the next covariance C' and the c4 sums, in einsums over M ----
            with _ns("matrices"):
                if ns:
                    fnp.fill_diagonal(M1, 0.0)
                    cp(M2, M1T)  # 3 T
                es("kij,yxki,zxkj,y,z->xji", M, LR, LR, e_l, e_r, out=LEGt3[3 * ns:3 * ns + 3])
                Cv = LEGt3[3 * ns + 2]
                lin = es("kij,yzki,wzkj,y,w->", M, LR, LR, e_ll, e_rl)
                sp11 = es("ij,ij->", Cv, Cv)
                Zs = fnp.stack([one0, lin, sp11])  # c4 for the next layer = cc (lin - 2 sp11)
                if ns:  # old legs to the activation side: columns (current neurons) scaled by w11
                    Lt = LEGt[:3 * ns * n]
                    mul(Lt, Pext[_P_U], out=Lt)
            Pprev = Pext
            ns += 1
        return PX[:, 1, :]


# ----------------------------------------------------------------------------------------------------------------
# Augmented (law-based) engine at the scored shape
# ----------------------------------------------------------------------------------------------------------------
# lam' table of k3_acc_aug.LAM_REFIT (five layer-group scalars fitted on training mini MLPs 50-52)
LAM_REFIT = [0.01798947, 0.01538665, 0.01345267, 0.01320129, 0.01193519, 0.0108929, 0.00879968, 0.00786735,
             0.00747465, 0.00754678, 0.00731031, 0.00691034, 0.00820556, 0.00669035, 0.00686969, 0.0043575]
_AUG_SHAPE = (1024, 16)
_G = 2.0
_EXP2 = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
_NE2 = len(_EXP2)

# VEC rows (per neuron, one layer)
(V_ONE, V_W01, V_W11, V_W21, V_W31, V_W41, V_W51, V_RHO, V_W11T, V_W31T, V_W41T, V_W11D, V_W21D, V_W31D, V_W41D,
 V_W51D, V_RHOD, V_PK1, V_PK2, V_PK3, V_PK4, V_LAM, V_LAMN, V_C, V_LW31, V_LW21) = range(26)
_NV = 26
assert V_ONE == 0 and V_W21 == 3  # the views [1, v] = VX[:, 0:4:3] and [v, 1] = VX[:, 3::-3]


def _p2(*monos):  # (coef, F row, power of a, sigma exponent) -> list of (coef, c, p, e index)
    return [(float(coef), c, p, _EXP2.index(e)) for coef, c, p, e in monos]


_WICK = {  # w_k1 = E[d^k ReLU(Z)], and w_02, w_03, w_04 (as in k3_simple), in (F, a^p, sigma^e) monomials
    "w01": _p2((1, _PH, 0, 1), (1, _PHI, 1, 1)),
    "w11": _p2((1, _PHI, 0, 0)),
    "w21": _p2((1, _PH, 0, -1)),
    "w31": _p2((-1, _PH, 1, -2)),
    "w41": _p2((1, _PH, 2, -3), (-1, _PH, 0, -3)),
    "w51": _p2((-1, _PH, 3, -4), (3, _PH, 1, -4)),
    "w02": _p2((1, _PH, 1, 2), (1, _PHI, 2, 2), (1, _PHI, 0, 2)),
    "w03": _p2((1, _PH, 2, 3), (2, _PH, 0, 3), (1, _PHI, 3, 3), (3, _PHI, 1, 3)),
    "w04": _p2((1, _PH, 3, 4), (5, _PH, 1, 4), (1, _PHI, 4, 4), (6, _PHI, 2, 4), (3, _PHI, 0, 4)),
    "rho": _p2((-0.5, _ONE, 1, -1)),
    "one": _p2((1, _ONE, 0, 0)),
}


def _vec_rows():
    """Each VEC row as a list of (coef, base, mod): base a _WICK key, mod None, "t3", "d4", "lam", "lamn"."""
    R = [None] * _NV
    R[V_ONE] = [(1.0, "one", None)]
    for v, k in ((V_W01, "w01"), (V_W11, "w11"), (V_W21, "w21"), (V_W31, "w31"), (V_W41, "w41"), (V_W51, "w51"),
                 (V_RHO, "rho")):
        R[v] = [(1.0, k, None)]
    for v, k in ((V_W11T, "w11"), (V_W31T, "w31"), (V_W41T, "w41")):
        R[v] = [(1.0, k, "t3")]
    for v, k in ((V_W11D, "w11"), (V_W21D, "w21"), (V_W31D, "w31"), (V_W41D, "w41"), (V_W51D, "w51"), (V_RHOD, "rho")):
        R[v] = [(1.0, k, "d4")]
    R[V_PK1] = [(1.0, "w01", None), (1 / 6, "w31", "t3"), (1 / 24, "w41", "d4")]
    R[V_PK2] = [(1.0, "w02", None), (1 / 3, "w21", "t3"), (1 / 12, "w31", "d4")]
    R[V_PK3] = [(1.0, "w03", None), (1.0, "w11", "t3"), (0.25, "w21", "d4")]
    R[V_PK4] = [(1.0, "w04", None), (4.0, "w01", "t3"), (1.0, "w11", "d4")]
    R[V_LAM] = [(1.0, "one", "lam")]
    R[V_LAMN] = [(1.0, "one", "lamn")]
    R[V_C] = [(_G / 4.0, "w11", "lam")]  # c = (g / 4) lam' w11 (the feed's u = (g / 4) w21 enters as w21)
    R[V_LW31] = [(1.0, "w31", "lam")]  # lam'_t w31 and lam'_t w21 (the K4 (3,1) slice)
    R[V_LW21] = [(1.0, "w21", "lam")]
    return R


def _expand(coef, base, mod, mode):
    """(coef, k, d, c, p, e) terms. mode "mid": k over [1, 3 t3] (diagonals of [I, 3T^T]), d over [1, dA, lam, lamn];
    mode "last": k = 0 only, d over [1, dA, lam, lamn, t3b, t3f] with t3 = t3b / 3 + t3f."""
    out = []
    for cw, c, p, e in _WICK[base]:
        w = coef * cw
        if mod is None:
            out.append((w, 0, 0, c, p, e))
        elif mod == "t3":
            if mode == "mid":
                out.append((w / 3.0, 1, 0, c, p, e))
            else:
                out.append((w / 3.0, 0, 4, c, p, e))
                out.append((w, 0, 5, c, p, e))
        elif mod == "d4":  # d4 = g dA + g lam'_t sigma^2
            out.append((w * _G, 0, 1, c, p, e))
            out.append((w * _G, 0, 2, c, p, e + 2))
        elif mod == "lam":
            out.append((w, 0, 2, c, p, e))
        elif mod == "lamn":
            out.append((w, 0, 3, c, p, e))
    return out


def _vec_tables():
    rows = _vec_rows()
    mid = _zeros(_NV, 2, 3, 4, _NP, _NE2)
    for r, parts in enumerate(rows):
        for coef, base, mod in parts:
            for w, k, d, c, p, e in _expand(coef, base, mod, "mid"):
                mid[r][k][c][d][p][e] += w
    last = _zeros(1, 3, 6, _NP, _NE2)  # only pK1 at the last layer
    for coef, base, mod in rows[V_PK1]:
        for w, k, d, c, p, e in _expand(coef, base, mod, "last"):
            last[0][c][d][p][e] += w
    return mid, last


# ---- L/R coefficient polynomials (degree <= 2 in the VEC rows): dict {(a, b): coef}, a <= b, ONE = row 0 ----
def _lin(*pairs):
    out = {}
    for v, c in pairs:
        key = (V_ONE, v)
        out[key] = out.get(key, 0.0) + float(c)
    return {k: v for k, v in out.items() if v != 0.0}


def _padd(*ps):
    out = {}
    for s, p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0.0) + s * v
    return {k: v for k, v in out.items() if v != 0.0}


def _pmul(p, q):
    out = {}
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            f = sorted(x for x in (a1, b1, a2, b2) if x != V_ONE)
            assert len(f) <= 2, "degree > 2"
            key = (V_ONE, V_ONE) if not f else ((V_ONE, f[0]) if len(f) == 1 else (f[0], f[1]))
            out[key] = out.get(key, 0.0) + c1 * c2
    return {k: v for k, v in out.items() if v != 0.0}


def _w(k, p, mod=None):
    """w_{k,p} (times t3 or d4) as a VEC polynomial; w_{k,2} = 2 w_{k-1,1}."""
    t = {None: {0: V_W01, 1: V_W11, 2: V_W21, 3: V_W31, 4: V_W41, 5: V_W51},
         "t3": {1: V_W11T, 3: V_W31T, 4: V_W41T},
         "d4": {1: V_W11D, 2: V_W21D, 3: V_W31D, 4: V_W41D, 5: V_W51D}}
    if p == 1:
        return _lin((t[mod][k], 1.0))
    if mod is None:
        return _lin((t[None][k - 1], 2.0))
    return _lin((t[mod][k - 1], 2.0))


# slots of M
K_S0, K_M1, K_M2, K_Q, K_ST1, K_ST2, K_S3, K_J, K_I = range(9)
_NK = 9
_KT = {K_S0: K_S0, K_M1: K_M2, K_M2: K_M1, K_Q: K_Q, K_ST1: K_ST2, K_ST2: K_ST1, K_S3: K_S3, K_J: K_J, K_I: K_I}


def _lw3(p):  # (g / 2) lam'_t w_{3,p}: the K4 (3,1) slice's lamk times w_{3,p}
    return _lin((V_LW31, _G / 2.0)) if p == 1 else _lin((V_LW21, _G))


def _diagrams():
    """P_pq[j, k] = sum_d coef M_slot[j, k] f_d(p)(j) g_d(q)(k): the AUG term set of k3_acc_aug (SIMPLE terms with the
    per-neuron K4, the K4 (2,2) and (3,1) slices, the AUG-only two-index diagrams), power-generalized."""
    return [
        (K_S0, lambda p: _w(1, p), lambda q: _w(1, q), 1.0),
        (K_Q, lambda p: _w(2, p), lambda q: _w(2, q), 0.5),
        (K_J, lambda p: _w(2, p, "d4"), lambda q: _w(2, q), 1 / 24),
        (K_J, lambda p: _w(2, p), lambda q: _w(2, q, "d4"), 1 / 24),
        (K_M1, lambda p: _w(1, p), lambda q: _w(2, q), 1 / 6),  # T^T = M1 / 3, coefficient 1/2
        (K_M2, lambda p: _w(2, p), lambda q: _w(1, q), 1 / 6),
        (K_S0, lambda p: _w(1, p), _lw3, 1 / 6),  # K4 (3,1) slice: lamk S0
        (K_S0, _lw3, lambda q: _w(1, q), 1 / 6),
        (K_S0, lambda p: _w(1, p), lambda q: _padd((1 / 6, _w(4, q, "t3")), (1 / 24, _w(5, q, "d4"))), 1.0),
        (K_S0, lambda p: _padd((1 / 6, _w(4, p, "t3")), (1 / 24, _w(5, p, "d4"))), lambda q: _w(1, q), 1.0),
        (K_S3, lambda p: _w(3, p), lambda q: _w(3, q), 1 / 6),
        (K_S0, lambda p: _w(3, p, "d4"), lambda q: _w(3, q), 1 / 24),
        (K_S0, lambda p: _w(3, p), lambda q: _w(3, q, "d4"), 1 / 24),
        (K_ST1, lambda p: _w(2, p), lambda q: _w(3, q), 1 / 6),  # S0 T^T = ST1 / 3, coefficient 1/2
        (K_ST2, lambda p: _w(3, p), lambda q: _w(2, q), 1 / 6),
    ]


def _hprime_terms():
    """H'[i, m] = w11_i S0 + (w21_i / 2) T_im + rho_m [w21_i Q + w11_i T_mi + (w21_i / 2) K4_iimm]."""
    one = _lin((V_ONE, 1.0))
    return [(K_S0, _lin((V_W11, 1.0)), one, 1.0),
            (K_M2, _lin((V_W21, 1.0)), one, 1 / 6),
            (K_Q, _lin((V_W21, 1.0)), _lin((V_RHO, 1.0)), 1.0),
            (K_M1, _lin((V_W11, 1.0)), _lin((V_RHO, 1.0)), 1 / 3),
            (K_J, _lin((V_W21D, 1.0)), _lin((V_RHO, 1.0)), 1 / 12),
            (K_J, _lin((V_W21, 1.0)), _lin((V_RHOD, 1.0)), 1 / 12)]


def _transpose(terms):
    return [(_KT[k], R, L, c) for k, L, R, c in terms]


def _window_terms(n):
    """Terms (slot, L, R, coef) of the five outputs, in the slab's orientation [row, column]."""
    D = _diagrams()
    one = _lin((V_ONE, 1.0))
    pk1 = _lin((V_PK1, 1.0))
    m2pk1 = _lin((V_PK1, -2.0))

    def cent(f):  # power-2 factor minus 2 pK1 times the power-1 factor
        return _padd((1.0, f(2)), (1.0, _pmul(m2pk1, f(1))))

    P11 = [(k, f(1), g(1), c) for k, f, g, c in D]
    K321 = [(k, cent(f), g(1), c) for k, f, g, c in D]
    K422 = [(k, cent(f), cent(g), c) for k, f, g, c in D]
    Hp = _hprime_terms()
    W0 = _transpose(Hp)  # H'^T
    # W1 = 3 Yc before its -3 w21_b H'_ab^2 part; Yc = K3_21 - w11_a^2 w11_b T - (1/3) u_a (H'_ab cf_b + H'_ba cf_a),
    # u = (g / 4) w21 and cf = lam' w11 = (4 / g) c; w21 rho = w31 / 2
    W1 = [(k, L, R, 3.0 * c) for k, L, R, c in K321]
    W1.append((K_M2, _pmul(_lin((V_W11, 1.0)), _lin((V_W11, 1.0))), _lin((V_W11, 1.0)), -1.0))
    for k, L, R, c in Hp:
        if k == K_S0:  # (1/3) u_a w11_a cf_b = (1/3) (c w21)_a w11_b: lam' moved to the a side (same R as K3_21's w11)
            W1.append((k, _pmul(_lin((V_C, 1.0 / 3.0)), _lin((V_W21, 1.0))), _lin((V_W11, 1.0)), -3.0 * c))
        else:
            W1.append((k, _pmul(_lin((V_W21, 1.0 / 3.0)), L), _pmul(_lin((V_C, 1.0)), R), -3.0 * c))
    w21r = {(V_ONE, V_ONE): _lin((V_W21, 1.0)), (V_ONE, V_RHO): _lin((V_W31, 0.5)),
            (V_ONE, V_RHOD): _lin((V_W31D, 0.5))}
    for k, L, R, c in _transpose(Hp):  # H'_ba = H'^T_ab: the a side carries H''s R factor
        (key, val), = L.items()
        W1.append((k, _pmul(_lin((V_C, val / 3.0)), w21r[key]), R, -3.0 * c))
    W1 += [(K_I, _lin((V_PK3, 1.0)), one, 1.0),  # diagonal x = K3_3 - w11^3 t3
           (K_I, _lin((V_PK2, -3.0)), pk1, 1.0),
           (K_I, _pmul(pk1, pk1), _lin((V_PK1, 2.0)), 1.0),
           (K_I, _pmul(_lin((V_W11, 1.0)), _lin((V_W11, -1.0))), _lin((V_W11T, 1.0)), 1.0)]
    # W2 = C' = P11 + diag(var_out)
    W2 = list(P11) + [(K_I, _lin((V_PK2, 1.0)), one, 1.0), (K_I, pk1, _lin((V_PK1, -1.0)), 1.0)]
    # W3 = R4: row sums = sum_k (K4_22 linear part) + K4_4 + 2 var^2 - lam'_n K^-1 var
    cc, c1 = n / 2.0 + 2.0, n / 2.0 + 1.0
    kd = 12.0 / (2.0 * n + 8.0)
    kt = 24.0 * (-1.0 / (16.0 * cc * (cc - 2.0)) + 1.0 / (32.0 * c1 * (c1 - 1.0)))
    kappa2 = kt / (kd * (kd + n * kt))  # K^-1 v = v / kd - kappa2 (sum v) 1
    kappa = kappa2 - 1.0 / kd
    pk1sq = _pmul(pk1, pk1)
    pk2 = _lin((V_PK2, 1.0))
    lamn = _lin((V_LAMN, 1.0))
    var_out = _padd((1.0, pk2), (-1.0, pk1sq))
    W3 = list(K422) + [
        (K_I, _lin((V_PK4, 1.0)), one, 1.0),
        (K_I, pk1, _lin((V_PK3, -4.0)), 1.0),
        (K_I, pk2, _padd((8.0, pk1sq), (-1.0, pk2), (kappa, lamn)), 1.0),
        (K_I, pk1sq, _padd((-4.0, pk1sq), (-kappa, lamn)), 1.0),
        (K_J, _lin((V_LAMN, kappa2)), var_out, 1.0)]
    return [W0, W1, W2, W3]


def _merge(terms):
    """Merge rank-1 terms (L, R) of one (output, slot): same R -> add the L's, same L -> add the R's."""
    items = [(dict(L), dict(R)) for L, R in terms]
    changed = True
    while changed:
        changed = False
        for side in (1, 0):
            groups = {}
            for L, R in items:
                key = tuple(sorted((L, R)[side].items()))
                if key in groups:
                    a, b = groups[key]
                    groups[key] = (_padd((1.0, a), (1.0, L)), b) if side == 1 else (a, _padd((1.0, b), (1.0, R)))
                    changed = True
                else:
                    groups[key] = (L, R)
            items = [(L, R) for L, R in groups.values() if L and R]
    return items


def _factor(terms):
    """Minimal-rank (L, R) pairs of sum_r L_r (x) R_r over the monomial basis (pivoted elimination)."""
    Z = {}
    for L, R in terms:
        for a, ca in L.items():
            for b, cb in R.items():
                Z[(a, b)] = Z.get((a, b), 0.0) + ca * cb
    scale = max((abs(v) for v in Z.values()), default=0.0)
    out = []
    while True:
        Z = {k: v for k, v in Z.items() if abs(v) > 1e-13 * scale}
        if not Z:
            return out
        (a0, b0), piv = max(Z.items(), key=lambda kv: abs(kv[1]))
        L = {a: v / piv for (a, b), v in Z.items() if b == b0}
        R = {b: v for (a, b), v in Z.items() if a == a0}
        out.append((L, R))
        for a, la in L.items():
            for b, rb in R.items():
                Z[(a, b)] = Z.get((a, b), 0.0) - la * rb


def _lr_table(n):
    W = _window_terms(n)
    per = {}
    for x, terms in enumerate(W):
        for k, L, R, c in terms:
            per.setdefault((x, k), []).append((_padd((c, L)), R))
    merged = {key: _factor(v) for key, v in per.items()}
    rank = max(len(v) for v in merged.values())
    mon = sorted({m for v in merged.values() for L, R in v for m in list(L) + list(R)})
    idx = {m: i for i, m in enumerate(mon)}
    T = _zeros(2, len(W), rank, _NK, len(mon))
    for (x, k), v in merged.items():
        for r, (L, R) in enumerate(v):
            for m, c in L.items():
                T[0][x][r][k][idx[m]] += c
            for m, c in R.items():
                T[1][x][r][k][idx[m]] += c
    SA, SB = _zeros(len(mon), _NV), _zeros(len(mon), _NV)  # one-hot selections: C2_c = VEC_a VEC_b
    for c, (a, b) in enumerate(mon):
        SA[c][a] = 1.0
        SB[c][b] = 1.0
    return T, [SA, SB], rank


_VEC_MID, _VEC_LAST = _vec_tables()
_LR_CACHE = {}


def _aug_lr(n):
    if n not in _LR_CACHE:
        _LR_CACHE[n] = _lr_table(n)
    return _LR_CACHE[n]


_aug_lr(_AUG_SHAPE[0])  # built at import for the scored width


# ---- exact-saving package (research/costfloor/make_ex_estimator.py): sparse forms of two coefficient tables ----
# E2 boxes (x0, x1, r0, r1, k0, k1) over the L/R stack: every (x, r, k) whose L and R rows are both non-zero lies in
# exactly one box (checked below); the first box of each output row x writes it, the others add
_EX_BOX = ((0, 1, 0, 1, 0, 4), (1, 4, 0, 4, 0, 1), (1, 4, 0, 1, 1, 7), (1, 2, 1, 2, 1, 4), (0, 1, 0, 2, 7, 8),
           (1, 4, 0, 4, 7, 9))
_EX_CACHE = {}


def _ex_tables(n):
    """E2: rows (y, box, x, r, k) of the L/R stack, each sum_f cp[q][f] base[ci[q][f]] with base_c = VEC_ai[c] VEC_bi[c]
    (the selections are one-hot with weight 1); E5: the VEC table over the monomials it uses."""
    if n in _EX_CACHE:
        return _EX_CACHE[n]
    T, (SA, SB), rank = _aug_lr(n)
    nc = len(SA)
    ai, bi = [], []
    for c in range(nc):
        a_ = [a for a in range(len(SA[c])) if SA[c][a] != 0.0]
        b_ = [b for b in range(len(SB[c])) if SB[c][b] != 0.0]
        assert len(a_) == 1 and len(b_) == 1 and SA[c][a_[0]] == 1.0 and SB[c][b_[0]] == 1.0
        ai.append(a_[0])
        bi.append(b_[0])
    nx, nk = len(T[0]), len(T[0][0][0])

    def nz(y, x, r, k):
        return r < rank and any(v != 0.0 for v in T[y][x][r][k])

    cover = {}
    for b, (x0, x1, r0, r1, k0, k1) in enumerate(_EX_BOX):
        for x in range(x0, x1):
            for r in range(r0, r1):
                for k in range(k0, k1):
                    assert (x, r, k) not in cover
                    cover[(x, r, k)] = b
    for x in range(nx):
        for r in range(rank):
            for k in range(nk):
                if nz(0, x, r, k) and nz(1, x, r, k):
                    assert (x, r, k) in cover, (x, r, k)
    rows, off = [[], []], []
    for b, (x0, x1, r0, r1, k0, k1) in enumerate(_EX_BOX):
        off.append(len(rows[0]))
        for y in (0, 1):
            for x in range(x0, x1):
                for r in range(r0, r1):
                    for k in range(k0, k1):
                        rows[y].append([(c, v) for c, v in enumerate(T[y][x][r][k]) if v != 0.0] if r < rank else [])
    rows = rows[0] + rows[1]
    nf = max(len(q) for q in rows)
    ci = [[c for c, _ in q] + [0] * (nf - len(q)) for q in rows]
    cpk = [[v for _, v in q] + [0.0] * (nf - len(q)) for q in rows]
    mono, cm = [], []
    for r_, Vr in enumerate(_VEC_MID):
        row = {}
        for k in range(len(Vr)):
            for c in range(len(Vr[k])):
                for d in range(len(Vr[k][c])):
                    for p in range(len(Vr[k][c][d])):
                        for e, v in enumerate(Vr[k][c][d][p]):
                            if v != 0.0:
                                m = (k, c, d, p, e)
                                if m not in mono:
                                    mono.append(m)
                                row[mono.index(m)] = row.get(mono.index(m), 0.0) + v
        cm.append(row)
    cm = [[row.get(m, 0.0) for m in range(len(mono))] for row in cm]
    _EX_CACHE[n] = {"ai": ai, "bi": bi, "ci": ci, "cp": cpk, "qy": len(rows) // 2, "off": off, "cm": cm,
                    "mono": [[m[j] for m in mono] for j in range(5)]}
    return _EX_CACHE[n]


_ex_tables(_AUG_SHAPE[0])


# ---- saturation-aware restriction (research/costfloor/make_sat_estimator.py; idea: ARC R7 10-06) ----
# keep fractions per layer t = 0..15: p99 over 392 dumps of this engine at the scored shape of the count of
# neurons with alpha > -th (alive_th) or |alpha| < th (band_th) (research/costfloor/sat_pad_tables.json)
_SAT = {'in': 1.75, 'out': 4.0, 'ta': 1.5, 'tb': 3.0, 'tb_band': True, 'mode': 'bins', 'q': 32, 'fbuf': True, 'e1c': 0, 'e1cv': 8}
_SAT_FRAC = {"alive_2.5": [1.0, 1.0, 0.999, 0.9893, 0.9736, 0.9572, 0.9386, 0.9219, 0.9053, 0.8907, 0.8763, 0.8637, 0.8458, 0.839, 0.8215, 0.8088], "band_2.5": [1.0, 1.0, 0.9962, 0.9727, 0.9395, 0.9035, 0.8711, 0.8303, 0.7993, 0.764, 0.7386, 0.7173, 0.6865, 0.6625, 0.6364, 0.6152], "alive_3.0": [1.0, 1.0, 1.0, 0.999, 0.9923, 0.9814, 0.9678, 0.9561, 0.9424, 0.9326, 0.9133, 0.9062, 0.8927, 0.8781, 0.8711, 0.8577], "band_3.0": [1.0, 1.0, 1.0, 0.9951, 0.9796, 0.9571, 0.9327, 0.9044, 0.8762, 0.8536, 0.8242, 0.8011, 0.7813, 0.7588, 0.7345, 0.7053], "alive_3.5": [1.0, 1.0, 1.0, 1.0, 0.998, 0.9942, 0.9863, 0.9786, 0.9688, 0.961, 0.9492, 0.9366, 0.9277, 0.9154, 0.9093, 0.8937], "band_3.5": [1.0, 1.0, 1.0, 1.0, 0.9951, 0.9854, 0.9718, 0.9523, 0.9346, 0.9103, 0.8898, 0.8667, 0.8507, 0.8313, 0.8116, 0.7844], "alive_4.0": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9981, 0.9961, 0.9912, 0.9854, 0.9777, 0.9727, 0.963, 0.9542, 0.9444, 0.9389, 0.9259], "band_4.0": [1.0, 1.0, 1.0, 1.0, 0.999, 0.9961, 0.9893, 0.9785, 0.965, 0.9522, 0.9376, 0.919, 0.9044, 0.8869, 0.874, 0.8468]}


_SAT_BINS = {"push": {"11": [0.9375, 0.96875], "12": [0.9375, 0.96875], "13": [0.9375, 0.96875], "14": [0.90625, 0.96875]}, "tt": {"1": [], "2": [[0.9375, 1.0]], "3": [[0.9375, 1.0]], "4": [[0.875, 1.0], [0.9375, 1.0]], "5": [[0.875, 0.9375], [0.875, 1.0]], "6": [[0.8125, 0.9375], [0.8125, 1.0], [0.875, 0.9375], [0.875, 1.0]], "7": [[0.8125, 0.875], [0.8125, 0.9375], [0.875, 0.9375]], "8": [[0.75, 0.875], [0.75, 0.9375], [0.8125, 0.875], [0.8125, 0.9375]], "9": [[0.75, 0.8125], [0.75, 0.875], [0.8125, 0.8125], [0.8125, 0.875]], "10": [[0.75, 0.8125], [0.75, 0.875], [0.8125, 0.8125], [0.8125, 0.875]], "11": [[0.75, 0.75], [0.75, 0.875], [0.8125, 0.875]], "12": [[0.75, 0.75], [0.75, 0.8125]], "13": [[0.6875, 0.6875], [0.75, 0.6875], [0.6875, 0.8125], [0.75, 0.8125]], "14": [[0.6875, 0.6875], [0.75, 0.6875], [0.6875, 0.8125], [0.75, 0.8125]]}, "skip_plain": [["tt", 7], ["tt", 8], ["tt", 9], ["tt", 10], ["push", 11], ["tt", 11], ["push", 12], ["tt", 12], ["push", 13], ["tt", 13], ["push", 14], ["tt", 14]]}  # r25b op: tt bins for (ta, tb) = (1.5, 3.0), research/r25b/ttbins.py k = 2; research/costfloor/satbins_table.json (k = 2, 392 mini MLPs)

_SAT_BINS_CACHE = {}


def _sat_pad(f, n, mult):
    """A bin's size at width n: fraction f of n rounded up to mult (the sat step's padding), at most n."""
    x = f * n
    k_ = int(x // mult) + (1 if x % mult > 1e-9 else 0)
    return min(n, k_ * mult)


def _sat_bins(t, n, d, dtt, q_, cf):
    """mode bins: layer t's restricted push sizes (none on the fold layers) and 3 T^T (Pa, Pb) pairs at width n, the
    pairs cheapest first; cached per argument set."""
    key = (t, n, d, dtt, q_, cf)
    v = _SAT_BINS_CACHE.get(key)
    if v is None:
        pu = [] if (t < 2 or t in cf) else sorted({_sat_pad(f, n, max(1 << d, q_)) for f in _SAT_BINS["push"].get(str(t), ())})
        tt = {(_sat_pad(fa, n, max(1 << dtt, q_)), _sat_pad(fb, n, max(1 << dtt, q_))) for fa, fb in _SAT_BINS["tt"].get(str(t), ())}
        v = _SAT_BINS_CACHE[key] = ([p for p in pu if p < n], sorted((p for p in tt if p[0] < n or p[1] < n), key=lambda p: (p[0] * p[1], p)))
    return v

_SAT_FRAC.update({'alive_2.0': [1.0, 1.0, 0.989, 0.965, 0.94, 0.917, 0.888, 0.874, 0.852, 0.838, 0.822, 0.81, 0.795, 0.784, 0.775, 0.759], 'alive_1.5': [1.0, 0.994, 0.954, 0.911, 0.881, 0.85, 0.82, 0.805, 0.786, 0.776, 0.762, 0.751, 0.735, 0.725, 0.716, 0.706]})
_SAT_FRAC['alive_1.75'] = [1.0, 0.999, 0.977, 0.944, 0.91, 0.885, 0.861, 0.841, 0.82, 0.809, 0.793, 0.783, 0.766, 0.756, 0.748, 0.733]  # r25b op: p99 alive fractions, 392 public dumps
_DC = {'theta': '2.0', 'gather': True}  # dead-center rule 'alive_<theta>'; gather False: zeros only
_R25B = {'s1': 4, 's1_bins': {4: [960], 5: [960], 6: [896, 960], 7: [896, 960], 8: [832, 896], 9: [832, 896], 10: [832, 896]}, 's8': {'out': 3.5, 'pairs': {11: [[768, 928], [768, 992], [832, 928], [832, 992]], 12: [[768, 928], [768, 960], [832, 928], [832, 960]], 13: [[768, 928], [768, 960], [800, 928], [800, 960]], 14: [[736, 896], [736, 960], [800, 896], [800, 960]]}}, 's2': {'band': 3.0, 'apa': False, 'g': 64, 'pairs': [[768, 704], [768, 768], [832, 704], [832, 768]], 'rb': True}, 's3': 1, 'esc': [(1, 3), (1, 2), (0, 1, 2)], 's7': 1, 's7_pb': [768, 832, 896, 960], 'ax': 1}  # R25-B cost cuts (research/r25b/make_r25b.py)


def _r25_pad(x, g):
    return int(-(-int(x) // g) * g)


def _r25_up(pairs, x, y, n):
    return next(((p, q) for p, q in pairs if p >= x and q >= y), (n, n))


def _r25_bview(swp, kind, dd, bc, m, K, N):
    """r25b: (depth, frames, None, None) of a batched product shape on kind's frames, cached in the plan."""
    key = ("r25b", kind, dd, tuple(bc), m, K, N)
    v = swp.views.get(key)
    if v is None:
        v = swp.views[key] = (dd, _rcall(swp.frames[(kind, dd)], tuple(bc), m >> dd, K >> dd, N >> dd), None, None)
        if swp.persist:
            v[1][8]["_orig"].update(id(x) for x in v[1][:8])
    return v
_QX = {'on': True, 'buf': None, 'k': 3, 'x': {4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10}, 'rows': 11}  # r25a qx: corrector v2 input taps (False: base)
_GF = {'scale': 0.5}  # r25a gf: the K4 star's G-slot supply folded into the dense G legs (None: base)
_K4F = {'scale': 1.0, 'live': True}  # r25a k4f: the K4 star (3,1) slice folded into 3 T^T (scale None: the base)
_X38 = {'on': True, 'scale': 1.0, 'eps': [1.490e-04, 2.270e-04, 2.740e-04, 2.930e-04, 2.930e-04, 3.010e-04, 2.950e-04, 2.880e-04, 2.800e-04, 2.790e-04, 2.760e-04, 2.660e-04, 2.580e-04, 2.480e-04, 2.340e-04]}  # r38: scale-mode extraction, eps_t = scale eps[t]
_WC = {'g': 1.0, 'floor': 0.02}  # r24 weighted compression (None: the base's per-leg normalization)
_R25B['kc'] = 'A'  # r25b kc: 'A', 'B' or None (off)


def _r25_kc_pin(t, n, d, cf, hp, cin_p, ord_p, gath):
    """r25b kc: the inner size the push of layer t + 1 reads the parked bases with (n: every column). Mirrors that
    push: s1 on the fold layers (cin_p padded to 2^ddf, the first bin at least that), s8 on the others (alive_in size,
    the first pair's inner size at least that; a pair overflow at t + 1 is not known yet and reads n there)."""
    t1 = t + 1
    if n != 1024 or cin_p is None or ord_p is None or not gath:
        return n
    if t1 in cf:
        if _R25B.get('s1') and t1 >= _R25B['s1']:
            p_ = _r25_pad(cin_p, 1 << (d + 1 if t1 in hp else d))
            return next((x for x in _R25B['s1_bins'].get(t1, []) if x >= p_), n)
        return n
    if _R25B.get('s8') and t1 in _R25B['s8']['pairs']:
        p_ = _sat_size("alive_" + str(_SAT["in"]), t, n, max(1 << d, _SAT.get("q", 32)), cin_p)
        return next((x for x, _ in _R25B['s8']['pairs'][t1] if x >= p_), n)
    return n


def _r25_kc(SCR, wr, k, n, ki, kc, kd, QRE, Qt, Ct4, tap_b):
    """r25b kc: the compression SVD on SCR's rows of the kept centers ki (both legs) and the kept columns kc; Qt (k, n)
    and Ct4 (2, n, k) get zeros elsewhere. Tap 6 (QRE) adds the energy dropped on the kept columns and, with tap_b, on
    the dropped columns kd (their energy less its projection on the left basis)."""
    es, mul = fnp.einsum, fnp.multiply
    A1 = fnp.take(SCR, fnp.concatenate([ki, fnp.add(ki, n)]), axis=0) if ki is not None else SCR
    A2 = fnp.take(A1, kc, axis=1) if kc is not None else A1
    Uc, sv, Vt = fnp.linalg.svd(A2, full_matrices=False, k=k)
    if kc is not None:
        fnp.copyto(Qt, 0.0)
        fnp.put_along_axis(Qt, fnp.broadcast_to(kc[None, :], (k, kc.shape[0])), Vt, axis=1)
    else:
        fnp.copyto(Qt, Vt)
    if ki is not None:
        r_ = ki.shape[0]
        fnp.copyto(Ct4, 0.0)
        ck = es("xjc,c,xj->xjc", Uc.reshape((2, r_, k)), sv, fnp.divide(1.0, fnp.take(wr, ki, axis=1)))
        fnp.put_along_axis(Ct4, fnp.broadcast_to(ki[None, :, None], (2, r_, k)), ck, axis=1)
    else:
        es("xjc,c,xj->xjc", Uc.reshape((2, n, k)), sv, fnp.divide(1.0, wr), out=Ct4)
    if QRE is not None:
        e_ = fnp.maximum(fnp.subtract(es("ri,ri->i", A2, A2), es("c,ci,ci->i", mul(sv, sv), Vt, Vt)), 0.0)
        if kc is None:
            fnp.add(QRE, e_, out=QRE)
        else:
            q_ = fnp.take(QRE, kc)
            fnp.put_along_axis(QRE, kc, fnp.add(q_, e_), axis=0)
            if tap_b:
                B_ = fnp.take(A1, kd, axis=1)
                pj = es("rc,rd->cd", Uc, B_)
                ed = fnp.maximum(fnp.subtract(es("rd,rd->d", B_, B_), es("cd,cd->d", pj, pj)), 0.0)
                qd = fnp.take(QRE, kd)
                fnp.put_along_axis(QRE, kd, fnp.add(qd, ed), axis=0)


def _dc_sizes(t, n, depth, age, k, Smc, d, cf=frozenset(), hp=frozenset()):
    """Push and 3 T^T rows of layer t when some live dense source has dropped centers (else None), from the fixed
    frames alone (the setup plan's view shapes): parked bases, then each dense source's kept H and G rows; the newest
    source's G slot is pushed in full when it carries C' (last layer) or C (fold layers), else not at all."""
    if _DC['theta'] is None or not _DC['gather'] or t < 1 or not d:
        return None
    Rs = [n] + [_sat_size('alive_' + _DC['theta'], s_, n, max(1 << d, 32)) for s_ in range(1, depth)]
    mc_ = min(Smc, max(0, t - age)) if Smc else 0
    dense = range(mc_, t)
    if not any(Rs[s_] < n for s_ in dense):
        return None
    rt = mc_ * k + sum(2 * Rs[s_] for s_ in dense)
    pad_ = (-(-Rs[t - 1] // (1 << (d + 1))) * (1 << (d + 1)) - Rs[t - 1]) if t in hp else 0  # newest H rows
    return rt - Rs[t - 1] + pad_ + (n if (t == depth - 1 or t in cf) else 0), rt


_SAT_DIAG = []  # mode mask (tests): (t, role, dropped sum of squares, total sum of squares)


def _sat_note(t, role, X, m):
    tot = float(fnp.sum(fnp.multiply(X, X)))
    kept = fnp.multiply(X, m)
    _SAT_DIAG.append((t, role, tot - float(fnp.sum(fnp.multiply(kept, kept))), tot))


def _sat_mask(idx, n, f32):
    """0/1 vector of length n, 1 on idx (mode mask, tests)."""
    m = fnp.zeros((n,), dtype=f32)
    fnp.put_along_axis(m, idx, fnp.ones(idx.shape, dtype=f32), axis=0)
    return m


def _sat_grab(pool, size, f32):
    """sat fbuf: the first flat buffer in pool with room for size floats (taken out of the pool), else a new one
    (not reached at the scored shape: SG, the dead LEGt blocks or the arena's free tail always have room)."""
    for i_, b_ in enumerate(pool):
        if b_ is not None and b_.shape[0] >= size:
            pool[i_] = None
            return b_[:size]
    return fnp.empty((size,), dtype=f32)


def _sat_size(rule, t, n, mult, count=None):
    """Rows kept by a rule: the layer's count (dyn) or its p99 fraction of n, rounded up to mult, at most n."""
    x = count if count is not None else _SAT_FRAC[rule][min(t, len(_SAT_FRAC[rule]) - 1)] * n
    k_ = int(x // mult) + (1 if x % mult > 1e-9 else 0)
    return min(n, k_ * mult)


# ----------------------------------------------------------------------------------------------------------------
# Learned mean correction (research/make_lc.py): feature table, folded network weights (fitted on mini 0-39)
# ----------------------------------------------------------------------------------------------------------------
_LC_RICH = (13, 14, 15)
_LC_FROWS = ['a', 's', 'Phi', 'phi', 'k3', 'k4', 'kA', 'R3n', 'R4n', 'E33n', 'E34n', 'E44n', 'W5n', 'p', 'readout', 'ones']
# (row, F row [phi, Phi, 1], power of a, D5 x, D5 y, coef); D5 = [1, k3, kA, lamn, sigma]
_LC_TERMS = [[0, 2, 1, 0, 0, 1.0], [1, 2, 0, 4, 0, 1.0], [2, 1, 0, 0, 0, 1.0], [3, 0, 0, 0, 0, 1.0], [4, 2, 0, 1, 0,
    1.0], [5, 2, 0, 2, 0, 2.0], [5, 2, 0, 3, 0, 2.0], [6, 2, 0, 2, 0, 1.0], [7, 0, 1, 1, 0, -0.16666666666666666], [8,
    0, 0, 2, 0, -0.08333333333333333], [8, 0, 0, 3, 0, -0.08333333333333333], [8, 0, 2, 2, 0, 0.08333333333333333], [8,
    0, 2, 3, 0, 0.08333333333333333], [9, 0, 0, 1, 1, 0.041666666666666664], [9, 0, 2, 1, 1, -0.08333333333333333], [9,
    0, 4, 1, 1, 0.013888888888888888], [10, 0, 1, 1, 2, -0.20833333333333334], [10, 0, 1, 1, 3, -0.20833333333333334],
    [10, 0, 3, 1, 2, 0.1388888888888889], [10, 0, 3, 1, 3, 0.1388888888888889], [10, 0, 5, 1, 2,
    -0.013888888888888888], [10, 0, 5, 1, 3, -0.013888888888888888], [11, 0, 0, 2, 2, -0.052083333333333336], [11, 0,
    0, 2, 3, -0.10416666666666667], [11, 0, 0, 3, 3, -0.052083333333333336], [11, 0, 2, 2, 2, 0.15625], [11, 0, 2, 2,
    3, 0.3125], [11, 0, 2, 3, 3, 0.15625], [11, 0, 4, 2, 2, -0.052083333333333336], [11, 0, 4, 2, 3,
    -0.10416666666666667], [11, 0, 4, 3, 3, -0.052083333333333336], [11, 0, 6, 2, 2, 0.003472222222222222], [11, 0, 6,
    2, 3, 0.006944444444444444], [11, 0, 6, 3, 3, 0.003472222222222222], [12, 0, 1, 0, 0, 3.0], [12, 0, 3, 0, 0, -1.0],
    [13, 0, 0, 0, 0, 1.0], [13, 0, 0, 2, 0, -0.08333333333333333], [13, 0, 0, 3, 0, -0.08333333333333333], [13, 0, 1,
    1, 0, -0.16666666666666666], [13, 0, 2, 2, 0, 0.08333333333333333], [13, 0, 2, 3, 0, 0.08333333333333333], [13, 1,
    1, 0, 0, 1.0], [14, 0, 0, 0, 4, 1.0], [14, 0, 0, 2, 4, -0.08333333333333333], [14, 0, 0, 3, 4,
    -0.08333333333333333], [14, 0, 1, 1, 4, -0.16666666666666666], [14, 0, 2, 2, 4, 0.08333333333333333], [14, 0, 2, 3,
    4, 0.08333333333333333], [14, 1, 1, 0, 4, 1.0], [15, 2, 0, 0, 0, 1.0]]


def _covfold_layers(n, depth, d, ht):
    """covfold: the layers 0 < t < depth - 1 whose push is unrestricted, from the same fixed-frame rule the sat step
    applies (data independent in mode fixed; no layer in the other modes)."""
    if not d or _SAT.get("mode") not in ("fixed", "bins"):  # mode bins keeps the fixed rule's fold set
        return frozenset()
    q_ = _SAT.get("q", 32)
    out, ord_p = set(), False
    for t in range(1, depth - 1):
        dtt = d + 1 if t in ht else d
        Pp = n
        if ord_p:
            Pp = max(_sat_size("alive_" + str(_SAT["in"]), t - 1, n, max(1 << d, q_)),
                     _sat_size("alive_" + str(_SAT["out"]), t, n, max(1 << d, q_)))
        rb = ("band_" if _SAT["tb_band"] else "alive_") + str(_SAT["tb"])
        Pa = _sat_size("alive_" + str(_SAT["ta"]), t, n, max(1 << dtt, q_))
        Pb = _sat_size(rb, t, n, max(1 << dtt, q_))
        nxt = _sat_size("alive_" + str(_SAT["in"]), t, n, max(1 << d, q_)) < n
        if Pp >= n:
            out.add(t)
        ord_p = Pp < n or Pa < n or Pb < n or nxt
    return frozenset(out)


def _lc_table():
    T = _zeros(len(_LC_FROWS), 3, 7, 5, 5)
    for r, c, p, x, y, v in _LC_TERMS:
        T[int(r)][int(c)][int(p)][int(x)][int(y)] += v
    return T


_LC_T = _lc_table()
# D5 rows from [1, 3 t3, dA, lam', 1] times sigma powers (rows of SV: sigma^-4..sigma^4)
_LC_SEL = [[0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1 / 3, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]]
# rich linear rows [S off-diagonal row sum / var, diagonal-only var / var, W^3 K3 / s^3, W^4 K4 / s^4]
_LC_RS = [[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0]]


def _lc_k34():
    """K3, K4 of a layer's activations from [1, pK1, pK2, pK3, pK4] (degree-4 table)."""
    T = _zeros(2, 5, 5, 5, 5)
    for k, (a, b, c, d), v in ((0, (3, 0, 0, 0), 1.0), (0, (2, 1, 0, 0), -3.0), (0, (1, 1, 1, 0), 2.0),
                               (1, (4, 0, 0, 0), 1.0), (1, (3, 1, 0, 0), -4.0), (1, (2, 2, 0, 0), -3.0),
                               (1, (2, 1, 1, 0), 12.0), (1, (1, 1, 1, 1), -6.0)):
        T[k][a][b][c][d] += v
    return T


_LC_K34 = _lc_k34()


def _lc_consts(f32, n):
    """The correction's constant arrays and its W^2, W^3, W^4 work buffer (Estimator.setup() builds them once)."""
    return (fnp.array(_LC_T, dtype=f32), fnp.array(_LC_SEL, dtype=f32), fnp.array(_LC_RS, dtype=f32),
            fnp.array(_LC_K34, dtype=f32), fnp.array([[float(p)] for p in range(7)], dtype=f32),
            fnp.array(_LC_W1A, dtype=f32), fnp.array(_LC_W2A, dtype=f32), fnp.array(_LC_W1B, dtype=f32),
            fnp.array(_LC_W2B, dtype=f32), fnp.empty((3, n, n), dtype=f32))
# layers 1-12: rows ['a', 's', 'Phi', 'phi', 'k3', 'k4', 'kA', 'R3n', 'R4n', 'E33n', 'E34n', 'E44n', 'W5n', 'p', 'readout', 'ones']
_LC_W1A = [[[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.67591594, -1.43591148, -1.91427647, 0.375957567,
    0.788609903, 1.79423, -0.19549532, -1.9080928, -0.518885119, -1.2375763, -0.868357991, -0.302654207, -0.508759435,
    1.45465866, -0.283179648, -1.15721226, 0, 0, 1]], [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252,
    -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963, 0.000616006732, -0.10860558, -0.184236964,
    -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886,
    0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289,
    0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552,
    -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746,
    0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0],
    [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764,
    -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814,
    0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508,
    0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307,
    34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817,
    16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231,
    -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494,
    2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407,
    -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079,
    -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115,
    -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674,
    -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815,
    -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627,
    -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649,
    -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288,
    -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664,
    -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0,
    0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612,
    -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401,
    -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629,
    0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232,
    0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.61900291,
    -1.4116435, -1.91032077, 0.387156651, 0.819120877, 1.75080686, -0.331018351, -2.19171, -0.391418248, -1.25995245,
    -0.882660347, -0.276911683, -0.563353548, 1.39153072, -0.531075456, -1.22713769, 0, 0, 1]], [[0.109164881,
    -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963,
    0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861,
    0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059,
    0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072,
    0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994,
    0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674,
    0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332,
    -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645,
    -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958,
    -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518,
    3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498,
    61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0,
    0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904,
    -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761,
    611.640494, 32.2672407, -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874,
    -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764,
    -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521,
    575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765,
    -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301,
    5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891,
    20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699,
    -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587,
    -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863,
    -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282,
    0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681,
    -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841,
    0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348,
    0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, -1, 0], [-1.56208988, -1.38737551, -1.90636507, 0.398355735, 0.84963185, 1.70738372, -0.466541382, -2.47532721,
    -0.263951376, -1.28232861, -0.896962702, -0.251169159, -0.61794766, 1.32840279, -0.778971264, -1.29706312, 0, 0,
    1]], [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.50517685, -1.36310753, -1.90240936, 0.409554819,
    0.880142823, 1.66396058, -0.602064413, -2.75894441, -0.136484505, -1.30470476, -0.911265057, -0.225426635,
    -0.672541772, 1.26527485, -1.02686707, -1.36698855, 0, 0, 1]], [[0.109164881, -0.0206606894, 0.00903607666,
    -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963, 0.000616006732, -0.10860558,
    -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355,
    0.209988886, 0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059, 0.879880234, 0.533746323,
    -0.0460492289, 0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072, 0.696906498, 0, 0, 0],
    [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994, 0.104305735, 0.171702943,
    -0.388171746, 0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0,
    0], [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573,
    3.05362764, -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0],
    [-6.51761814, 0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958, -0.088657638, 0.769716908,
    -1.38569508, 0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0],
    [16.3304307, 34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175,
    37.3492817, 16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441,
    -28.5775231, -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032,
    3.87208494, 2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494,
    32.2672407, -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874, -511.74011,
    -6.78961079, -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921,
    127.535115, -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707,
    -1001.23674, -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682,
    13867.8815, -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824,
    14437.0627, -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253,
    -5815.56649, -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334,
    20383.5288, -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868,
    -44809.1664, -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356,
    -47583.0523, 0, 0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282, 0.0104205345,
    -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681, -0.322466369,
    0.0954504401, -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841, 0.019619278,
    -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348, 0.0656880051,
    -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
    [-1.44826381, -1.33883955, -1.89845366, 0.420753903, 0.910653797, 1.62053743, -0.737587445, -3.04256161,
    -0.00901763321, -1.32708092, -0.925567413, -0.19968411, -0.727135884, 1.20214691, -1.27476288, -1.43691398, 0, 0,
    1]], [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.39135078, -1.31457157, -1.89449796, 0.431952986,
    0.94116477, 1.57711429, -0.873110476, -3.32617881, 0.118449238, -1.34945707, -0.939869768, -0.173941586,
    -0.781729996, 1.13901897, -1.52265869, -1.50683941, 0, 0, 1]], [[0.109164881, -0.0206606894, 0.00903607666,
    -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963, 0.000616006732, -0.10860558,
    -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355,
    0.209988886, 0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059, 0.879880234, 0.533746323,
    -0.0460492289, 0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072, 0.696906498, 0, 0, 0],
    [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994, 0.104305735, 0.171702943,
    -0.388171746, 0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0,
    0], [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573,
    3.05362764, -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0],
    [-6.51761814, 0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958, -0.088657638, 0.769716908,
    -1.38569508, 0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0],
    [16.3304307, 34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175,
    37.3492817, 16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441,
    -28.5775231, -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032,
    3.87208494, 2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494,
    32.2672407, -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874, -511.74011,
    -6.78961079, -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921,
    127.535115, -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707,
    -1001.23674, -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682,
    13867.8815, -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824,
    14437.0627, -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253,
    -5815.56649, -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334,
    20383.5288, -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868,
    -44809.1664, -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356,
    -47583.0523, 0, 0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282, 0.0104205345,
    -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681, -0.322466369,
    0.0954504401, -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841, 0.019619278,
    -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348, 0.0656880051,
    -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
    [-1.33443775, -1.29030359, -1.89054226, 0.44315207, 0.971675744, 1.53369115, -1.00863351, -3.60979602, 0.24591611,
    -1.37183323, -0.954172123, -0.148199062, -0.836324108, 1.07589103, -1.7705545, -1.57676484, 0, 0, 1]],
    [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.27752472, -1.26603561, -1.88658656, 0.454351154,
    1.00218672, 1.49026801, -1.14415654, -3.89341322, 0.373382982, -1.39420938, -0.968474479, -0.122456538,
    -0.89091822, 1.01276309, -2.01845031, -1.64669026, 0, 0, 1]], [[0.109164881, -0.0206606894, 0.00903607666,
    -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963, 0.000616006732, -0.10860558,
    -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355,
    0.209988886, 0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059, 0.879880234, 0.533746323,
    -0.0460492289, 0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072, 0.696906498, 0, 0, 0],
    [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994, 0.104305735, 0.171702943,
    -0.388171746, 0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0,
    0], [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573,
    3.05362764, -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0],
    [-6.51761814, 0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958, -0.088657638, 0.769716908,
    -1.38569508, 0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0],
    [16.3304307, 34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175,
    37.3492817, 16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441,
    -28.5775231, -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032,
    3.87208494, 2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494,
    32.2672407, -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874, -511.74011,
    -6.78961079, -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921,
    127.535115, -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707,
    -1001.23674, -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682,
    13867.8815, -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824,
    14437.0627, -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253,
    -5815.56649, -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334,
    20383.5288, -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868,
    -44809.1664, -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356,
    -47583.0523, 0, 0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282, 0.0104205345,
    -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681, -0.322466369,
    0.0954504401, -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841, 0.019619278,
    -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348, 0.0656880051,
    -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
    [-1.22061168, -1.24176762, -1.88263085, 0.465550238, 1.03269769, 1.44684487, -1.27967957, -4.17703042, 0.500849853,
    -1.41658553, -0.982776834, -0.0967140132, -0.945512332, 0.94963515, -2.26634611, -1.71661569, 0, 0, 1]],
    [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.16369865, -1.21749964, -1.87867515, 0.476749321,
    1.06320866, 1.40342173, -1.4152026, -4.46064763, 0.628316725, -1.43896169, -0.997079189, -0.0709714889,
    -1.00010644, 0.88650721, -2.51424192, -1.78654112, 0, 0, 1]], [[0.109164881, -0.0206606894, 0.00903607666,
    -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662, -0.00743083963, 0.000616006732, -0.10860558,
    -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112, -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355,
    0.209988886, 0.273670716, 0.487303198, -0.717340286, -0.461826853, 0.219871059, 0.879880234, 0.533746323,
    -0.0460492289, 0.591669942, 0.0535283315, 0.409596943, -0.683877351, -0.568245072, 0.696906498, 0, 0, 0],
    [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589, 0.203541209, -0.774840994, 0.104305735, 0.171702943,
    -0.388171746, 0.206087327, 0.450045169, -0.139948331, -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0,
    0], [0.74304932, 0.588239493, -0.0734142032, -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573,
    3.05362764, -0.367357405, -3.04768053, -1.24419152, -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0],
    [-6.51761814, 0.230704197, -1.72620385, -0.652349823, -1.37455443, -7.88551958, -0.088657638, 0.769716908,
    -1.38569508, 0.507686492, 4.43187456, -1.12182507, -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0],
    [16.3304307, 34.6146776, 22.1509274, -26.7382808, -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175,
    37.3492817, 16.0813059, 20.7263164, -3.47152332, -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441,
    -28.5775231, -178.611181, 30.5946915, 9.16977966, 27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032,
    3.87208494, 2.09626588, -61.9674657, -31.1756717, -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494,
    32.2672407, -279.646783, 687.842838, 427.841579, 384.15802, 210.377832, 114.315685, 128.254874, -511.74011,
    -6.78961079, -686.185385, -631.916647, -97.6338552, 0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921,
    127.535115, -871.591764, 1296.14251, 658.610185, -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707,
    -1001.23674, -55.603342, -347.043709, -320.578453, 0, 0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682,
    13867.8815, -4382.25786, -11829.6291, -5783.64412, -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824,
    14437.0627, -15976.2208, -13242.9118, 0, 0, 0], [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253,
    -5815.56649, -17313.733, -18799.3888, -5686.87772, 11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334,
    20383.5288, -8282.32671, 0, 0, 0], [-46901.3041, 77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868,
    -44809.1664, -15982.2642, -13942.3549, -32608.814, 11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356,
    -47583.0523, 0, 0, 0], [0.889426294, -0.26140947, 0.0574921061, 0.493942758, 0.199761282, 0.0104205345,
    -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842, -0.948982429, 0.1167347, 0.145420681, -0.322466369,
    0.0954504401, -0.473506449, 0, 0, 0], [0.19502018, -0.0894217033, 0.0280212854, 0.138326841, 0.019619278,
    -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049, 0.00310417011, -0.0791072595, 0.240253348, 0.0656880051,
    -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
    [-1.10678562, -1.19323166, -1.87471945, 0.487948405, 1.09371964, 1.35999859, -1.55072563, -4.74426483, 0.755783596,
    -1.46133784, -1.01138154, -0.0452289647, -1.05470056, 0.823379271, -2.76213773, -1.85646655, 0, 0, 1]],
    [[0.109164881, -0.0206606894, 0.00903607666, -0.0358417252, -0.166515756, -0.0422210583, -0.0496728662,
    -0.00743083963, 0.000616006732, -0.10860558, -0.184236964, -0.0656998123, -0.0125784431, 0.0361042112,
    -0.0281204669, 0.109960861, 0, 0, 0], [0.799489355, 0.209988886, 0.273670716, 0.487303198, -0.717340286,
    -0.461826853, 0.219871059, 0.879880234, 0.533746323, -0.0460492289, 0.591669942, 0.0535283315, 0.409596943,
    -0.683877351, -0.568245072, 0.696906498, 0, 0, 0], [0.0299789552, -0.0214855934, 0.0360590754, 0.140645589,
    0.203541209, -0.774840994, 0.104305735, 0.171702943, -0.388171746, 0.206087327, 0.450045169, -0.139948331,
    -0.374191786, 0.0711474674, 0.0246122349, 1.33145368, 0, 0, 0], [0.74304932, 0.588239493, -0.0734142032,
    -0.31348402, 1.38911975, -1.47186332, -2.07298684, 1.48411573, 3.05362764, -0.367357405, -3.04768053, -1.24419152,
    -0.403018878, 1.54552013, 1.95667645, -1.01300986, 0, 0, 0], [-6.51761814, 0.230704197, -1.72620385, -0.652349823,
    -1.37455443, -7.88551958, -0.088657638, 0.769716908, -1.38569508, 0.507686492, 4.43187456, -1.12182507,
    -2.99831402, 1.11851445, -1.39377518, 3.36908656, 0, 0, 0], [16.3304307, 34.6146776, 22.1509274, -26.7382808,
    -23.3746117, 3.07417236, 42.4876498, 61.4012919, -11.7219175, 37.3492817, 16.0813059, 20.7263164, -3.47152332,
    -32.2022976, 17.283142, 35.6360811, 0, 0, 0], [86.4262441, -28.5775231, -178.611181, 30.5946915, 9.16977966,
    27.574585, -5.05481189, 5.65433137, 92.0941904, -113.084032, 3.87208494, 2.09626588, -61.9674657, -31.1756717,
    -29.3840668, 31.8058611, 0, 0, 0], [-130.325761, 611.640494, 32.2672407, -279.646783, 687.842838, 427.841579,
    384.15802, 210.377832, 114.315685, 128.254874, -511.74011, -6.78961079, -686.185385, -631.916647, -97.6338552,
    0.450716189, 0, 0, 0], [-1240.20886, -106.144764, -648.685921, 127.535115, -871.591764, 1296.14251, 658.610185,
    -112.578964, 1110.70422, 900.013357, -2694.78521, 575.466707, -1001.23674, -55.603342, -347.043709, -320.578453, 0,
    0, 0], [-2822.13455, -21701.0989, -17585.6765, -2427.93682, 13867.8815, -4382.25786, -11829.6291, -5783.64412,
    -1806.47018, -20978.4304, -1180.56877, -41558.2301, 5084.7824, 14437.0627, -15976.2208, -13242.9118, 0, 0, 0],
    [12411.8763, -24116.351, 2111.56313, -27044.1891, 20801.8253, -5815.56649, -17313.733, -18799.3888, -5686.87772,
    11780.2447, 25499.081, -9226.46132, 31242.0699, -3086.77334, 20383.5288, -8282.32671, 0, 0, 0], [-46901.3041,
    77165.374, -108770.421, -32871.4186, -30524.6587, -74483.5868, -44809.1664, -15982.2642, -13942.3549, -32608.814,
    11329.3053, 20694.6955, 61520.6399, -107880.863, -72368.6356, -47583.0523, 0, 0, 0], [0.889426294, -0.26140947,
    0.0574921061, 0.493942758, 0.199761282, 0.0104205345, -0.0553090612, -0.0975036301, 0.00935503619, -0.029782842,
    -0.948982429, 0.1167347, 0.145420681, -0.322466369, 0.0954504401, -0.473506449, 0, 0, 0], [0.19502018,
    -0.0894217033, 0.0280212854, 0.138326841, 0.019619278, -0.0903205629, 0.0707357757, -0.0306954068, 0.267075049,
    0.00310417011, -0.0791072595, 0.240253348, 0.0656880051, -0.389289232, 0.00633571985, 0.173701057, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-1.04987259, -1.16896368, -1.87076375, 0.499147489,
    1.12423061, 1.31657545, -1.68624866, -5.02788203, 0.883250468, -1.483714, -1.0256839, -0.0194864404, -1.10929467,
    0.760251332, -3.01003354, -1.92639198, 0, 0, 1]]]
_LC_W2A = [1.18886108e-05, 1.0797162e-05, 8.06381128e-06, -1.30153975e-05, -1.26797307e-05, 1.4678989e-05,
    -1.91612146e-05, 4.00326913e-05, -1.46743644e-05, -1.17989633e-05, 1.63794115e-05, -9.9765367e-06, 1.38094921e-05,
    -1.59438149e-05, 1.88250654e-05, 1.14257466e-05, 1, -1, 3.81774501e-06]
# layers 13-15: rows ['a', 's', 'Phi', 'phi', 'k3', 'k4', 'kA', 'R3n', 'R4n', 'E33n', 'E34n', 'E44n', 'W5n', 'p', 'readout', 'ones', 'Lsoffsq', 'Lcoff', 'Ls0', 'Lt3row', 'Lt3col', 'Lvar', 'soffsum', 'dfrac', 'dk3', 'dk4']
_LC_W1B = [[[0.0121698474, 0.0480854299, 0.00812933353, -0.0126857349, -0.00793664811, 0.0575442617, 0.0893989663,
    -0.005464908, 0.0124702079, -0.0811655324, 0.0254390072, 0.0613470236, -0.0840409351, -0.0122381987,
    -0.00547355718, -0.0152790681, -0.0291967699, -0.0300083087, 0.0277645446, 0.105752801, -0.00943615054,
    -0.0117494846, -0.0082033514, 0.0466504246, 0.0608832031, 0.0139216943, 0.0491026377, -0.0687410946, -0.0432446185,
    0.0178622326, -0.0628037523, -0.0583070886, 0, 0, 0], [-2.52402159, 1.83892744, 2.66762092, -0.510325866,
    -0.223316421, -1.35527324, 0.845233247, -0.249054807, -2.28590209, 6.96418165, 6.07701097, 1.18815813, 4.93789867,
    3.59777272, 5.45303674, 8.42689744, 9.89084834, 5.16469545, -2.27999674, -5.4078195, 1.91225383, 1.5138683,
    -0.255444743, -0.00614493243, 0.931412372, 4.18421106, -5.1689647, -2.50948022, 0.397510684, 2.06279938,
    2.53763887, -5.2451757, 0, 0, 0], [0.0315222315, -0.103147981, 0.62173113, 0.279799203, 0.468465454, -1.07305389,
    0.455366578, 0.159708343, 0.488398777, -0.452518913, 0.464655169, 0.162000685, -0.253787946, -0.0930729909,
    0.0325427201, 0.51420894, 0.231743299, -0.627926207, 0.305095162, 0.467596274, -0.708040549, 0.0685000038,
    0.0562234241, 0.60154213, -0.70632214, 0.516864429, 0.0360230365, 0.853213721, -0.0750807299, -0.539225922,
    -0.206490746, 0.393512301, 0, 0, 0], [1.18596636, -1.14885132, -2.93813019, -1.12541015, 1.18570988, 1.56757808,
    -0.164567227, -0.438824703, 1.58792603, -2.80558237, -0.746440403, 1.04348357, -3.82953829, 0.193539944,
    -1.1191716, -1.54172458, -0.538605842, 0.826425858, 1.41571534, -1.28260756, 2.13887279, 2.53897754, 0.831484114,
    1.8849918, 1.96452118, 1.58005354, 1.00083089, -2.65322095, 0.235172616, -1.39920857, -1.55284835, 0.121980263, 0,
    0, 0], [0.655091158, -2.80775084, -1.03658657, 0.606410188, 5.48908889, 2.23022933, 3.28747289, 0.602341207,
    2.43213406, 1.70548566, -3.54293244, 4.63152872, -0.116990306, -0.298271665, 0.464316172, 3.69806548, 5.27257076,
    -5.45236462, -1.41209722, 1.47940022, 2.04674065, -2.66176786, -0.969392943, -2.39505644, 0.289622641, -1.44648129,
    1.12967597, 0.557535817, 0.845424003, 4.02460452, -3.13615246, -3.16958253, 0, 0, 0], [53.5023069, -1.64095829,
    -39.8318441, 35.0645392, 59.1149589, -33.9914747, 9.19194109, -30.8116991, 38.2004231, 80.999977, 23.6656119,
    27.1725655, 1.48930447, 1.69282625, 0.546193827, 9.58624631, -24.1863511, 69.6274012, -50.6945642, -14.3577712,
    30.5340351, -24.0285057, 3.23633077, -9.52417481, -23.7250013, 36.3108895, -40.3764812, 46.93171, 9.14887174,
    -14.265871, -57.5786911, -28.7340209, 0, 0, 0], [26.656975, -18.7718999, -18.7269216, 17.6452221, -6.7395171,
    -12.9940138, -23.158598, -3.78314158, 3.95140397, -13.5492757, -33.5325309, -11.0666001, 26.0380127, 10.806756,
    10.7012893, -24.2234299, -19.5596668, 35.1280248, 4.88281529, 45.6103469, -1.19563131, 30.799245, 5.89131906,
    -13.6822583, -6.14697496, -7.87844152, -17.95419, 32.4162388, -9.95510432, -20.5703053, -11.1808617, -10.2761287,
    0, 0, 0], [-121.843182, 397.232197, 61.3160111, -266.639925, -301.295081, -218.634846, -118.753815, -526.296967,
    114.591279, 512.617129, -352.866096, -56.9760613, -703.039138, -158.23975, -458.371985, -338.945864, 175.790753,
    -4.72667984, 401.531919, 64.360756, 584.598892, 174.32841, -228.124844, 447.26224, -305.410501, -36.996053,
    162.685215, 52.726945, -231.710091, -523.10488, 63.4493122, -303.995124, 0, 0, 0], [-1220.97201, -176.701444,
    -112.139176, 2377.90652, -768.773091, -531.432997, -1117.07335, -134.79941, -260.390021, 1175.96982, -585.273742,
    -340.207519, -1071.82447, -508.121908, 1101.26041, 1075.88662, -737.986594, -1409.4861, -883.251193, -106.322865,
    150.414423, -863.028683, -724.927714, 670.19748, 327.493899, -626.64328, -358.543787, 628.180697, -1963.33144,
    -351.363746, -443.753458, -1006.85899, 0, 0, 0], [-7444.36802, 9965.09588, -6595.40194, -20780.3152, -85.7492371,
    12877.5641, -8007.8167, 2704.99056, -4919.58993, -11356.06, -9142.6624, 7733.0659, 2351.72687, -2162.53295,
    4626.5602, -25188.2471, 1343.64595, -6465.49698, -2827.07353, -7852.75293, -3377.29757, 12509.5294, -4474.95607,
    -4683.16543, 7290.56541, -17627.843, 14037.7802, -8534.91393, -21478.3515, -3942.6825, -932.709584, 2544.583, 0, 0,
    0], [3011.7552, -16631.0725, -25772.8367, -5175.3529, 6374.13781, -11584.303, 11946.7893, -26493.4402, -3293.67965,
    6098.77136, 4920.50024, -9730.75683, 20812.2747, 5981.50384, 15971.3678, -6077.16295, 382.858485, -4559.46101,
    16722.1739, -12228.1765, -14046.748, 14971.3167, 4507.648, -14125.528, -8392.65982, -8856.90277, 6186.16361,
    -1522.21541, 10723.5967, 13228.2314, -13269.8699, 17085.9004, 0, 0, 0], [-37990.0488, -32889.0011, 4242.24985,
    -22165.928, -240.943029, -15527.0755, 8454.01634, 38446.5339, -35844.5538, 37297.3198, 21423.8734, -9222.02182,
    -6344.19586, -13166.931, -11837.4568, -44325.3927, -37178.5245, 3126.73384, -983.692839, -5162.83618, 24681.4379,
    26081.3216, -17354.9285, -18174.9489, 19917.6294, 14944.0106, -15864.647, 15143.3554, -24551.3811, -24844.6795,
    -33427.8746, -7066.85003, 0, 0, 0], [-0.263017178, -0.591676463, -0.710250933, -0.76287053, -0.59517508,
    0.471453567, 0.81946323, 0.383363418, 0.429964592, -0.651122858, 0.175990118, 0.0904491239, 1.02955421,
    -1.13006915, 0.0303955896, 0.491543816, -0.63376044, 0.233695786, -0.86539338, 1.81639158, -1.80717562,
    0.768226777, -0.041447703, 0.790833912, -0.783618882, 0.345516431, -0.0117506486, 1.31455113, 0.227813617,
    -2.46213748, 0.201390998, 1.10738516, 0, 0, 0], [-0.101092996, 0.123823489, 0.173368791, -0.0801089456,
    -0.122156098, 0.318687185, 0.0993276522, -0.105829661, 0.106504378, 0.0105845634, -0.0374855727, 0.034185223,
    0.0246337014, -0.00397947936, 0.0295326695, 0.0070466438, 0.0922986886, 0.00112533447, 0.0490717979, 0.169772787,
    -0.19669593, 0.073791758, 0.0128464263, 0.00661827128, 0.0910929344, 0.0497318092, 0.194199718, 0.198862109,
    -0.0176864267, 0.156513782, -0.0694447573, -0.0344094037, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-2.99175557, -1.06350794, -1.84887209, -2.73404305,
    -2.50483785, -0.220671112, -1.42667748, 1.67834877, -0.769095476, -9.1077464, -3.35897611, -3.04409273,
    -3.22143493, 0.659374821, -5.62250792, -7.08079358, -8.46853758, -8.01060159, 5.06235183, 1.26035129, -0.508855485,
    -2.40342485, 1.45317518, 0.897358952, 4.37176666, -12.11176, 5.33932054, -2.69616541, -0.290332392, -0.522003825,
    10.5707588, 2.89376141, 0, 0, 1], [1.03294365, 1.83772777, 2.43754687, 0.671934095, 0.836093004, 0.0854877133,
    0.329937382, 0.615333754, 0.0107688406, 1.93656271, -1.00270552, 1.57359183, -1.43215538, -0.228472212,
    -1.30457282, 1.0848645, 2.40233033, 1.03869176, -0.292279917, -0.03249484, 0.506833575, -1.83298633, -0.355351552,
    0.0481526804, -0.617615169, 1.07795533, -0.951325502, 1.0933878, 0.363445056, -0.172472599, -1.63316578,
    1.30488932, 0, 0, 0], [0.159559689, 0.440956626, -0.0470462231, -0.0516244632, -0.406326164, -1.28120114,
    0.040969129, -0.116904373, 0.133726591, 0.880174942, -0.3214183, -0.228342913, 0.791279982, 0.197727426,
    0.175162931, 0.0738881657, 0.358770094, -0.139604959, -0.624187974, 0.322061717, -0.00905075721, 0.205948604,
    0.372406514, -0.861759265, 0.0428978892, 0.525755386, -0.0704746067, 0.882928096, -0.708214025, 0.171936382,
    -0.281769429, 0.203630066, 0, 0, 0], [0.375614712, -0.79825469, -0.741061752, 0.845609829, -0.142975221,
    0.0337731986, 0.0467625249, 0.463639674, -0.329561469, -0.520244672, 0.530084192, 0.105741374, 0.493673839,
    -0.256779844, 0.673691478, -0.00901293816, -0.0823291981, 0.368096046, -0.511503547, -0.432175448, -0.690214491,
    0.758852018, 0.077135909, -0.149149904, -0.261414383, 0.666861335, 0.0820755243, -0.417678091, 0.272954551,
    0.0373583384, -0.744237143, -0.290546783, 0, 0, 0], [0.110953778, -0.00527695174, -0.0636941352, -0.138178204,
    -0.232903361, 0.113465688, -0.0525944875, 0.092974407, 0.285171515, -0.269197964, -0.0281207094, -0.640327885,
    0.156286234, 0.0217830416, 0.248923197, 0.128431675, -0.0786062476, 0.124970459, 0.0227206285, 0.141327689,
    0.243220101, -0.218857363, -0.378291182, 0.245071993, 0.630384381, -0.206567569, 0.0427329402, -0.00735029257,
    -0.379894399, -0.10591022, -0.242258096, -0.178776506, 0, 0, 0], [-0.594245892, -0.435637439, -0.448163604,
    -0.284506458, -0.775035515, 0.464343155, 0.185332482, -0.394215036, -0.149799178, -0.0831254976, -0.118005887,
    -0.486441518, 0.238058743, -0.197428064, 0.237983431, 0.03908313, 0.280143784, -0.762434616, 0.428527924,
    0.5666764, 0.0467444374, 0.789966452, -0.133690651, 0.20675046, -0.317511421, 0.470962115, 0.117896301,
    0.0475324792, 0.439339939, -0.267871231, 0.0105916449, -0.2178936, 0, 0, 0], [-1.68635975, -1.63811624,
    -1.76330458, -1.66378493, 2.45023339, 0.590453931, -1.23355206, -1.02041622, -0.0359849349, -3.53601557,
    2.02645904, 0.478327624, -0.888631823, 1.10198433, -0.549283071, -2.80202387, -5.96215506, -0.621973219,
    1.50219722, -1.83879513, -0.485030268, 0.0230441469, 1.34756376, 0.570045617, 0.733644445, -5.33432774, 1.39756145,
    -3.23782217, -0.0347077864, 1.04770011, 6.01334406, -1.2459349, 0, 0, 0], [-0.0243010996, 0.0217462342,
    0.0200999424, -0.00713093661, -0.0304961831, -0.0185541651, 0.0153777128, -0.0416235416, 0.0280875023,
    0.0146188233, -0.0379597105, -0.0646086507, 0.00122925746, -0.0035729101, 0.0330687176, -0.073108378,
    -0.0178467325, -0.00921186512, -0.0566713831, 0.0114687033, 0.0280866996, 0.0214321593, -0.0200301029,
    -0.0395469607, 0.059598252, 0.0564504131, 0.00243485684, 0.0220352958, 0.0263353867, -0.00662857667, 0.00585529621,
    0.00756301354, 0, 0, 0], [-0.969146673, -0.911489065, 0.292792569, -1.62352722, 1.40842426, 2.32576327, -1.4869628,
    -1.66106071, -0.304733849, -0.941140967, 0.537216574, -0.2459091, 0.617710657, -0.36104045, 2.45779119,
    -1.20932396, -1.99204906, 1.30710204, 0.403154191, 0.927651828, -0.994630734, 2.80134407, 0.0403307285,
    -0.299351861, -2.36663286, 0.0920285238, -0.398350962, -1.29822493, -1.68923388, 1.03916473, -0.0305037534,
    -0.647566912, 0, 0, 0], [-56.5162928, -59.8583394, -12.793314, 68.5909488, -28.3954082, 59.4305729, 144.195524,
    -93.0357862, 29.243408, 36.7454452, 17.942581, 80.5993699, 9.71865354, -35.676093, -11.5665081, 48.7567412,
    19.8202357, -68.8809585, 46.479747, 134.036907, -42.8916035, 0.352959518, -12.2013771, 97.8528986, 153.11599,
    27.2469792, 51.4413457, -151.980811, 59.6854672, -134.97435, -64.335066, 104.021066, 0, 0, 0], [-561.114733,
    -311.11198, -259.257249, -159.496965, 215.499645, -903.576129, -211.874979, -681.344159, -719.140216, -540.017022,
    126.98767, 1417.52363, 893.590177, -73.9970402, 107.384776, 394.131284, -460.167544, -417.918083, 166.466481,
    -150.052887, -144.079125, 733.76128, -525.659115, 87.8765677, 475.251311, 264.829064, 899.204276, -153.027578,
    692.963095, -18.5376433, 824.841519, -398.068435, 0, 0, 0]], [[0.0121698474, 0.0480854299, 0.00812933353,
    -0.0126857349, -0.00793664811, 0.0575442617, 0.0893989663, -0.005464908, 0.0124702079, -0.0811655324, 0.0254390072,
    0.0613470236, -0.0840409351, -0.0122381987, -0.00547355718, -0.0152790681, -0.0291967699, -0.0300083087,
    0.0277645446, 0.105752801, -0.00943615054, -0.0117494846, -0.0082033514, 0.0466504246, 0.0608832031, 0.0139216943,
    0.0491026377, -0.0687410946, -0.0432446185, 0.0178622326, -0.0628037523, -0.0583070886, 0, 0, 0], [-2.52402159,
    1.83892744, 2.66762092, -0.510325866, -0.223316421, -1.35527324, 0.845233247, -0.249054807, -2.28590209,
    6.96418165, 6.07701097, 1.18815813, 4.93789867, 3.59777272, 5.45303674, 8.42689744, 9.89084834, 5.16469545,
    -2.27999674, -5.4078195, 1.91225383, 1.5138683, -0.255444743, -0.00614493243, 0.931412372, 4.18421106, -5.1689647,
    -2.50948022, 0.397510684, 2.06279938, 2.53763887, -5.2451757, 0, 0, 0], [0.0315222315, -0.103147981, 0.62173113,
    0.279799203, 0.468465454, -1.07305389, 0.455366578, 0.159708343, 0.488398777, -0.452518913, 0.464655169,
    0.162000685, -0.253787946, -0.0930729909, 0.0325427201, 0.51420894, 0.231743299, -0.627926207, 0.305095162,
    0.467596274, -0.708040549, 0.0685000038, 0.0562234241, 0.60154213, -0.70632214, 0.516864429, 0.0360230365,
    0.853213721, -0.0750807299, -0.539225922, -0.206490746, 0.393512301, 0, 0, 0], [1.18596636, -1.14885132,
    -2.93813019, -1.12541015, 1.18570988, 1.56757808, -0.164567227, -0.438824703, 1.58792603, -2.80558237,
    -0.746440403, 1.04348357, -3.82953829, 0.193539944, -1.1191716, -1.54172458, -0.538605842, 0.826425858, 1.41571534,
    -1.28260756, 2.13887279, 2.53897754, 0.831484114, 1.8849918, 1.96452118, 1.58005354, 1.00083089, -2.65322095,
    0.235172616, -1.39920857, -1.55284835, 0.121980263, 0, 0, 0], [0.655091158, -2.80775084, -1.03658657, 0.606410188,
    5.48908889, 2.23022933, 3.28747289, 0.602341207, 2.43213406, 1.70548566, -3.54293244, 4.63152872, -0.116990306,
    -0.298271665, 0.464316172, 3.69806548, 5.27257076, -5.45236462, -1.41209722, 1.47940022, 2.04674065, -2.66176786,
    -0.969392943, -2.39505644, 0.289622641, -1.44648129, 1.12967597, 0.557535817, 0.845424003, 4.02460452, -3.13615246,
    -3.16958253, 0, 0, 0], [53.5023069, -1.64095829, -39.8318441, 35.0645392, 59.1149589, -33.9914747, 9.19194109,
    -30.8116991, 38.2004231, 80.999977, 23.6656119, 27.1725655, 1.48930447, 1.69282625, 0.546193827, 9.58624631,
    -24.1863511, 69.6274012, -50.6945642, -14.3577712, 30.5340351, -24.0285057, 3.23633077, -9.52417481, -23.7250013,
    36.3108895, -40.3764812, 46.93171, 9.14887174, -14.265871, -57.5786911, -28.7340209, 0, 0, 0], [26.656975,
    -18.7718999, -18.7269216, 17.6452221, -6.7395171, -12.9940138, -23.158598, -3.78314158, 3.95140397, -13.5492757,
    -33.5325309, -11.0666001, 26.0380127, 10.806756, 10.7012893, -24.2234299, -19.5596668, 35.1280248, 4.88281529,
    45.6103469, -1.19563131, 30.799245, 5.89131906, -13.6822583, -6.14697496, -7.87844152, -17.95419, 32.4162388,
    -9.95510432, -20.5703053, -11.1808617, -10.2761287, 0, 0, 0], [-121.843182, 397.232197, 61.3160111, -266.639925,
    -301.295081, -218.634846, -118.753815, -526.296967, 114.591279, 512.617129, -352.866096, -56.9760613, -703.039138,
    -158.23975, -458.371985, -338.945864, 175.790753, -4.72667984, 401.531919, 64.360756, 584.598892, 174.32841,
    -228.124844, 447.26224, -305.410501, -36.996053, 162.685215, 52.726945, -231.710091, -523.10488, 63.4493122,
    -303.995124, 0, 0, 0], [-1220.97201, -176.701444, -112.139176, 2377.90652, -768.773091, -531.432997, -1117.07335,
    -134.79941, -260.390021, 1175.96982, -585.273742, -340.207519, -1071.82447, -508.121908, 1101.26041, 1075.88662,
    -737.986594, -1409.4861, -883.251193, -106.322865, 150.414423, -863.028683, -724.927714, 670.19748, 327.493899,
    -626.64328, -358.543787, 628.180697, -1963.33144, -351.363746, -443.753458, -1006.85899, 0, 0, 0], [-7444.36802,
    9965.09588, -6595.40194, -20780.3152, -85.7492371, 12877.5641, -8007.8167, 2704.99056, -4919.58993, -11356.06,
    -9142.6624, 7733.0659, 2351.72687, -2162.53295, 4626.5602, -25188.2471, 1343.64595, -6465.49698, -2827.07353,
    -7852.75293, -3377.29757, 12509.5294, -4474.95607, -4683.16543, 7290.56541, -17627.843, 14037.7802, -8534.91393,
    -21478.3515, -3942.6825, -932.709584, 2544.583, 0, 0, 0], [3011.7552, -16631.0725, -25772.8367, -5175.3529,
    6374.13781, -11584.303, 11946.7893, -26493.4402, -3293.67965, 6098.77136, 4920.50024, -9730.75683, 20812.2747,
    5981.50384, 15971.3678, -6077.16295, 382.858485, -4559.46101, 16722.1739, -12228.1765, -14046.748, 14971.3167,
    4507.648, -14125.528, -8392.65982, -8856.90277, 6186.16361, -1522.21541, 10723.5967, 13228.2314, -13269.8699,
    17085.9004, 0, 0, 0], [-37990.0488, -32889.0011, 4242.24985, -22165.928, -240.943029, -15527.0755, 8454.01634,
    38446.5339, -35844.5538, 37297.3198, 21423.8734, -9222.02182, -6344.19586, -13166.931, -11837.4568, -44325.3927,
    -37178.5245, 3126.73384, -983.692839, -5162.83618, 24681.4379, 26081.3216, -17354.9285, -18174.9489, 19917.6294,
    14944.0106, -15864.647, 15143.3554, -24551.3811, -24844.6795, -33427.8746, -7066.85003, 0, 0, 0], [-0.263017178,
    -0.591676463, -0.710250933, -0.76287053, -0.59517508, 0.471453567, 0.81946323, 0.383363418, 0.429964592,
    -0.651122858, 0.175990118, 0.0904491239, 1.02955421, -1.13006915, 0.0303955896, 0.491543816, -0.63376044,
    0.233695786, -0.86539338, 1.81639158, -1.80717562, 0.768226777, -0.041447703, 0.790833912, -0.783618882,
    0.345516431, -0.0117506486, 1.31455113, 0.227813617, -2.46213748, 0.201390998, 1.10738516, 0, 0, 0], [-0.101092996,
    0.123823489, 0.173368791, -0.0801089456, -0.122156098, 0.318687185, 0.0993276522, -0.105829661, 0.106504378,
    0.0105845634, -0.0374855727, 0.034185223, 0.0246337014, -0.00397947936, 0.0295326695, 0.0070466438, 0.0922986886,
    0.00112533447, 0.0490717979, 0.169772787, -0.19669593, 0.073791758, 0.0128464263, 0.00661827128, 0.0910929344,
    0.0497318092, 0.194199718, 0.198862109, -0.0176864267, 0.156513782, -0.0694447573, -0.0344094037, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-3.17705494,
    -0.951722777, -1.53740645, -2.83214987, -2.57472911, -0.0414215023, -1.53438528, 1.65567993, -0.549156455,
    -9.19561311, -3.36611747, -2.7444359, -3.58866877, 0.569065758, -5.7498931, -7.13271841, -8.42658366, -8.00499548,
    5.24499358, 1.16423298, -0.542296834, -2.52291043, 1.63806628, 0.847200689, 4.04645046, -12.0125931, 5.45714208,
    -3.11612094, -0.318392526, -0.400353171, 10.597205, 2.79515394, 0, 0, 1], [1.03294365, 1.83772777, 2.43754687,
    0.671934095, 0.836093004, 0.0854877133, 0.329937382, 0.615333754, 0.0107688406, 1.93656271, -1.00270552,
    1.57359183, -1.43215538, -0.228472212, -1.30457282, 1.0848645, 2.40233033, 1.03869176, -0.292279917, -0.03249484,
    0.506833575, -1.83298633, -0.355351552, 0.0481526804, -0.617615169, 1.07795533, -0.951325502, 1.0933878,
    0.363445056, -0.172472599, -1.63316578, 1.30488932, 0, 0, 0], [0.159559689, 0.440956626, -0.0470462231,
    -0.0516244632, -0.406326164, -1.28120114, 0.040969129, -0.116904373, 0.133726591, 0.880174942, -0.3214183,
    -0.228342913, 0.791279982, 0.197727426, 0.175162931, 0.0738881657, 0.358770094, -0.139604959, -0.624187974,
    0.322061717, -0.00905075721, 0.205948604, 0.372406514, -0.861759265, 0.0428978892, 0.525755386, -0.0704746067,
    0.882928096, -0.708214025, 0.171936382, -0.281769429, 0.203630066, 0, 0, 0], [0.375614712, -0.79825469,
    -0.741061752, 0.845609829, -0.142975221, 0.0337731986, 0.0467625249, 0.463639674, -0.329561469, -0.520244672,
    0.530084192, 0.105741374, 0.493673839, -0.256779844, 0.673691478, -0.00901293816, -0.0823291981, 0.368096046,
    -0.511503547, -0.432175448, -0.690214491, 0.758852018, 0.077135909, -0.149149904, -0.261414383, 0.666861335,
    0.0820755243, -0.417678091, 0.272954551, 0.0373583384, -0.744237143, -0.290546783, 0, 0, 0], [0.110953778,
    -0.00527695174, -0.0636941352, -0.138178204, -0.232903361, 0.113465688, -0.0525944875, 0.092974407, 0.285171515,
    -0.269197964, -0.0281207094, -0.640327885, 0.156286234, 0.0217830416, 0.248923197, 0.128431675, -0.0786062476,
    0.124970459, 0.0227206285, 0.141327689, 0.243220101, -0.218857363, -0.378291182, 0.245071993, 0.630384381,
    -0.206567569, 0.0427329402, -0.00735029257, -0.379894399, -0.10591022, -0.242258096, -0.178776506, 0, 0, 0],
    [-0.594245892, -0.435637439, -0.448163604, -0.284506458, -0.775035515, 0.464343155, 0.185332482, -0.394215036,
    -0.149799178, -0.0831254976, -0.118005887, -0.486441518, 0.238058743, -0.197428064, 0.237983431, 0.03908313,
    0.280143784, -0.762434616, 0.428527924, 0.5666764, 0.0467444374, 0.789966452, -0.133690651, 0.20675046,
    -0.317511421, 0.470962115, 0.117896301, 0.0475324792, 0.439339939, -0.267871231, 0.0105916449, -0.2178936, 0, 0,
    0], [-1.68635975, -1.63811624, -1.76330458, -1.66378493, 2.45023339, 0.590453931, -1.23355206, -1.02041622,
    -0.0359849349, -3.53601557, 2.02645904, 0.478327624, -0.888631823, 1.10198433, -0.549283071, -2.80202387,
    -5.96215506, -0.621973219, 1.50219722, -1.83879513, -0.485030268, 0.0230441469, 1.34756376, 0.570045617,
    0.733644445, -5.33432774, 1.39756145, -3.23782217, -0.0347077864, 1.04770011, 6.01334406, -1.2459349, 0, 0, 0],
    [-0.0243010996, 0.0217462342, 0.0200999424, -0.00713093661, -0.0304961831, -0.0185541651, 0.0153777128,
    -0.0416235416, 0.0280875023, 0.0146188233, -0.0379597105, -0.0646086507, 0.00122925746, -0.0035729101,
    0.0330687176, -0.073108378, -0.0178467325, -0.00921186512, -0.0566713831, 0.0114687033, 0.0280866996, 0.0214321593,
    -0.0200301029, -0.0395469607, 0.059598252, 0.0564504131, 0.00243485684, 0.0220352958, 0.0263353867, -0.00662857667,
    0.00585529621, 0.00756301354, 0, 0, 0], [-0.969146673, -0.911489065, 0.292792569, -1.62352722, 1.40842426,
    2.32576327, -1.4869628, -1.66106071, -0.304733849, -0.941140967, 0.537216574, -0.2459091, 0.617710657, -0.36104045,
    2.45779119, -1.20932396, -1.99204906, 1.30710204, 0.403154191, 0.927651828, -0.994630734, 2.80134407, 0.0403307285,
    -0.299351861, -2.36663286, 0.0920285238, -0.398350962, -1.29822493, -1.68923388, 1.03916473, -0.0305037534,
    -0.647566912, 0, 0, 0], [-56.5162928, -59.8583394, -12.793314, 68.5909488, -28.3954082, 59.4305729, 144.195524,
    -93.0357862, 29.243408, 36.7454452, 17.942581, 80.5993699, 9.71865354, -35.676093, -11.5665081, 48.7567412,
    19.8202357, -68.8809585, 46.479747, 134.036907, -42.8916035, 0.352959518, -12.2013771, 97.8528986, 153.11599,
    27.2469792, 51.4413457, -151.980811, 59.6854672, -134.97435, -64.335066, 104.021066, 0, 0, 0], [-561.114733,
    -311.11198, -259.257249, -159.496965, 215.499645, -903.576129, -211.874979, -681.344159, -719.140216, -540.017022,
    126.98767, 1417.52363, 893.590177, -73.9970402, 107.384776, 394.131284, -460.167544, -417.918083, 166.466481,
    -150.052887, -144.079125, 733.76128, -525.659115, 87.8765677, 475.251311, 264.829064, 899.204276, -153.027578,
    692.963095, -18.5376433, 824.841519, -398.068435, 0, 0, 0]], [[0.0121698474, 0.0480854299, 0.00812933353,
    -0.0126857349, -0.00793664811, 0.0575442617, 0.0893989663, -0.005464908, 0.0124702079, -0.0811655324, 0.0254390072,
    0.0613470236, -0.0840409351, -0.0122381987, -0.00547355718, -0.0152790681, -0.0291967699, -0.0300083087,
    0.0277645446, 0.105752801, -0.00943615054, -0.0117494846, -0.0082033514, 0.0466504246, 0.0608832031, 0.0139216943,
    0.0491026377, -0.0687410946, -0.0432446185, 0.0178622326, -0.0628037523, -0.0583070886, 0, 0, 0], [-2.52402159,
    1.83892744, 2.66762092, -0.510325866, -0.223316421, -1.35527324, 0.845233247, -0.249054807, -2.28590209,
    6.96418165, 6.07701097, 1.18815813, 4.93789867, 3.59777272, 5.45303674, 8.42689744, 9.89084834, 5.16469545,
    -2.27999674, -5.4078195, 1.91225383, 1.5138683, -0.255444743, -0.00614493243, 0.931412372, 4.18421106, -5.1689647,
    -2.50948022, 0.397510684, 2.06279938, 2.53763887, -5.2451757, 0, 0, 0], [0.0315222315, -0.103147981, 0.62173113,
    0.279799203, 0.468465454, -1.07305389, 0.455366578, 0.159708343, 0.488398777, -0.452518913, 0.464655169,
    0.162000685, -0.253787946, -0.0930729909, 0.0325427201, 0.51420894, 0.231743299, -0.627926207, 0.305095162,
    0.467596274, -0.708040549, 0.0685000038, 0.0562234241, 0.60154213, -0.70632214, 0.516864429, 0.0360230365,
    0.853213721, -0.0750807299, -0.539225922, -0.206490746, 0.393512301, 0, 0, 0], [1.18596636, -1.14885132,
    -2.93813019, -1.12541015, 1.18570988, 1.56757808, -0.164567227, -0.438824703, 1.58792603, -2.80558237,
    -0.746440403, 1.04348357, -3.82953829, 0.193539944, -1.1191716, -1.54172458, -0.538605842, 0.826425858, 1.41571534,
    -1.28260756, 2.13887279, 2.53897754, 0.831484114, 1.8849918, 1.96452118, 1.58005354, 1.00083089, -2.65322095,
    0.235172616, -1.39920857, -1.55284835, 0.121980263, 0, 0, 0], [0.655091158, -2.80775084, -1.03658657, 0.606410188,
    5.48908889, 2.23022933, 3.28747289, 0.602341207, 2.43213406, 1.70548566, -3.54293244, 4.63152872, -0.116990306,
    -0.298271665, 0.464316172, 3.69806548, 5.27257076, -5.45236462, -1.41209722, 1.47940022, 2.04674065, -2.66176786,
    -0.969392943, -2.39505644, 0.289622641, -1.44648129, 1.12967597, 0.557535817, 0.845424003, 4.02460452, -3.13615246,
    -3.16958253, 0, 0, 0], [53.5023069, -1.64095829, -39.8318441, 35.0645392, 59.1149589, -33.9914747, 9.19194109,
    -30.8116991, 38.2004231, 80.999977, 23.6656119, 27.1725655, 1.48930447, 1.69282625, 0.546193827, 9.58624631,
    -24.1863511, 69.6274012, -50.6945642, -14.3577712, 30.5340351, -24.0285057, 3.23633077, -9.52417481, -23.7250013,
    36.3108895, -40.3764812, 46.93171, 9.14887174, -14.265871, -57.5786911, -28.7340209, 0, 0, 0], [26.656975,
    -18.7718999, -18.7269216, 17.6452221, -6.7395171, -12.9940138, -23.158598, -3.78314158, 3.95140397, -13.5492757,
    -33.5325309, -11.0666001, 26.0380127, 10.806756, 10.7012893, -24.2234299, -19.5596668, 35.1280248, 4.88281529,
    45.6103469, -1.19563131, 30.799245, 5.89131906, -13.6822583, -6.14697496, -7.87844152, -17.95419, 32.4162388,
    -9.95510432, -20.5703053, -11.1808617, -10.2761287, 0, 0, 0], [-121.843182, 397.232197, 61.3160111, -266.639925,
    -301.295081, -218.634846, -118.753815, -526.296967, 114.591279, 512.617129, -352.866096, -56.9760613, -703.039138,
    -158.23975, -458.371985, -338.945864, 175.790753, -4.72667984, 401.531919, 64.360756, 584.598892, 174.32841,
    -228.124844, 447.26224, -305.410501, -36.996053, 162.685215, 52.726945, -231.710091, -523.10488, 63.4493122,
    -303.995124, 0, 0, 0], [-1220.97201, -176.701444, -112.139176, 2377.90652, -768.773091, -531.432997, -1117.07335,
    -134.79941, -260.390021, 1175.96982, -585.273742, -340.207519, -1071.82447, -508.121908, 1101.26041, 1075.88662,
    -737.986594, -1409.4861, -883.251193, -106.322865, 150.414423, -863.028683, -724.927714, 670.19748, 327.493899,
    -626.64328, -358.543787, 628.180697, -1963.33144, -351.363746, -443.753458, -1006.85899, 0, 0, 0], [-7444.36802,
    9965.09588, -6595.40194, -20780.3152, -85.7492371, 12877.5641, -8007.8167, 2704.99056, -4919.58993, -11356.06,
    -9142.6624, 7733.0659, 2351.72687, -2162.53295, 4626.5602, -25188.2471, 1343.64595, -6465.49698, -2827.07353,
    -7852.75293, -3377.29757, 12509.5294, -4474.95607, -4683.16543, 7290.56541, -17627.843, 14037.7802, -8534.91393,
    -21478.3515, -3942.6825, -932.709584, 2544.583, 0, 0, 0], [3011.7552, -16631.0725, -25772.8367, -5175.3529,
    6374.13781, -11584.303, 11946.7893, -26493.4402, -3293.67965, 6098.77136, 4920.50024, -9730.75683, 20812.2747,
    5981.50384, 15971.3678, -6077.16295, 382.858485, -4559.46101, 16722.1739, -12228.1765, -14046.748, 14971.3167,
    4507.648, -14125.528, -8392.65982, -8856.90277, 6186.16361, -1522.21541, 10723.5967, 13228.2314, -13269.8699,
    17085.9004, 0, 0, 0], [-37990.0488, -32889.0011, 4242.24985, -22165.928, -240.943029, -15527.0755, 8454.01634,
    38446.5339, -35844.5538, 37297.3198, 21423.8734, -9222.02182, -6344.19586, -13166.931, -11837.4568, -44325.3927,
    -37178.5245, 3126.73384, -983.692839, -5162.83618, 24681.4379, 26081.3216, -17354.9285, -18174.9489, 19917.6294,
    14944.0106, -15864.647, 15143.3554, -24551.3811, -24844.6795, -33427.8746, -7066.85003, 0, 0, 0], [-0.263017178,
    -0.591676463, -0.710250933, -0.76287053, -0.59517508, 0.471453567, 0.81946323, 0.383363418, 0.429964592,
    -0.651122858, 0.175990118, 0.0904491239, 1.02955421, -1.13006915, 0.0303955896, 0.491543816, -0.63376044,
    0.233695786, -0.86539338, 1.81639158, -1.80717562, 0.768226777, -0.041447703, 0.790833912, -0.783618882,
    0.345516431, -0.0117506486, 1.31455113, 0.227813617, -2.46213748, 0.201390998, 1.10738516, 0, 0, 0], [-0.101092996,
    0.123823489, 0.173368791, -0.0801089456, -0.122156098, 0.318687185, 0.0993276522, -0.105829661, 0.106504378,
    0.0105845634, -0.0374855727, 0.034185223, 0.0246337014, -0.00397947936, 0.0295326695, 0.0070466438, 0.0922986886,
    0.00112533447, 0.0490717979, 0.169772787, -0.19669593, 0.073791758, 0.0128464263, 0.00661827128, 0.0910929344,
    0.0497318092, 0.194199718, 0.198862109, -0.0176864267, 0.156513782, -0.0694447573, -0.0344094037, 0, 0, 0], [0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [-3.36235431,
    -0.839937613, -1.22594081, -2.93025669, -2.64462037, 0.137828107, -1.64209307, 1.63301108, -0.329217435,
    -9.28347982, -3.37325882, -2.44477907, -3.95590261, 0.478756696, -5.87727829, -7.18464325, -8.38462975,
    -7.99938937, 5.42763533, 1.06811468, -0.575738183, -2.64239601, 1.82295738, 0.797042425, 3.72113427, -11.9134261,
    5.57496361, -3.53607647, -0.34645266, -0.278702518, 10.6236513, 2.69654648, 0, 0, 1], [1.03294365, 1.83772777,
    2.43754687, 0.671934095, 0.836093004, 0.0854877133, 0.329937382, 0.615333754, 0.0107688406, 1.93656271,
    -1.00270552, 1.57359183, -1.43215538, -0.228472212, -1.30457282, 1.0848645, 2.40233033, 1.03869176, -0.292279917,
    -0.03249484, 0.506833575, -1.83298633, -0.355351552, 0.0481526804, -0.617615169, 1.07795533, -0.951325502,
    1.0933878, 0.363445056, -0.172472599, -1.63316578, 1.30488932, 0, 0, 0], [0.159559689, 0.440956626, -0.0470462231,
    -0.0516244632, -0.406326164, -1.28120114, 0.040969129, -0.116904373, 0.133726591, 0.880174942, -0.3214183,
    -0.228342913, 0.791279982, 0.197727426, 0.175162931, 0.0738881657, 0.358770094, -0.139604959, -0.624187974,
    0.322061717, -0.00905075721, 0.205948604, 0.372406514, -0.861759265, 0.0428978892, 0.525755386, -0.0704746067,
    0.882928096, -0.708214025, 0.171936382, -0.281769429, 0.203630066, 0, 0, 0], [0.375614712, -0.79825469,
    -0.741061752, 0.845609829, -0.142975221, 0.0337731986, 0.0467625249, 0.463639674, -0.329561469, -0.520244672,
    0.530084192, 0.105741374, 0.493673839, -0.256779844, 0.673691478, -0.00901293816, -0.0823291981, 0.368096046,
    -0.511503547, -0.432175448, -0.690214491, 0.758852018, 0.077135909, -0.149149904, -0.261414383, 0.666861335,
    0.0820755243, -0.417678091, 0.272954551, 0.0373583384, -0.744237143, -0.290546783, 0, 0, 0], [0.110953778,
    -0.00527695174, -0.0636941352, -0.138178204, -0.232903361, 0.113465688, -0.0525944875, 0.092974407, 0.285171515,
    -0.269197964, -0.0281207094, -0.640327885, 0.156286234, 0.0217830416, 0.248923197, 0.128431675, -0.0786062476,
    0.124970459, 0.0227206285, 0.141327689, 0.243220101, -0.218857363, -0.378291182, 0.245071993, 0.630384381,
    -0.206567569, 0.0427329402, -0.00735029257, -0.379894399, -0.10591022, -0.242258096, -0.178776506, 0, 0, 0],
    [-0.594245892, -0.435637439, -0.448163604, -0.284506458, -0.775035515, 0.464343155, 0.185332482, -0.394215036,
    -0.149799178, -0.0831254976, -0.118005887, -0.486441518, 0.238058743, -0.197428064, 0.237983431, 0.03908313,
    0.280143784, -0.762434616, 0.428527924, 0.5666764, 0.0467444374, 0.789966452, -0.133690651, 0.20675046,
    -0.317511421, 0.470962115, 0.117896301, 0.0475324792, 0.439339939, -0.267871231, 0.0105916449, -0.2178936, 0, 0,
    0], [-1.68635975, -1.63811624, -1.76330458, -1.66378493, 2.45023339, 0.590453931, -1.23355206, -1.02041622,
    -0.0359849349, -3.53601557, 2.02645904, 0.478327624, -0.888631823, 1.10198433, -0.549283071, -2.80202387,
    -5.96215506, -0.621973219, 1.50219722, -1.83879513, -0.485030268, 0.0230441469, 1.34756376, 0.570045617,
    0.733644445, -5.33432774, 1.39756145, -3.23782217, -0.0347077864, 1.04770011, 6.01334406, -1.2459349, 0, 0, 0],
    [-0.0243010996, 0.0217462342, 0.0200999424, -0.00713093661, -0.0304961831, -0.0185541651, 0.0153777128,
    -0.0416235416, 0.0280875023, 0.0146188233, -0.0379597105, -0.0646086507, 0.00122925746, -0.0035729101,
    0.0330687176, -0.073108378, -0.0178467325, -0.00921186512, -0.0566713831, 0.0114687033, 0.0280866996, 0.0214321593,
    -0.0200301029, -0.0395469607, 0.059598252, 0.0564504131, 0.00243485684, 0.0220352958, 0.0263353867, -0.00662857667,
    0.00585529621, 0.00756301354, 0, 0, 0], [-0.969146673, -0.911489065, 0.292792569, -1.62352722, 1.40842426,
    2.32576327, -1.4869628, -1.66106071, -0.304733849, -0.941140967, 0.537216574, -0.2459091, 0.617710657, -0.36104045,
    2.45779119, -1.20932396, -1.99204906, 1.30710204, 0.403154191, 0.927651828, -0.994630734, 2.80134407, 0.0403307285,
    -0.299351861, -2.36663286, 0.0920285238, -0.398350962, -1.29822493, -1.68923388, 1.03916473, -0.0305037534,
    -0.647566912, 0, 0, 0], [-56.5162928, -59.8583394, -12.793314, 68.5909488, -28.3954082, 59.4305729, 144.195524,
    -93.0357862, 29.243408, 36.7454452, 17.942581, 80.5993699, 9.71865354, -35.676093, -11.5665081, 48.7567412,
    19.8202357, -68.8809585, 46.479747, 134.036907, -42.8916035, 0.352959518, -12.2013771, 97.8528986, 153.11599,
    27.2469792, 51.4413457, -151.980811, 59.6854672, -134.97435, -64.335066, 104.021066, 0, 0, 0], [-561.114733,
    -311.11198, -259.257249, -159.496965, 215.499645, -903.576129, -211.874979, -681.344159, -719.140216, -540.017022,
    126.98767, 1417.52363, 893.590177, -73.9970402, 107.384776, 394.131284, -460.167544, -417.918083, 166.466481,
    -150.052887, -144.079125, 733.76128, -525.659115, 87.8765677, 475.251311, 264.829064, 899.204276, -153.027578,
    692.963095, -18.5376433, 824.841519, -398.068435, 0, 0, 0]]]
_LC_W2B = [1.23058566e-05, 1.38192896e-05, -1.50159491e-05, -1.27035527e-05, 5.05048028e-06, -1.0925194e-05,
    9.14723862e-06, -7.03095492e-06, 2.29448007e-06, -1.47897083e-05, 6.07678482e-06, 5.50384019e-06, 9.81985172e-06,
    2.53880754e-06, 6.30285101e-06, -8.07081148e-06, 1.21661397e-05, 7.2531246e-06, -7.90402424e-06, 4.31135784e-06,
    8.54846496e-06, -9.77130383e-06, 1.42746747e-06, 6.92939157e-06, -9.12509841e-06, 4.62479751e-06, -1.10190413e-05,
    -8.60670931e-06, 4.56431674e-06, 8.00490119e-06, -3.74258446e-06, 1.09686372e-05, 1, -1, 1.26429597e-06]


_LAW_ROWS = (  # per-neuron K4 law: [1, c1, c2, 0, 0] for the dA, lam'_t and lam'_{t+1} rows of FF
    (1.0, 0.01915420412393816, -0.012942676523887354, 0.0, 0.0),
    (1.0, 0.00778923125415545, -0.0032661093936979203, 0.0, 0.0),
    (1.0, 0.032542152563791754, -0.0032124939582045525, 0.0, 0.0),
)


# ----------------------------------------------------------------------------------------------------------------
# W-message recurrent corrector (wmsg.npz, research/gru_build/make_wmsg_npz.py)
# ----------------------------------------------------------------------------------------------------------------
_WM_KEYS = ("hT", "kT", "fT", "sel5", "pexp", "selp", "e4", "selq", "relu", "e6", "e2", "b2", "wfb", "wat", "msc",
            "wbig", "cm", "mc")


def _wmsg(weights, VX, QB, SVB, P, f32, parts=None, QX=None):
    """Add the corrector's dp_L to VX[L, pK1] in place (L = depth - 1). QB[t] (t = 0..L) holds the engine's
    [phi(a), Phi(a), 1, dA, lam'_t, lam'_{t+1}, t3, a, var] (the K4 law applied), SVB[t] its sigma^-4..sigma^4.

    Per-neuron work runs over all layers at once (three small einsums give 17 feature rows per layer; a ReLU cumulant
    table gives the previous layer's k2..k6). The neighbour sums take five products per layer (pass 1). The recurrent
    pass then takes 11 calls per layer: the messages W^T [h', dp] and (W o W)^T |dp|, one product into the gates
    (z-score, depth, biases and sigma(x) = (1 + tanh(x / 2)) / 2 folded into the weights), two tanh, the update and dp.
    Columns of MSG: [m2 raw, 1, NN (32), Hm raw (32), push, h' (32), dp]."""
    es, us, cp = fnp.einsum, fnp.unstack, fnp.copyto
    depth, n = len(weights), weights[0].shape[0]
    L = depth - 1
    with _ns("wmsg"):
        # t3 of the middle layers from their VEC rows (the last layer's is stored with its FF block)
        fnp.divide(VX[1:L, V_W11T], fnp.maximum(VX[1:L, V_W11], 1e-30), out=QB[1:L, 6])
        # ---- 17 feature rows of every layer 0..L: [a, Phi, phi, k3, k4, kA, R3n, R4n, E33n, E34n, E44n, W5n,
        #      He4 phi, p, 1, sigma, D] ----
        Ap = fnp.power(QB[:, 7:8], P["pexp"])
        D5 = es("txi,xye,tei->tyi", QB[:, 2:7], P["sel5"], SVB)  # [1, k3, kA, lam'/var, sigma]
        K = es("qxy,txi,tyi->tqi", P["kT"], D5, D5)  # [1, k3, k4, k3^2, k3 k4, k4^2, kA, sigma]
        Bas = es("hcp,tci,tpi->thi", P["hT"], QB[:, 0:3], Ap)  # phi He_0..6, Phi, a Phi, 1, a
        FB = es("rhq,thi,tqi->tri", P["fT"], Bas, K)
        # ---- vectors of layers 0..L-1 for the neighbour sums of layers 1..L:
        #      [a, R4n, k4, Phi, 1, k2 | k4c, k2^2, 1, 0, 0, 0 | k3, k5 | k6] (k: ReLU cumulants of N(m, var)) ----
        VV = fnp.zeros((L, 15, n), dtype=f32)
        cp(VV[:, 4:9:4], 1.0)
        es("tfi,fk->tki", FB[0:L], P["selp"], out=VV[:, 0:4])
        pa = QB[0:L, 7]  # float64 from here to the cumulants: the table cancels strongly at large |a|
        Q4 = fnp.stack([flops.stats.norm.cdf(pa), flops.stats.norm.pdf(pa), pa, QB[0:L, 8]], axis=1)  # Phi, phi, a, var
        P0, P1, P2, P3 = us(fnp.power(Q4[:, :, None, :], P["e4"]), axis=1)
        CUM = es("rijp,tin,tjn,tpn,trn->trn", P["relu"], P0, P1, P2, es("rq,tqn->trn", P["selq"], P3))
        cp(VV[:, 5:8], CUM[:, 0:3])
        cp(VV[:, 12:15], CUM[:, 3:6])
        # ---- pass 1: sums through W o W (6 vectors), W^4 (3, padded to 6), W^3, W^5 and W^6 ----
        WP = fnp.empty((2, n, n), dtype=f32)
        W2, W4 = us(WP)
        SS = fnp.empty((L, 15, n), dtype=f32)  # [Sxa, SxR4n, Sxk4, SxPhi, cn2, S2 | S4, Q4, cn4, 0, 0, 0 | S3, S5 | S6]
        if _R30B.get('w4p'):  # r30b w4p: the W^4 sums of the three zero padding vectors, zero once
            cp(SS[:, 9:12], 0.0)
        V24, S24 = us(VV[:, 0:12].reshape((L, 2, 6, n))), us(SS[:, 0:12].reshape((L, 2, 6, n)))
        V35, S35, V6, S6 = us(VV[:, 12:14]), us(SS[:, 12:14]), us(VV[:, 14]), us(SS[:, 14])
        for t in range(1, depth):
            W = weights[t]
            if _dtname(W.dtype) != _dtname(f32):
                W = W.astype(f32)
            fnp.multiply(W, W, out=W2)
            fnp.multiply(W2, W2, out=W4)
            if _R30B.get('w4p'):  # r30b w4p: W o W on its six vectors, W^4 on its three (not the zero padding)
                es("ia,ci->ca", W2, V24[t - 1][0], out=S24[t - 1][0])
                es("ia,ci->ca", W4, V24[t - 1][1, 0:3], out=S24[t - 1][1, 0:3])
            else:
                es("pia,pci->pca", WP, V24[t - 1], out=S24[t - 1])
            es("ia,pia,pi->pa", W, WP, V35[t - 1], out=S35[t - 1])
            es("ia,ia,i->a", W2, W4, V6[t - 1], out=S6[t - 1])
        # ---- atoms of the neighbour features of layers 1..L ----
        FAC = fnp.empty((L, 10, n), dtype=f32)  # [1, s^-2, s^-3, s^-4, s^-5, s^-6, 1/cn2, 1/cn2^2, W5n s^-5, He4phi s^-6]
        fnp.power(QB[1:, 8:9], P["e6"], out=FAC[:, 0:6])
        fnp.power(SS[:, 4:5], P["e2"], out=FAC[:, 6:8])
        fnp.multiply(FB[1:, 11:13], FAC[:, 4:6], out=FAC[:, 8:10])
        AT = fnp.empty((L, 15, n), dtype=f32)
        es("tki,krf,tfi->tri", SS, P["b2"], FAC, out=AT[:, 0:13])
        fnp.log(SS[:, 5:8:2], out=AT[:, 13:15])  # log S2, log Q4
        # ---- input side of every layer's gates, the message scales, the chain coefficients ----
        PRE = fnp.add(es("tki,tkh->tih", FB[1:, 0:16], P["wfb"]), es("tki,kh->tih", AT, P["wat"]))
        if QX is not None and "wqx" in P:  # r25a v2: the fold taps, arcsinh(QX / c), into the gates
            kq_ = P["wqx"].shape[0]
            Q_ = fnp.arcsinh(fnp.divide(QX[1:, 0:kq_], P["qxc"][:, :, None]))
            fnp.add(PRE, es("tki,kh->tih", Q_, P["wqx"]), out=PRE)
        SC = es("tki,kc->tic", SVB[1:, 3:5], P["msc"])
        CO = es("tki,kc->tci", FB[1:, 15:17], P["mc"])
        PREs, SCs, COs = us(PRE), us(SC), us(CO)
        # ---- pass 2: the recurrent corrector ----
        MSG = fnp.zeros((n, 100), dtype=f32)
        cp(MSG[:, 1], 1.0)
        ZB = fnp.empty((n, 128), dtype=f32)  # [x_z / 2, x_r / 2, n_in + n_h / 2, n_h / 2]
        TRZ = fnp.empty((n, 3, 32), dtype=f32)  # [1, tanh(x_z / 2), tanh(x_r / 2)]
        cp(TRZ[:, 0], 1.0)
        TMP = fnp.empty((n, 32), dtype=f32)
        ABS = fnp.empty((n,), dtype=f32)
        m_dph, m_hp, m_dp, m_m2 = MSG[:, 67:100], MSG[:, 34:67], MSG[:, 99], MSG[:, 0]
        m_in, m_nn, m_h, m_j = MSG[:, 0:67], MSG[:, 2:34], MSG[:, 67:99], MSG[:, 1:99]
        HN = MSG[:, 2:66].reshape((n, 2, 32))
        zb96, zb64, zbn, zbh = ZB[:, 0:96], ZB[:, 0:64], ZB[:, 64:96], ZB[:, 96:128]
        trz, tr, tz1 = TRZ[:, 1:3].reshape((n, 64)), TRZ[:, 2], TRZ[:, 0:2]
        for t in range(1, depth):
            if t > 1:
                W = weights[t]
                if _dtname(W.dtype) != _dtname(f32):
                    W = W.astype(f32)
                es("ia,ib->ab", W, m_dph, out=m_hp)  # [W^T h', W^T dp]
                fnp.abs(m_dp, out=ABS)
                es("ia,ia,i->a", W, W, ABS, out=m_m2)  # (W o W)^T |dp|
            es("ac,ac,kc->ak", m_in, SCs[t - 1], P["wbig"], out=ZB)
            fnp.add(zb96, PREs[t - 1], out=zb96)
            fnp.tanh(zb64, out=trz)
            fnp.multiply(tr, zbh, out=TMP)
            fnp.add(zbn, TMP, out=TMP)
            fnp.tanh(TMP, out=m_nn)
            es("ajk,jm,amk->ak", HN, P["cm"], tz1, out=m_h)  # z Hm + (1 - z) NN
            es("ca,ac->a", COs[t - 1], m_j, out=m_dp)  # D push + 1e-4 sigma (w . h' + b)
        pk = VX[L, V_PK1]
        if parts is not None:  # research only
            parts["pre15"], parts["dp15"] = fnp.copy(pk), fnp.copy(m_dp)
        fnp.add(pk, m_dp, out=pk)


class _AugEngine:
    """Law-based factored AUGMENT k_max=3 (k3_acc_aug preset "recommended") in few large flopscope operations, every
    source in the 2-leg form E = G diag(xv) + H diag(e) + Z Lz^T: only its H and G legs are dense n x n (pushed,
    rebuilt and contracted); Z is kz thin rows (pushed with the feed rows), e, xv and Lz are fixed at birth."""

    def __init__(self, dtype=None, lam=None, age=4, rank=320, sw_depth=4, sw_depth_batched=3, kz=16, kq=32,
                 lc=False, lcc=None, law=False, sw_inplace=False, sw_hi_push=(), sw_hi_tt=(), sw_hi_batched=(),
                 swp=None, wm=None, wm_stack=False, parts=None) -> None:
        self._dtype = dtype
        self._swp = swp  # Strassen-Winograd plan built in Estimator.setup() (frames and views), or None
        self._ip = bool(sw_inplace)
        self._hp, self._ht, self._hb = (frozenset(int(t) for t in s) for s in (sw_hi_push, sw_hi_tt, sw_hi_batched))
        self._lc = bool(lc)
        self._law = bool(law)
        self._lcc = lcc  # constants built in Estimator.setup(), or None
        self._wm = wm  # the W-message corrector's arrays (wmsg.npz, loaded in Estimator.setup()), or None
        self._wm_stack = bool(wm_stack)  # research only: run the learned correction too and add the corrector to it
        self._parts = parts  # research only: a dict that receives intermediate final means
        self._age, self._rank, self._d, self._d2 = age, rank, sw_depth, sw_depth_batched
        self._kz, self._kq = max(1, int(kz)), max(int(kq), int(kz))
        self._lam = LAM_REFIT if lam is None else [float(v) for v in lam]

    def propagate(self, weights):
        f32 = fnp.float32 if self._dtype is None else self._dtype
        depth = len(weights)
        n = weights[0].shape[0]
        S = depth - 1  # sources (one per layer but the last); depth >= 2 here
        mul, cp, es = fnp.multiply, fnp.copyto, fnp.einsum
        age, k, kz, kq = self._age, self._rank, min(self._kz, n), min(self._kq, n)
        # sources compressed by the end of layer depth - 2 (source s is compressed once t - s >= age)
        Smc = max(0, depth - 1 - age - _R25B.get('s3', 0)) if (age is not None and age >= 1 and 0 < k < n) else 0
        lam_tab = (list(self._lam) + [0.0] * depth)[:depth]
        lam_layer = [0.0] + lam_tab[:depth - 1] + [0.0]  # lam' used at layer t (from layer t - 1), padded
        cc, c1 = n / 2.0 + 2.0, n / 2.0 + 1.0
        kd = 12.0 / (2.0 * n + 8.0)
        kt = 24.0 * (-1.0 / (16.0 * cc * (cc - 2.0)) + 1.0 / (32.0 * c1 * (c1 - 1.0)))
        tLR, tC2, _rank = _aug_lr(n)

        # ---- constants (a fixed number of calls per predict) ----
        XM = _ex_tables(n)  # ex E5: the VEC table over its monomials (k, c, d, p, e)
        XCM = fnp.array(XM["cm"], dtype=f32)
        XKQ, XCQ, XDQ, XPQ, XEQ = (fnp.array(v_) for v_ in XM["mono"])
        cVl = fnp.array(_VEC_LAST, dtype=f32)
        cLR = fnp.array(tLR, dtype=f32)
        cSAB = fnp.array(tC2, dtype=f32)
        cSA, cSB = cSAB[0], cSAB[1]
        small = [[e / 2.0 for e in _EXP2], [float(p) for p in range(_NP)], [1.0 if e == -1 else 0.0 for e in _EXP2],
                 [1.0 if v == V_PK1 else 0.0 for v in range(_NV)], [2.0, 0.5], [3.0, 1.5], [1.0, -2.0], [1.0, 0.0],
                 [0.0, 1.0], [3.0 if v == V_W21 else 0.0 for v in range(_NV)], [2.0], [3.0], [1.0, 2.0], [0.5],
                 [3.0] * kz]
        law = self._law and (n, depth) == _AUG_SHAPE  # the per-neuron K4 law (fitted for the scored shape only)
        nsm = len(small)
        if law:
            small += [list(r) for r in _LAW_ROWS]
        wid = max(_NV, kz)
        CSM = fnp.array([r + [0.0] * (wid - len(r)) for r in small], dtype=f32)
        LAWC = CSM[nsm:nsm + 3, :_NP] if law else None
        HE = CSM[0, :_NE2][:, None]
        PEXP = CSM[1, :_NP][:, None]
        e_sig = CSM[2, :_NE2]
        e_pk1 = CSM[3, :_NV]
        c_ah, c_t, c_g = CSM[4, :2], CSM[5, :2], CSM[6, :2]  # on [3 v, 2 e]: AH = [6 v, e], t3b = [9 v, 3 e]
        e_l, e_r, e_3w21 = CSM[7, :2], CSM[8, :2], CSM[9, :_NV]  # selectors: L, R of the L/R stack; 3 w21 of VEC
        c_two, c_three, c_12, c_half, c_3z = CSM[10, :1], CSM[11, :1], CSM[12, :2], CSM[13, :1], CSM[14, :kz]
        # rows: a^0..a^4 at a = 0, [phi(0), Phi(0), 1, 0], then lam'_t for t = 0..depth (lam'_depth = 0)
        CR = fnp.array([[1.0] * n] + [[0.0] * n] * (_NP - 1) + [[0.3989422804014327] * n, [0.5] * n, [1.0] * n,
                       [0.0] * n] + [[lam_layer[t]] * n for t in range(depth + 1)], dtype=f32)
        MON0, ONEN, ZN = CR[0:_NP], CR[_NP + 2], CR[_NP + 3]
        LAMV = CR[_NP + 4:]
        LMK4 = fnp.array([[1.0], [1.0], [0.0], [0.0]], dtype=f32)  # r38: lam' rows out of the engine's VEC
        LMK6 = fnp.array([[1.0], [1.0], [0.0], [0.0], [1.0], [1.0]], dtype=f32)
        wm = self._wm if (n, depth) == _AUG_SHAPE else None  # the W-message corrector (scored shape only)
        # the learned correction (weights for the scored shape only); the corrector replaces it unless stacked
        lc = self._lc and (n, depth) == _AUG_SHAPE and (wm is None or self._wm_stack)
        if lc:
            LCT, LCSEL, LCRS, LCK, LCP, LCW1A, LCW2A, LCW1B, LCW2B, WP = (
                self._lcc if self._lcc is not None else _lc_consts(f32, n))  # WP: W^2, W^3, W^4 at the rich layers
            dp, lcR5, lcpre = None, None, None  # p' - p of the previous layer; [1, 3 t3, dA, lam', 1]; layer t-1

            def lc_rich(t, W, var, SV, pre, soff2):
                """Covariance rows of layer t: 6 logs (S off-diagonal, C' off-diagonal, S0 and 3T^T rows of layer
                t - 1 through W^2, var) and 4 linear rows (S off-diagonal row sum, diagonal-only var, W^3 K3, W^4 K4)."""
                mul(W, W, out=WP[0])
                mul(WP[0], W, out=WP[1])
                mul(WP[0], WP[0], out=WP[2])
                P2 = es("ia,ri->ra", WP[0], pre)
                P5 = fnp.stack([ONEN, VX[t - 1, V_PK1], VX[t - 1, V_PK2], VX[t - 1, V_PK3], VX[t - 1, V_PK4]])
                dk = es("pia,pi->pa", WP[1:3], es("kabcd,ai,bi,ci,di->ki", LCK, P5, P5, P5, P5))
                lg = fnp.log(fnp.stack([soff2[0], P2[1], P2[2], P2[3], P2[4], var]))
                ln = es("ri,re,ei->ri", fnp.stack([soff2[1], P2[0], dk[0], dk[1]]), LCRS, SV)
                return lg, ln

            def lc_step(t, W, a, SV, R5, dp, rich):
                """Corrected mean of layer t and its offset from the base mean; R5 = [1, 3 t3, dA, lam', 1]."""
                a2 = a if dp is None else fnp.add(a, es("ia,i,a->a", W, dp, SV[3]))  # SV[3] = 1 / sigma
                FC = fnp.stack([flops.stats.norm.pdf(a2), flops.stats.norm.cdf(a2), ONEN], dtype=f32)
                D5 = es("xi,xe,ei->xi", R5, LCSEL, SV)
                F = es("rcpxy,ci,pi,xi,yi->ri", LCT, FC, fnp.power(a2, LCP), D5, D5)
                if rich is None:
                    pc = es("k,ki->i", LCW2A, fnp.maximum(es("fk,fi->ki", LCW1A[t - 1], F), 0.0))
                else:
                    Fr = fnp.concatenate([F, rich[0], rich[1]])
                    pc = es("k,ki->i", LCW2B, fnp.maximum(es("fk,fi->ki", LCW1B[t - _LC_RICH[0]], Fr), 0.0))
                return pc, fnp.subtract(pc, VX[t, V_PK1])

        M = fnp.empty((_NK, n, n), dtype=f32)  # [S0, 3T^T, 3T, Q, S0*3T^T, S0*3T, S0^3, J, I]; empty: no symmetry tag
        M0, M1, M2, M3, M6, MJ, MI = M[K_S0], M[K_M1], M[K_M2], M[K_Q], M[K_S3], M[K_J], M[K_I]
        MBX = [M[b_[4]:b_[5]] for b_ in _EX_BOX]  # esplit: the boxes' M slices (views, once per predict)
        ESX, seen_ = [], set()  # esplit: whether a box is the first over its x range (each x covered by exactly one first box)
        for b_ in _EX_BOX:
            xs_ = set(range(b_[0], b_[1]))
            assert not (xs_ & seen_) or xs_ <= seen_
            ESX.append(not (xs_ & seen_))
            seen_ |= xs_
        assert seen_ == set(range(4))
        M01, M012, M345 = M[0:2], M[0:3], M[3:6]
        MZ = M[K_I:0:-(K_I - K_M1)]  # [I, 3T^T]: diagonals [1, 3 t3]
        M1T = M1.T
        cp(MI, fnp.eye(n, dtype=f32))
        fnp.subtract(1.0, MI, out=MJ)
        Kmat = fnp.add(mul(MI, kd), kt)  # 12 c11 I + 24 (c12 + c22) 1 1^T
        VX = fnp.zeros((depth, _NV, n), dtype=f32)
        VXs, VW11s, VW21s = fnp.unstack(VX), fnp.unstack(VX[:, V_W11]), fnp.unstack(VX[:, V_W21])  # vcache
        TH = 1 + kz  # thin rows per source: its feed row u, then its kz rows Z^T
        T0 = S * TH
        NB = 2 * S + 3  # dense blocks: [H, G] per source, two more for the last birth window, a block of ones
        LEGt = fnp.zeros((T0 + NB * n, n), dtype=f32)
        LEGt3 = LEGt[T0:].reshape((NB, n, n))
        cp(LEGt3[NB - 1], 1.0)  # permanent block of ones (row sums in the dA einsum)
        LEGt4 = LEGt3[:2 * S].reshape((S, 2, n, n))
        LEGtT = LEGt.T
        THIN = LEGt[:T0].reshape((S, TH, n))
        UP, ZT = THIN[:, 0], THIN[:, 1:]  # feed rows u; Z^T rows
        ASt = fnp.zeros((T0 + 2 * S * n, n), dtype=f32)  # [r row, Lz^T (G*G) rows] per source; [AH, AG] blocks
        ASt4 = ASt[T0:].reshape((S, 2, n, n))
        ATHIN = ASt[:T0].reshape((S, TH, n))
        AStU, AZT = ATHIN[:, 0], ATHIN[:, 1:]
        Ht, Gt = LEGt4[:, 0], LEGt4[:, 1]
        AH, AG = ASt4[:, 0], ASt4[:, 1]
        GH = LEGt4[:, ::-1]  # [G, H]
        CSRC = VX[:S, V_C]  # c = (g / 4) lam' w11 at birth
        # per source [[3 v, 2 e], [0, 3 xv]] (v = w21 at birth, cv = 3 v; e: E's H coefficient; xv: E's birth diagonal),
        # so 3 A2 + xv 3 A3 = 3 v HH + 2 e HG + 3 xv GG is one einsum over the leg pair
        COEFG = fnp.zeros((S, 2, 2, n), dtype=f32)
        C3V2E, XV3 = COEFG[:, 0], COEFG[:, 1, 1]
        LZ = fnp.zeros((S, n, kz), dtype=f32)  # per source Lz (birth index x residual rank)
        LZT = fnp.swapaxes(LZ, 1, 2)
        SG = fnp.empty((S, n, n), dtype=f32)  # scratch: G*G, then Lz Z^T
        E1CH = [(Ht[c0_:c0_ + 8], Gt[c0_:c0_ + 8], CSRC[c0_:c0_ + 8], AStU[c0_:c0_ + 8], LEGt4[c0_:c0_ + 8],
                 C3V2E[c0_:c0_ + 8], AH[c0_:c0_ + 8], COEFG[c0_:c0_ + 8], AG[c0_:c0_ + 8], ASt4[c0_:c0_ + 8],
                 GH[c0_:c0_ + 8], UP[c0_:c0_ + 8]) for c0_ in range(0, S, 8)]  # e1c: chunk views, made once
        GSL = [(Gt[:j_], SG[:j_]) for j_ in range(S + 1)]  # e1c: live-slot views of G and SG, made once
        K4F = _K4F.get('scale')  # r25a k4f
        if K4F is not None:  # per source -scale w31 of its birth layer (row m: birth neuron)
            W3F = fnp.zeros((S, n), dtype=f32)
            W3Fs, VW31s = fnp.unstack(W3F), fnp.unstack(VX[:, V_W31])
            K4L = ([None] + [(Ht[:j_], Gt[:j_], W3F[:j_], AH[:j_], AG[:j_], SG[:j_]) for j_ in range(1, S + 1)]
                   if _K4F.get('live') else None)  # live slots only
        GF = _GF.get('scale')  # r25a gf
        if GF is not None:  # per source -scale alpha / sigma of its birth layer (row m: birth neuron)
            LAMG = fnp.zeros((S, n), dtype=f32)
            LAMGs = fnp.unstack(LAMG)
        QXON = _QX.get('on') and K4F is not None and GF is not None  # r25a qx
        QXX, QRE = (_QX.get('x') or {}) if QXON else {}, None  # r25a qx2: tap -> QX row; tap 6's carried energy
        if QXON:
            QX = fnp.zeros((depth, _QX.get('rows', 4), n), dtype=f32)
            _QX['buf'] = QX
            if 6 in QXX:
                QRE = fnp.zeros((n,), dtype=f32)
        # sat fbuf: flat views of the buffers the restriction's gathers borrow while they are dead (made once)
        ASTF, SGF, LEGTF, MQF = ASt.reshape(-1), SG.reshape(-1), LEGt.reshape(-1), M[3:7].reshape(-1)
        if Smc:
            # compressed source s: [H, G]^T (2n, n) = Ct_s (2n, k) @ Qt_s (k, n), Qt_s with orthonormal rows
            LEGtS = LEGt[T0:T0 + 2 * S * n].reshape((S, 2 * n, n))
            AStS = ASt[T0:].reshape((S, 2 * n, n))
            QtS = fnp.zeros((Smc * k, n), dtype=f32)  # bases, stacked
            QtS3 = QtS.reshape((Smc, k, n))
            CtS = fnp.zeros((Smc, 2 * n, k), dtype=f32)  # coefficients
            CtS4 = CtS.reshape((Smc, 2, n, k))
            CtST = fnp.swapaxes(CtS, 1, 2)
            PS = fnp.zeros((Smc * k, n), dtype=f32)  # P_s = Ct_s^T [AH; AG]_s (k, n), stacked
            PS3 = PS.reshape((Smc, k, n))
            SCR = ASt[T0:T0 + 2 * n]  # scratch for the normalized legs of the source being compressed (A is dead then)
            SCR3 = SCR.reshape((2, n, n))
        # Strassen-Winograd work frames on one arena, sized for the longest push / T^T operand of the schedule
        # (dense leg rows only: the thin u and Z^T rows go through small plain products)
        d, d2 = self._d, (self._d2 if Smc else 0)
        hp, ht, hb = self._hp, self._ht, (self._hb if d2 else frozenset())
        CF = _covfold_layers(n, depth, d, ht)  # covfold: layers whose push carries C (S0 = W^T (C W) by Strassen)
        swp = self._swp  # the setup() plan when it is for this shape and setting: its frames are carved again (billed)
        if swp is not None and swp.key == (n, depth, age, k, d, d2, hp, ht, hb, self._ip, CF):
            pass  # r143-B norecarve: setup() made the frames (views of the kept workspace); nothing to carve again
        else:  # carve here, views per call
            swp = _SwPlan(f32, n, depth, age, k, d, d2, hp, ht, hb, self._ip, views=False, cf=CF)
        ip = self._ip
        qM1 = {}  # rsw: root quadrants of the 3 T^T output M1 per Strassen-Winograd depth (made at its first 3 T^T)
        qM0 = {}  # covfold: root quadrants of S0's buffer M0 per depth (made at the first fold layer)
        RQ = {}  # rsw: root quadrants of the batched products' fixed operands, once per predict (one reshape each)
        if d2:
            for dd_ in sorted({v_[0] for kk_, v_ in swp.calls.items() if kk_[0] in ("rebuild", "p")}):
                RQ[dd_] = (_rq(_rt(CtS, dd_)), _rq(_rt(QtS3, dd_, True)), _rq(_rt(LEGtS[:Smc], dd_)),
                           _rq(_rt(CtST, dd_)), _rq(_rt(AStS[:Smc], dd_, True)), _rq(_rt(PS3, dd_)))
        # per-source and per-block views that the layers index, one unstack each (every view is a round trip)
        us = fnp.unstack
        Gs, UPs, ZTs, LZs, XV3s = us(Gt), us(UP), us(ZT), us(LZ), us(XV3)
        C3Vs, C2Es = us(COEFG[:, 0, 0]), us(COEFG[:, 0, 1])  # 3 v, 2 e per source
        BLK, LAMVs = us(LEGt3), us(LAMV)
        if Smc:
            LEG4s, QtS3s, CtS4s = us(LEGt4), us(QtS3), us(CtS4)
        if wm is not None:  # the corrector reads every layer's per-neuron quantities after the loop (written in place)
            QB = fnp.zeros((depth, 9, n), dtype=f32)  # [phi, Phi, 1, dA, lam'_t, lam'_t+1, t3, a, var]
            SVB = fnp.empty((depth, _NE2, n), dtype=f32)
            QFFs, QAs, QVs, SVBs = us(QB[:, 0:6]), us(QB[:, 7]), us(QB[:, 8]), us(SVB)

        Cv = None
        dA = ZN
        ns = 0
        mc = 0  # sources compressed so far (always the oldest: 0..mc-1)
        ord_p = None  # sat: argsort of the previous layer's alpha (ascending)
        dc_on = _DC['theta'] is not None and d > 0
        dc_g = dc_on and _DC['gather']  # dc: gather the kept rows (False: zeros only, the reference)
        DCK, DCH, DCG = {}, {}, {}  # dc: keep masks (sources with dropped centers), H and G row indices
        DCHP = {}  # dc: a newest source's H rows padded to 2^(d + 1) for a push in hp
        if dc_g:  # gathered push / 3 T^T operands (rows: parked bases, the dense sources' kept rows, C)
            dc_max = (Smc * k if Smc else 0) + (2 * min(S, age if age else S) + 1 + 2 * _R25B.get('s3', 0)) * n
            DCXB = getattr(swp, "dcx", None)  # dcfix: made and first-touched once in setup() with the plan, reused
            if DCXB is None or tuple(DCXB.shape) != (2, dc_max, n):
                DCXB = fnp.empty((2, dc_max, n), dtype=f32)
            DCX1, DCX2 = fnp.unstack(DCXB)
        dyn_a_prev = None
        r25_pin = r25_pout = n  # r25b s8
        SAT_THR, cin_p = None, None
        for t in range(depth):
            W = weights[t]
            if _dtname(W.dtype) != _dtname(f32):
                W = W.astype(f32)
            last = t == depth - 1
            pre_a = False  # r25b s2: a and SV made at the last push
            dstart = T0 + 2 * mc * n  # first dense leg row; compressed sources' legs live in rows [T0, dstart)
            lo = dstart - mc * k  # the bases (and, for T, the blocks P) are parked in rows [lo, dstart)
            nth = ns * TH  # thin rows of the live sources
            thin = LEGt[:nth] if ns else None
            Qpark, Qv = (LEGt[lo:dstart], QtS[:mc * k]) if mc else (None, None)
            if lc and t in _LC_RICH:  # layer t - 1's C', S0^2 and 3T^T rows before the cov step and window reuse them
                with _ns("lc"):
                    lcpre = fnp.stack([es("ii->i", Cv), es("ij,ij->i", Cv, Cv), es("ij->i", M3),
                                       es("ij,ij->i", M1, M1), es("ij,ij->j", M1, M1)])
                    fnp.subtract(lcpre[1], mul(lcpre[0], lcpre[0]), out=lcpre[1])  # C' off-diagonal

            # ---- linear step ----
            lastvar = last and not lc and ns > 0  # ex E3: the last layer's variance comes out of the push (C' W)
            fold = t in CF and ns > 0
            DGF_t, LAM_t, GHL_t = None, None, None  # r30b q0, q12: this layer's DGF, lam, G H LAMG
            if fold:  # covfold: the unrestricted push first, C's rows in the newest source's dead G slot (T = C W)
                with _ns("push"):
                    fnp.matmul(thin, W, out=thin)  # u and Z^T rows of the live sources (plain products)
                    if mc:
                        cp(Qpark, Qv)
                    cp(Gs[ns - 1], Cv)
                    blk = LEGt[lo:T0 + 2 * ns * n]  # bases, dense legs and C
                    if dc_g and any(s_ in DCK for s_ in range(mc, ns)):  # dc: bases, kept rows, all of C
                        pc_ = [fnp.arange(lo, dstart, dtype=fnp.int32)] if mc else []
                        for s_ in range(mc, ns):
                            pc_.append(DCHP[s_] if (s_ == ns - 1 and t in hp) else DCH[s_])
                            pc_.append(DCG[s_] if s_ < ns - 1 else
                                       fnp.arange(T0 + (2 * s_ + 1) * n, T0 + (2 * s_ + 2) * n, dtype=fnp.int32))
                        RI_ = fnp.concatenate(pc_)
                        m_ = RI_.shape[0]
                        Xr = DCX1[:m_]
                        fnp.take(LEGt, RI_, axis=0, out=Xr, mode="clip")
                        ddf_ = d + 1 if t in hp else d
                        Pin_ = n
                        if _R25B.get('s1') and n == 1024 and t >= _R25B['s1'] and cin_p is not None and ord_p is not None:
                            Pin_ = _r25_pad(cin_p, 1 << ddf_)
                            Pin_ = next((x for x in _R25B['s1_bins'].get(t, []) if x >= Pin_), n)
                        if Pin_ < n:  # r25b s1: leg rows over layer t-1's alive top-Pin inner columns; C's n rows in full
                            ml_ = m_ - n
                            Iin_ = ord_p[n - Pin_:]
                            Xs_ = fnp.take(Xr[:ml_], Iin_, axis=1, out=ASTF[:ml_ * Pin_].reshape((ml_, Pin_)), mode="clip")
                            Ws_ = fnp.take(W, Iin_, axis=0, out=SGF[:Pin_ * n].reshape((Pin_, n)), mode="clip")
                            dd_, va_, vb_, vm_ = _r25_bview(swp, "push_k%d" % Pin_, ddf_, (), ml_, Pin_, n)
                            _sw(Xs_, Ws_, Xr[:ml_], va_, vb_, vm_, dd_, ip)
                            dd_, va_, vb_, vm_ = _r25_bview(swp, "pushC", ddf_, (), n, n, n)
                            _sw(Xr[ml_:], W, Xr[ml_:], va_, vb_, vm_, dd_, ip)
                        else:
                            dd_, va_, vb_, vm_ = swp.view_shape("push", ddf_, m_, n, n)
                            _sw(Xr, W, Xr, va_, vb_, vm_, dd_, ip)
                        fnp.put_along_axis(LEGt, fnp.broadcast_to(RI_[:, None], (m_, n)), Xr, axis=0)
                    else:
                        dd_, va_, vb_, vm_ = swp.view(("push", t))
                        _sw(blk, W, blk, va_, vb_, vm_, dd_, ip)
                with _ns("cov"):  # S0 = W^T T by Strassen-Winograd, made exactly symmetric (M3 is rewritten later)
                    _covsym(W, Gs[ns - 1], M0, swp, ip)  # r143-B covsym (M3 is rewritten later)
                with _ns("push"):
                    cp(Gs[ns - 1], W)  # newest G^T = W (its slot held the birth E, dead after the 2-leg split)
                    if (ns - 1) in DCK:  # dc: the dropped centers' G rows
                        mul(Gs[ns - 1], DCK[ns - 1][:, None], out=Gs[ns - 1])
                    if mc:
                        cp(Qv, Qpark)
            with _ns("cov"):
                if t == 0:
                    _covsym(W, W, M0, swp, ip)  # r143-B covsw: W^T W from blocks
                    cp(M[1:3], 0.0)  # no T before the first source
                elif not lastvar and not fold and not last:  # r143-B covsw: T = C W into M3, S0 from blocks
                    dT_, vaT, _, _ = swp.view(("covT", 0))
                    _sw(Cv, W, M3, vaT, None, None, dT_, ip)
                    _covsym(W, M3, M0, swp, ip)
                elif not lastvar and not fold:
                    Cs = flops.as_symmetric(Cv, symmetry=(0, 1))
                    if last and not lc:
                        var_raw = es("ij,ia,ja->a", Cs, W, W)
                    else:  # with the correction, the last layer forms the full S too (its rows feed the correction)
                        es("ij,ia,jb->ab", Cs, W, W, out=M0)
                if not lastvar:
                    if not last or lc:
                        var_raw = fnp.diagonal(M0)
                    if wm is None:
                        var = fnp.maximum(var_raw, 1e-10)
                    else:
                        var = QVs[t]
                        fnp.maximum(var_raw, 1e-10, out=var)

            # ---- sat: this layer's alpha (moved here from the per-neuron step) and the restricted sizes ----
            early = 0 < t < depth - 1
            Pp, Pa, Pb, ord_t, ordb_t = n, n, n, None, None
            if early:
                with _ns("wick_pk"):
                    if wm is None:
                        SV = fnp.power(var, HE)
                        a = es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig)  # VEC: the previous layer's rows
                    else:
                        SV = SVBs[t]
                        fnp.power(var, HE, out=SV)
                        a = QAs[t]
                        es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig, out=a)
                if d:
                    with _ns("sat"):
                        dyn = _SAT["mode"] != "fixed"
                        th_in, th_out, th_ta, th_tb = _SAT["in"], _SAT["out"], _SAT["ta"], _SAT["tb"]
                        rb = ("band_" if _SAT["tb_band"] else "alive_") + str(th_tb)
                        dtt = d + 1 if t in ht else d
                        q_ = _SAT.get("q", 32)
                        if dyn:  # this MLP's counts from the sorted alpha: one sort, one search, one read
                            ord_t = fnp.argsort(a)  # ascending: the top-P by alpha are its last P entries
                            if SAT_THR is None:
                                SAT_THR = fnp.array([-th_in, -th_out, -th_ta, -th_tb, th_tb], dtype=f32)
                            cs_ = fnp.searchsorted(fnp.take(a, ord_t), SAT_THR).tolist()
                            c_out, c_ta = n - cs_[1], n - cs_[2]
                            c_tb = (cs_[4] - cs_[3]) if _SAT["tb_band"] else n - cs_[3]
                        else:
                            c_out = c_ta = c_tb = None
                        if ord_p is not None:
                            p_in = _sat_size("alive_" + str(th_in), t - 1, n, max(1 << d, q_), cin_p if dyn else None)
                            r25_pin = p_in  # r25b s8: this push's inner size
                            r25_pout = _r25_pad(n - cs_[3] if _R25B.get('s8', {}).get('out') == 3.5 else c_out, max(1 << d, q_)) if dyn else n
                            p_out = _sat_size("alive_" + str(th_out), t, n, max(1 << d, q_), c_out)
                            Pp = max(p_in, p_out)
                        if not last:
                            Pa = _sat_size("alive_" + str(th_ta), t, n, max(1 << dtt, q_), c_ta)
                            Pb = _sat_size(rb, t, n, max(1 << dtt, q_), c_tb)
                        if _SAT["mode"] == "bins":  # round up to the setup plan's bins (the fold layers keep an unrestricted push)
                            pu_, tt_ = _sat_bins(t, n, d, dtt, q_, CF)
                            Pp = n if t in CF else next((x for x in pu_ if x >= Pp), n)
                            if Pa < n or Pb < n:
                                Pa, Pb = next(((x, y) for x, y in tt_ if x >= Pa and y >= Pb), (n, n))
                        cin_p = (n - cs_[0]) if dyn else None  # the next push's inner count
                        nxt = dyn or _sat_size("alive_" + str(th_in), t, n, max(1 << d, q_)) < n  # the next push reads this order
                        if not dyn and (Pp < n or Pa < n or Pb < n or nxt):
                            ord_t = fnp.argsort(a)  # ascending: the top-P by alpha are its last P entries
                        if Pb < n and _SAT["tb_band"]:
                            ordb_t = fnp.argsort(fnp.abs(a))  # the first P have the smallest |alpha|

            # ---- push the thin rows and the dense legs through W^T ----
            with _ns("push"):
                s2l_ = bool(_R25B.get('s2')) and n == 1024 and last and ns and not fold and dc_g and ord_p is not None \
                    and cin_p is not None and any(s_ in DCK for s_ in range(mc, ns))
                s7_ = bool(_R25B.get('s7')) and n == 1024 and not last and mc > 0 and bool(d2) and Pb < n \
                    and ordb_t is not None and _SAT["tb_band"]  # r25b s7
                Ib7_ = ordb_t[:Pb] if s7_ else None
                if s2l_:  # r25b s2: C' rows alone in full (var of every neuron), a, then the leg rows on the band
                    o2_ = _R25B['s2']
                    fnp.matmul(thin, W, out=thin)  # u and Z^T rows of the live sources (plain products)
                    if mc:
                        cp(Qpark, Qv)
                    cp(Gs[ns - 1], Cv)
                    dd_, va_, vb_, vm_ = _r25_bview(swp, "pushC", d + 1 if t in hp else d, (), n, n, n)
                    _sw(Gs[ns - 1], W, Gs[ns - 1], va_, vb_, vm_, dd_, ip)
                    var_raw = es("ja,ja->a", W, Gs[ns - 1])
                    if wm is None:
                        var = fnp.maximum(var_raw, 1e-10)
                        SV = fnp.power(var, HE)
                        a = es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig)
                    else:
                        var = QVs[t]
                        fnp.maximum(var_raw, 1e-10, out=var)
                        SV = SVBs[t]
                        fnp.power(var, HE, out=SV)
                        a = QAs[t]
                        es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig, out=a)
                    pre_a = True
                    aa_ = fnp.abs(a)
                    ordb_ = fnp.argsort(aa_)  # the first Pb have the smallest |alpha|
                    cb_ = fnp.searchsorted(fnp.take(aa_, ordb_), fnp.array([o2_['band']], dtype=f32)).tolist()[0]
                    Pi_, Pb_ = _r25_up(o2_['pairs'], _r25_pad(cin_p, o2_['g']), _r25_pad(cb_, o2_['g']), n)
                    Iin_, Ib_ = ord_p[n - Pi_:], ordb_[:Pb_]
                    pc_ = [fnp.arange(lo, dstart, dtype=fnp.int32)] if mc else []
                    for s_ in range(mc, ns):
                        pc_.append(DCHP[s_] if (s_ == ns - 1 and t in hp) else DCH[s_])
                        if s_ < ns - 1:
                            pc_.append(DCG[s_])
                    RI_ = fnp.concatenate(pc_)
                    m_ = RI_.shape[0]
                    Xr = DCX1[:m_]
                    fnp.take(LEGt, RI_, axis=0, out=Xr, mode="clip")
                    Xs = fnp.take(Xr, Iin_, axis=1, out=ASTF[:m_ * Pi_].reshape((m_, Pi_)), mode="clip")
                    W1_ = fnp.take(W, Iin_, axis=0, out=SGF[:Pi_ * n].reshape((Pi_, n)), mode="clip")
                    Ws = fnp.take(W1_, Ib_, axis=1, out=SGF[Pi_ * n:Pi_ * n + Pi_ * Pb_].reshape((Pi_, Pb_)), mode="clip")
                    o_ = Pi_ * n + Pi_ * Pb_
                    Ys = SGF[o_:o_ + m_ * Pb_].reshape((m_, Pb_))
                    if o2_['apa']:
                        dd_, va_, vb_, vm_ = swp.view_shape("push", d, m_, Pi_, Pb_)
                        _swk("push", swp, Xs, Ws, Ys, va_, vb_, vm_, dd_, ip)
                    else:
                        ks2_ = "push_s2_%d_%d" % (Pi_, Pb_)
                        if (ks2_, d + 1 if t in hp else d) in swp.frames:  # r25b: exact frames of this pair
                            dd_, va_, vb_, vm_ = _r25_bview(swp, ks2_, d + 1 if t in hp else d, (), m_, Pi_, Pb_)
                        else:
                            dd_, va_, vb_, vm_ = swp.view_shape("push", d + 1 if t in hp else d, m_, Pi_, Pb_)
                        _sw(Xs, Ws, Ys, va_, vb_, vm_, dd_, ip)
                    cp(Xr, 0.0)
                    fnp.put_along_axis(Xr, Ib_[None, :], Ys, axis=1)
                    fnp.put_along_axis(LEGt, fnp.broadcast_to(RI_[:, None], (m_, n)), Xr, axis=0)
                    cp(Gs[ns - 1], W)  # newest G^T = W
                    if (ns - 1) in DCK:  # dc: the dropped centers' G rows
                        mul(Gs[ns - 1], DCK[ns - 1][:, None], out=Gs[ns - 1])
                    if mc:
                        cp(Qv, Qpark)
                elif ns and not fold:
                    fnp.matmul(thin, W, out=thin)  # u and Z^T rows of the live sources (plain products)
                    if mc:
                        cp(Qpark, Qv)
                    if lastvar:  # ex E3: C' into the newest source's dead G slot, pushed with the legs
                        cp(Gs[ns - 1], Cv)
                    blk = LEGt[lo:T0 + (2 * ns - (0 if last else 1)) * n]  # bases, dense legs (and C' at the last layer)
                    RI_ = None
                    if dc_g and any(s_ in DCK for s_ in range(mc, ns)):  # dc: the kept rows only
                        pc_ = [fnp.arange(lo, dstart, dtype=fnp.int32)] if mc else []
                        for s_ in range(mc, ns):
                            pc_.append(DCHP[s_] if (s_ == ns - 1 and t in hp) else DCH[s_])
                            if s_ < ns - 1:
                                pc_.append(DCG[s_])
                            elif last:  # the newest G slot holds C' (all n rows)
                                pc_.append(fnp.arange(T0 + (2 * s_ + 1) * n, T0 + (2 * s_ + 2) * n, dtype=fnp.int32))
                        RI_ = fnp.concatenate(pc_)
                    if RI_ is not None:
                        m_ = RI_.shape[0]
                        Xr = DCX1[:m_]
                        fnp.take(LEGt, RI_, axis=0, out=Xr, mode="clip")
                        if Pp < n and _R25B.get('s8') and n == 1024 and t in _R25B['s8']['pairs']:  # r25b s8: inner and output sizes apart
                            Pi_, Po_ = _r25_up(_R25B['s8']['pairs'][t], r25_pin, r25_pout, n)
                            Iin, Iout = ord_p[n - Pi_:], ord_t[n - Po_:]
                            Xs = fnp.take(Xr, Iin, axis=1, out=ASTF[:m_ * Pi_].reshape((m_, Pi_)), mode="clip")
                            W1_ = fnp.take(W, Iin, axis=0, out=SGF[:Pi_ * n].reshape((Pi_, n)), mode="clip")
                            Ws = fnp.take(W1_, Iout, axis=1, out=SGF[Pi_ * n:Pi_ * n + Pi_ * Po_].reshape((Pi_, Po_)), mode="clip")
                            o_ = Pi_ * n + Pi_ * Po_
                            Ys = SGF[o_:o_ + m_ * Po_].reshape((m_, Po_))
                            dd_, va_, vb_, vm_ = swp.view_shape("push", d, m_, Pi_, Po_)
                            _swk("push", swp, Xs, Ws, Ys, va_, vb_, vm_, dd_, ip)
                            cp(Xr, 0.0)
                            fnp.put_along_axis(Xr, Iout[None, :], Ys, axis=1)
                        elif Pp < n:
                            Iin, Iout = ord_p[n - Pp:], ord_t[n - Pp:]
                            Xs = fnp.take(Xr, Iin, axis=1, out=ASTF[:m_ * Pp].reshape((m_, Pp)), mode="clip")
                            W1_ = fnp.take(W, Iin, axis=0, out=SGF[:Pp * n].reshape((Pp, n)), mode="clip")
                            Ws = fnp.take(W1_, Iout, axis=1, out=SGF[Pp * n:Pp * n + Pp * Pp].reshape((Pp, Pp)), mode="clip")
                            dd_, va_, vb_, vm_ = swp.view_shape("push", d, m_, Pp, Pp)
                            _swk("push", swp, Xs, Ws, Xs, va_, vb_, vm_, dd_, ip)
                            cp(Xr, 0.0)
                            fnp.put_along_axis(Xr, Iout[None, :], Xs, axis=1)
                        else:
                            dd_, va_, vb_, vm_ = swp.view_shape("push", d + 1 if t in hp else d, m_, n, n)
                            _swk("push" if not last else "pushc", swp, Xr, W, Xr, va_, vb_, vm_, dd_, ip)
                        fnp.put_along_axis(LEGt, fnp.broadcast_to(RI_[:, None], (m_, n)), Xr, axis=0)
                    elif Pp < n:  # sat: inner columns alive at t - 1, output columns alive at t (one size, in place)
                        Iin, Iout = ord_p[n - Pp:], ord_t[n - Pp:]
                        if _SAT["mode"] == "mask":
                            _sat_note(t, "in", blk, _sat_mask(Iin, n, f32))
                            mul(blk, _sat_mask(Iin, n, f32), out=blk)
                            dd_, va_, vb_, vm_ = swp.view(("push", t))
                            _swk("push" if not last else "pushc", swp, blk, W, blk, va_, vb_, vm_, dd_, ip)
                            _sat_note(t, "out", blk, _sat_mask(Iout, n, f32))
                            mul(blk, _sat_mask(Iout, n, f32), out=blk)
                        else:
                            m_ = blk.shape[0]  # sat fbuf: ASt and SG are dead during the push
                            Xs = fnp.take(blk, Iin, axis=1, out=ASTF[:m_ * Pp].reshape((m_, Pp)), mode="clip")
                            W1_ = fnp.take(W, Iin, axis=0, out=SGF[:Pp * n].reshape((Pp, n)), mode="clip")
                            Ws = fnp.take(W1_, Iout, axis=1, out=SGF[Pp * n:Pp * n + Pp * Pp].reshape((Pp, Pp)), mode="clip")
                            dd_, va_, vb_, vm_ = swp.view_shape("push", d, m_, Pp, Pp)
                            _swk("push", swp, Xs, Ws, Xs, va_, vb_, vm_, dd_, ip)
                            cp(blk, 0.0)
                            fnp.put_along_axis(blk, Iout[None, :], Xs, axis=1)
                    elif d:
                        dd_, va_, vb_, vm_ = swp.view(("push", t))
                        _swk("push" if not last else "pushc", swp, blk, W, blk, va_, vb_, vm_, dd_, ip)
                    else:
                        fnp.matmul(blk, W, out=blk)
                    if lastvar:  # var = diag(W^T C' W) = sum_j W_ja (C' W)_ja
                        var_raw = es("ja,ja->a", W, Gs[ns - 1])
                        if wm is None:
                            var = fnp.maximum(var_raw, 1e-10)
                        else:
                            var = QVs[t]
                            fnp.maximum(var_raw, 1e-10, out=var)
                    cp(Gs[ns - 1], W)  # newest G^T = W (its slot held the birth E, dead after the 2-leg split)
                    if (ns - 1) in DCK:  # dc: the dropped centers' G rows
                        mul(Gs[ns - 1], DCK[ns - 1][:, None], out=Gs[ns - 1])
                    if mc:
                        cp(Qv, Qpark)
            with _ns("rebuild"):
                if ns and mc:  # compressed legs, rebuilt
                    if d2 and s2l_ and o2_.get('rb'):  # r25b s2b: the compressed legs on the band columns only
                        QtB_ = fnp.take(QtS3[:mc], Ib_, axis=2, out=SGF[:mc * k * Pb_].reshape((mc, k, Pb_)), mode="clip")
                        LegB_ = ASTF[:mc * 2 * n * Pb_].reshape((mc, 2 * n, Pb_))
                        krb_ = "rebuild_k%d" % Pb_ if ("rebuild_k%d" % Pb_, d2) in swp.frames else "rebuild"
                        dd_, va_, vb_, vm_ = _r25_bview(swp, krb_, d2, (mc,), 2 * n, k, Pb_)
                        _sw(CtS[:mc], QtB_, LegB_, va_, vb_, vm_, dd_, ip)
                        cp(LEGtS[:mc], 0.0)
                        fnp.put_along_axis(LEGtS[:mc], fnp.broadcast_to(Ib_[None, None, :], (mc, 2 * n, Pb_)), LegB_, axis=2)
                    elif d2 and s7_:  # r25b s7: the compressed legs on the 3 T^T band columns only
                        QtB_ = fnp.take(QtS3[:mc], Ib7_, axis=2, out=SGF[:mc * k * Pb].reshape((mc, k, Pb)), mode="clip")
                        LegB_ = ASTF[:mc * 2 * n * Pb].reshape((mc, 2 * n, Pb))
                        kr_ = "rebuild_k%d" % Pb if ("rebuild_k%d" % Pb, d2) in swp.frames else "rebuild"
                        dd_, va_, vb_, vm_ = _r25_bview(swp, kr_, d2, (mc,), 2 * n, k, Pb)
                        _sw(CtS[:mc], QtB_, LegB_, va_, vb_, vm_, dd_, ip)
                        fnp.put_along_axis(LEGtS[:mc], fnp.broadcast_to(Ib7_[None, None, :], (mc, 2 * n, Pb)), LegB_, axis=2)
                    elif d2:
                        dd_, va_, vb_, vm_ = swp.view(("rebuild", t))
                        RQ_ = RQ[dd_]
                        _sw(None, None, None, va_, vb_, vm_, dd_, ip, qa0=_rqc(RQ_[0], mc, Smc),
                            qb0=_rqc(RQ_[1], mc, Smc), qp0=_rqc(RQ_[2], mc, Smc))
                    else:
                        fnp.matmul(CtS[:mc], QtS3[:mc], out=LEGtS[:mc])

            # ---- slices: 3 T^T (diagonal 3 t3) with the feed and the thin legs, or t3 alone at the last layer ----
            with _ns("k3_slices"):
                # elementwise work runs over every slot (unborn slots carry zero coefficients), the thin products over
                # the live sources only
                if ns:
                    SGL = GSL[ns][1]
                    mul(GSL[ns][0], GSL[ns][0], out=GSL[ns][1])  # G*G over the live slots (out=: no temporary; e1c)
                    fnp.matmul(LZT[:ns], SGL, out=AZT[:ns])  # Lz^T (G*G): E's thin part against 3 A3 = G*G
                if ns and last:
                    t3b = fnp.add(es("sxji,sji,sji,sxj,x->i", LEGt4, Ht, Gt, C3V2E, c_t),
                                  es("sji,sji,sj->i", SG, Gt, XV3))  # 9 v HHG + 3 e HGG + 3 xv GGG
                    t3b = fnp.add(t3b, es("sci,sci,c->i", ZT[:ns], AZT[:ns], c_3z))  # + 3 (Lz Z^T) GG
                    t3f = es("si,sji,sji,sj->i", UP, Ht, Gt, CSRC)  # feed: u r, r = (H*G) c
                elif ns:
                    E1L_ = _r28_sl(E1CH, ns) if _R28B.get('sl') else E1CH[:(ns + 7) // 8]  # r28b sl: live slots only
                    for h_, g_, cs_, au_, l4_, c3_, ah_, cg_, ag_, a4_, gh_, up_ in E1L_:  # e1c
                        es("sji,sji,sj->si", h_, g_, cs_, out=au_)  # r = (H*G) c
                        es("sxji,sji,sxj,x->sji", l4_, g_, c3_, c_ah, out=ah_)  # 6 v HG + e GG
                        es("sxji,syji,sxyj->sji", l4_, l4_, cg_, out=ag_)  # 3 v HH + 2 e HG + 3 xv GG
                    AGL = AG[:ns]
                    fnp.matmul(LZ[:ns], ZT[:ns], out=SGL)  # Lz Z^T (E's thin part, dense)
                    fnp.add(AGL, es("sji,sji,x->sji", SGL, GSL[ns][0], c_two), out=AGL)  # + 2 (Lz Z^T) G
                    for h_, g_, cs_, au_, l4_, c3_, ah_, cg_, ag_, a4_, gh_, up_ in E1L_:  # e1c
                        fnp.add(a4_, es("sxji,sj,si->sxji", gh_, cs_, up_), out=a4_)  # + [G, H] diag(c) (x) u
                    if K4F is not None:  # r25a k4f: the K4 star (3,1) slice rides on the A rows (header)
                        with _ns("k4f"):
                            h_, g_, w_, ah_, ag_, sg_ = K4L[ns] if K4L is not None else (Ht, Gt, W3F, AH, AG, SG)
                            LAM_t = mul(a, SV[3])  # r30b q0: lam, kept for tap 0
                            QF = es("sji,sji,sj,i->sji", h_, h_, w_, LAM_t, out=sg_)  # s w31 lam H^2
                            fnp.add(ah_, es("sji,sji,x->sji", QF, g_, c_three), out=ah_)  # + 3 s w31 lam G H^2
                            fnp.add(ag_, mul(QF, h_), out=ag_)  # + s w31 lam H^3
                            DGF = es("sji,sji,sji->i", QF, g_, h_)  # 3T^T diagonal share / 4
                            if _R30B.get('q0') and K4L is not None:  # r30b q0: DGF = lam (tap 0) on the live slots
                                DGF_t = DGF
                    if mc:  # compressed sources enter T^T as Qt_s^T P_s, P_s = Ct_s^T [AH; AG]_s, parked
                        with _ns("P"):
                            if d2 and s7_:  # r25b s7: P on the band columns (zero elsewhere); thin A rows kept on the band
                                km7_ = fnp.zeros((n,), dtype=f32)
                                fnp.put_along_axis(km7_, Ib7_, fnp.ones((Pb,), dtype=f32), axis=0)
                                mul(ATHIN[:mc], km7_, out=ATHIN[:mc])
                                AB_ = fnp.take(AStS[:mc], Ib7_, axis=2, out=SGF[:mc * 2 * n * Pb].reshape((mc, 2 * n, Pb)), mode="clip")
                                PB_ = MQF[:mc * k * Pb].reshape((mc, k, Pb))
                                kp_ = "p_k%d" % Pb if ("p_k%d" % Pb, d2) in swp.frames else "p"
                                dd_, va_, vb_, vm_ = _r25_bview(swp, kp_, d2, (mc,), k, 2 * n, Pb)
                                _sw(CtST[:mc], AB_, PB_, va_, vb_, vm_, dd_, ip)
                                cp(PS3[:mc], 0.0)
                                fnp.put_along_axis(PS3[:mc], fnp.broadcast_to(Ib7_[None, None, :], (mc, k, Pb)), PB_, axis=2)
                            elif d2:
                                dd_, va_, vb_, vm_ = swp.view(("p", t))
                                RQ_ = RQ[dd_]
                                _sw(None, None, None, va_, vb_, vm_, dd_, ip, qa0=_rqc(RQ_[3], mc, Smc),
                                    qb0=_rqc(RQ_[4], mc, Smc), qp0=_rqc(RQ_[5], mc, Smc))
                            else:
                                fnp.matmul(CtST[:mc], AStS[:mc], out=PS3[:mc])
                            cp(ASt[lo:dstart], PS[:mc * k])
                            cp(Qpark, Qv)
                    with _ns("TT"):  # 3 T^T over the dense legs (and parked blocks), plus the thin rows' u r^T, Z AZ
                        RIt_ = None
                        if dc_g and any(s_ in DCK for s_ in range(mc, ns)):  # dc: the kept rows only
                            pc_ = [fnp.arange(lo, dstart, dtype=fnp.int32)] if mc else []
                            for s_ in range(mc, ns):
                                pc_.append(DCH[s_])
                                pc_.append(DCG[s_])
                            RIt_ = fnp.concatenate(pc_)
                        if RIt_ is not None:
                            K_ = RIt_.shape[0]
                            Lr = DCX1[:K_]
                            Ar = DCX2[:K_]
                            fnp.take(LEGt, RIt_, axis=0, out=Lr, mode="clip")
                            fnp.take(ASt, RIt_, axis=0, out=Ar, mode="clip")
                            if Pa < n or Pb < n:
                                Ia = ord_t[n - Pa:] if Pa < n else None
                                Ib = (ordb_t[:Pb] if _SAT["tb_band"] else ord_t[n - Pb:]) if Pb < n else None
                                pool_ = [SGF, LEGTF[(T0 + 2 * ns * n) * n:(T0 + (2 * S + 2) * n) * n],
                                         swp.arena[sum(_regions(swp.lays[("tt", dtt)]).values()):]]
                                Lg = (fnp.take(Lr, Ia, axis=1, out=_sat_grab(pool_, K_ * Pa, f32).reshape((K_, Pa)),
                                               mode="clip") if Ia is not None else Lr)
                                Ag = (fnp.take(Ar, Ib, axis=1, out=_sat_grab(pool_, K_ * Pb, f32).reshape((K_, Pb)),
                                               mode="clip") if Ib is not None else Ar)
                                Yt = MQF[:Pa * Pb].reshape((Pa, Pb))
                                dd_, va_, vb_, vm_ = swp.view_shape("tt", dtt, Pa, K_, Pb)
                                _swk("tt", swp, Lg.T, Ag, Yt, va_, vb_, vm_, dd_, ip)
                                cp(M1, 0.0)
                                if Ia is None:
                                    fnp.put_along_axis(M1, Ib[None, :], Yt, axis=1)
                                elif Ib is None:
                                    fnp.put_along_axis(M1, Ia[:, None], Yt, axis=0)
                                else:
                                    Zt = MQF[n * n:n * n + Pa * n].reshape((Pa, n))
                                    cp(Zt, 0.0)
                                    fnp.put_along_axis(Zt, Ib[None, :], Yt, axis=1)
                                    fnp.put_along_axis(M1, Ia[:, None], Zt, axis=0)
                            else:
                                dd_, va_, vb_, vm_ = swp.view_shape("tt", dtt if d else 0, n, K_, n)
                                if dd_ not in qM1:
                                    qM1[dd_] = _rq(_rt(M1, dd_))
                                _swk("tt", swp, Lr.T, Ar, M1, va_, vb_, vm_, dd_, ip, qp0=qM1[dd_])
                        elif Pa < n or Pb < n:  # sat: rows (single index) by ta, columns (doubled index) by tb
                            Ia = ord_t[n - Pa:] if Pa < n else None
                            Ib = (ordb_t[:Pb] if _SAT["tb_band"] else ord_t[n - Pb:]) if Pb < n else None
                            Lr, Ar = LEGt[lo:T0 + 2 * ns * n], ASt[lo:T0 + 2 * ns * n]
                            if _SAT["mode"] == "mask":
                                dd_, va_, vb_, vm_ = swp.view(("tt", t))
                                if dd_ not in qM1:
                                    qM1[dd_] = _rq(_rt(M1, dd_))
                                _swk("tt", swp, LEGtT[:, lo:T0 + 2 * ns * n], Ar, M1, va_, vb_, vm_, dd_, ip, qp0=qM1[dd_])
                                if Ia is not None:
                                    _sat_note(t, "ta", M1, _sat_mask(Ia, n, f32)[:, None])
                                    mul(M1, _sat_mask(Ia, n, f32)[:, None], out=M1)
                                if Ib is not None:
                                    _sat_note(t, "tb", M1, _sat_mask(Ib, n, f32))
                                    mul(M1, _sat_mask(Ib, n, f32), out=M1)
                            else:
                                K_ = Lr.shape[0]  # sat fbuf: buffers dead during this step, first with room
                                pool_ = [SGF, LEGTF[(T0 + 2 * ns * n) * n:(T0 + (2 * S + 2) * n) * n],
                                         swp.arena[sum(_regions(swp.lays[("tt", dtt)]).values()):]]
                                Lg = (fnp.take(Lr, Ia, axis=1, out=_sat_grab(pool_, K_ * Pa, f32).reshape((K_, Pa)),
                                               mode="clip") if Ia is not None else Lr)
                                Ag = (fnp.take(Ar, Ib, axis=1, out=_sat_grab(pool_, K_ * Pb, f32).reshape((K_, Pb)),
                                               mode="clip") if Ib is not None else Ar)
                                Yt = MQF[:Pa * Pb].reshape((Pa, Pb))  # M[3:7] is rewritten in the matrices step
                                dd_, va_, vb_, vm_ = swp.view_shape("tt", dtt, Pa, Lr.shape[0], Pb)
                                _swk("tt", swp, Lg.T, Ag, Yt, va_, vb_, vm_, dd_, ip)
                                cp(M1, 0.0)
                                if Ia is None:
                                    fnp.put_along_axis(M1, Ib[None, :], Yt, axis=1)
                                elif Ib is None:
                                    fnp.put_along_axis(M1, Ia[:, None], Yt, axis=0)
                                else:
                                    Zt = MQF[n * n:n * n + Pa * n].reshape((Pa, n))
                                    cp(Zt, 0.0)
                                    fnp.put_along_axis(Zt, Ib[None, :], Yt, axis=1)
                                    fnp.put_along_axis(M1, Ia[:, None], Zt, axis=0)
                        elif d:
                            dd_, va_, vb_, vm_ = swp.view(("tt", t))
                            if dd_ not in qM1:
                                qM1[dd_] = _rq(_rt(M1, dd_))
                            _swk("tt", swp, LEGtT[:, lo:T0 + 2 * ns * n], ASt[lo:T0 + 2 * ns * n], M1, va_, vb_, vm_, dd_, ip,
                                qp0=qM1[dd_])
                        else:
                            fnp.matmul(LEGtT[:, lo:T0 + 2 * ns * n], ASt[lo:T0 + 2 * ns * n], out=M1)
                        fnp.add(M1, fnp.matmul(LEGtT[:, :nth], ASt[:nth]), out=M1)
                        if RIt_ is not None and not (Pa < n or Pb < n):
                            pass  # unrestricted: the product already holds the diagonal
                        elif Pa < n or Pb < n:  # sat: the exact diagonal 3 t3 of every neuron (its own K3 mean reads it)
                            fnp.fill_diagonal(M1, fnp.add(es("ri,ri->i", Lr, Ar), es("ri,ri->i", LEGt[:nth], ASt[:nth])))
                        if K4F is not None:  # r25a k4f: the diagonal stays the K3 read 3 t3
                            fnp.fill_diagonal(M1, fnp.subtract(es("ii->i", M1), mul(DGF, 4.0)))

            if QXON and t and ns:  # r25a qx: taps 0-2
                with _ns("qx"):
                    if DGF_t is not None and not bool(fnp.any(fnp.equal(LAM_t, 0.0))):  # r30b q0: tap 0 = DGF / lam
                        fnp.divide(DGF_t, LAM_t, out=QX[t, 0])
                    else:
                        H3_ = mul(mul(Ht[:ns], Ht[:ns]), Ht[:ns])
                        es("sji,sji,sj->i", Gt[:ns], H3_, W3F[:ns], out=QX[t, 0])
                    if ns > mc:
                        GHd_ = mul(Gt[mc:ns], Ht[mc:ns])
                        if _R30B.get('q12') and GF is not None and not last:  # r30b q12: G o H kept for the gf fold
                            GHL_ = mul(GHd_, LAMG[mc:ns][:, :, None])  # the taps' own einsum order (bit-equal)
                            es("sji,sji->i", GHL_, Ht[mc:ns], out=QX[t, 1])
                            es("sji,sji->i", GHL_, Gt[mc:ns], out=QX[t, 2])
                            GHL_t = GHd_
                        else:
                            es("sji,sji,sj->i", GHd_, Ht[mc:ns], LAMG[mc:ns], out=QX[t, 1])
                            es("sji,sji,sj->i", GHd_, Gt[mc:ns], LAMG[mc:ns], out=QX[t, 2])
            if QXX and t:  # r25a qx2: taps 4-7 (SV: this layer's sigma powers, made before the push)
                with _ns("qx"):
                    if QRE is not None:  # tap 6: the compression residual carried through W o W
                        es("ia,ia,i->a", W, W, QRE, out=QX[t, QXX[6]])
                    if not last and ns and (4 in QXX or 5 in QXX):
                        d1_ = es("ii->i", M1)
                        dd_ = mul(mul(d1_, d1_), mul(SV[1], SV[1]))  # (3 t3)^2 / s^6
                        if 4 in QXX:
                            fnp.subtract(mul(es("ik,ik,k->i", M1, M1, SV[0]), SV[2]), dd_, out=QX[t, QXX[4]])
                        if 5 in QXX:
                            fnp.subtract(mul(es("ik,ik,i->k", M1, M1, SV[2]), SV[0]), dd_, out=QX[t, QXX[5]])
                    if not last and 7 in QXX:
                        d0_ = mul(es("ii->i", M0), SV[2])
                        fnp.subtract(mul(es("ia,ia,a->i", M0, M0, SV[2]), SV[2]), mul(d0_, d0_), out=QX[t, QXX[7]])
                    if not last and ns and 8 in QXX:  # r25a qx3 tap 8: the signed off-diagonal T column sum
                        fnp.subtract(mul(es("ik,i->k", M1, SV[3]), SV[2]), mul(es("ii->i", M1), SV[1]), out=QX[t, QXX[8]])
                    if not last and 10 in QXX:  # r25a qx3 tap 10: the signed pre-activation correlation sum
                        fnp.subtract(mul(es("ia,a->i", M0, SV[3]), SV[3]), mul(es("ii->i", M0), SV[2]), out=QX[t, QXX[10]])
                    if t > 1 and 11 in QXX:  # r25a qx3 tap 11: the previous layer's tap 5 through W o W
                        es("ia,ia,i->a", W, W, QX[t - 1, QXX[5]], out=QX[t, QXX[11]])
            # ---- per-neuron algebra ----
            with _ns("wick_pk"):
                if not (early or pre_a):  # sat: made before the push at 0 < t < depth - 1 (r25b s2: at the last push)
                    if wm is None:
                        SV = fnp.power(var, HE)
                    else:
                        SV = SVBs[t]
                        fnp.power(var, HE, out=SV)
                if t:
                    if not (early or pre_a):
                        if wm is None:
                            a = es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig)  # VEC: the previous layer's rows
                        else:
                            a = QAs[t]
                            es("ia,ri,r,ea,e->a", W, VEC, e_pk1, SV, e_sig, out=a)
                    Mon = fnp.power(a, PEXP)
                else:  # the input mean is zero: a = 0
                    Mon = MON0
                if last:
                    FF = fnp.stack([flops.stats.norm.pdf(a), flops.stats.norm.cdf(a), ONEN, dA, LAMVs[t], LAMVs[t + 1],
                                    t3b, t3f], dtype=f32)
                    if law and t:  # dA, lam'_t, lam'_{t+1} times 1 + c1 a + c2 a^2, before every consumer
                        mul(FF[3:6], es("kp,pi->ki", LAWC, Mon), out=FF[3:6])
                    es("rcdpe,ci,di,pi,ei->ri", cVl, FF[0:3], mul(FF[2:8], LMK6), Mon, SV, out=VX[t, V_PK1:V_PK1 + 1])
                    if wm is not None:  # the corrector's copy of this layer's FF rows and t3 = t3b / 3 + t3f
                        cp(QFFs[t], FF[0:6])
                        fnp.add(mul(t3b, 1.0 / 3.0), t3f, out=QB[t, 6])
                    if lc and self._parts is not None:  # research only
                        self._parts["base15"] = fnp.copy(VX[t, V_PK1])
                    if lc:  # the corrected final mean replaces the base one
                        with _ns("lc"):
                            d15 = es("ii->i", M0)
                            soff2 = fnp.stack([fnp.subtract(es("ij,ij->i", M0, M0), mul(d15, d15)),
                                               fnp.subtract(es("ij->i", M0), d15)])
                            rich = lc_rich(t, W, var, SV, lcpre, soff2)
                            R5 = fnp.stack([ONEN, fnp.add(t3b, mul(t3f, 3.0)), FF[3], FF[4], ONEN])  # 3 t3 = t3b + 3 t3f
                            pc, _ = lc_step(t, W, a, SV, R5, dp, rich)
                            cp(VX[t, V_PK1], pc)
                    break
                if t:
                    if wm is None:
                        FF = fnp.stack([flops.stats.norm.pdf(a), flops.stats.norm.cdf(a), ONEN, dA, LAMVs[t],
                                        LAMVs[t + 1]], dtype=f32)
                    else:  # the same rows, kept for the corrector
                        FF = QFFs[t]
                        fnp.stack([flops.stats.norm.pdf(a), flops.stats.norm.cdf(a), ONEN, dA, LAMVs[t], LAMVs[t + 1]],
                                  out=FF)
                    if law:  # dA, lam'_t, lam'_{t+1} times 1 + c1 a + c2 a^2, before every consumer
                        mul(FF[3:6], es("kp,pi->ki", LAWC, Mon), out=FF[3:6])
                else:
                    FF = CR[_NP:_NP + 6]  # [phi(0), Phi(0), 1, dA = 0, lam'_0, lam'_1]
                    if wm is not None:
                        cp(QFFs[0], FF)
                VEC = VXs[t]
                # ex E5: = einsum(rkcdpe,kii,ci,di,pi,ei->ri) over the 49 monomials the table uses
                XA3 = es("kii,ci,di->kcdi", MZ, FF[0:3], mul(FF[2:6], LMK4))
                XB2 = es("pi,ei->pei", Mon, SV)
                es("rm,mi,mi->ri", XCM, XA3[XKQ, XCQ, XDQ], XB2[XPQ, XEQ], out=VEC)
                if lc and t:  # copied now: the einsum diagonal is a view and the matrices step zeroes M1's diagonal
                    lcR5 = fnp.stack([ONEN, es("ii->i", M1), FF[3], FF[4], ONEN])  # [1, 3 t3, dA, lam', 1]
                LRB_ = es("ca,cb,ai,bi->ci", cSA, cSB, VEC, VEC)  # r30b lr6: the base rows; LR per box below
                LR = None if _R30B.get('lr6') else es("yxrkc,ci->yxrki", cLR, LRB_)

            # ---- matrices: H'^T, E, C', R4 in one einsum over M; E's H'*H' part; next dA ----
            with _ns("matrices"):
                mul(M01, MJ, out=M01)  # S0 and 3T^T off-diagonal
                if QXX and 9 in QXX and t and ns:  # r25a qx3 tap 9: the off-diagonal T column energy as C' reads it
                    es("ik,ik,i->k", M1, M1, mul(mul(VW11s[t], VW11s[t]), SV[2]), out=QX[t, QXX[9]])
                    mul(QX[t, QXX[9]], SV[0], out=QX[t, QXX[9]])
                if ns:
                    cp(M2, M1T)  # 3T
                mul(M012, M0, out=M345)  # Q, S0*3T^T, S0*3T
                mul(M3, M0, out=M6)  # S0^3
                win = LEGt3[2 * ns:2 * ns + 4]  # [H'^T, E (in the new G slot), C', R4 (in the next source's slots)]
                wv = {}  # esplit: one einsum per E2 box on its (x, r, k) slices; the first box over an x range writes it
                for (x0_, x1_, r0_, r1_, k0_, k1_), MB_, first_ in zip(_EX_BOX, MBX, ESX):
                    w_ = wv.get((x0_, x1_))
                    if w_ is None:
                        w_ = wv[(x0_, x1_)] = win[x0_:x1_]
                    L_ = (es("yxrkc,ci->yxrki", cLR[:, x0_:x1_, r0_:r1_, k0_:k1_], LRB_) if LR is None
                          else LR[:, x0_:x1_, r0_:r1_, k0_:k1_])  # r30b lr6: only the rows this box reads
                    if first_:
                        es("kij,yxrki,zxrkj,y,z->xij", MB_, L_, L_, e_l, e_r, out=w_, optimize=_R25B.get("esc", "auto"))
                    else:
                        fnp.add(w_, es("kij,yxrki,zxrkj,y,z->xij", MB_, L_, L_, e_l, e_r, optimize=_R25B.get("esc", "auto")), out=w_)
                w0, w1 = BLK[2 * ns], BLK[2 * ns + 1]
                fnp.subtract(w1, es("ba,ba,rb,r->ab", w0, w0, VEC, e_3w21), out=w1)
                Wn = weights[t + 1]
                if _dtname(Wn.dtype) != _dtname(f32):
                    Wn = Wn.astype(f32)
                ones_c = LEGt3[NB - 1:2 * ns + 1:-(NB - 3 - 2 * ns)]  # [ones, C']
                dA = es("ia,ia,ij,yjk,yjk,y->a", Wn, Wn, Kmat, win[3:1:-1], ones_c, c_g)
                Cv = BLK[2 * ns + 2]
                x38_ = _X38.get('on') and t < len(_X38['eps'])
                if x38_:  # r38: the scale mode out of C', the K4 core rows (dA) and the newborn's K3 (E; feed pair below)
                    with _ns("x38"):
                        eps_ = float(_X38['eps'][t]) * float(_X38['scale'])
                        m_ = VX[t, V_PK1]
                        fnp.subtract(Cv, fnp.outer(m_, fnp.multiply(m_, eps_)), out=Cv)
                        mul(Cv, 1.0 / (1.0 + eps_), out=Cv)
                        v_ = es("ii->i", Cv)
                        # 12 eps v^2 + 4 eps v (sum v - v) + 8 eps (rowsum C'^2 - v^2) = 4 eps v sum v + 8 eps rowsum C'^2
                        dr_ = fnp.add(mul(v_, mul(fnp.sum(v_), 4.0 * eps_)), mul(es("ij,ij->i", Cv, Cv), 8.0 * eps_))
                        kr_ = fnp.add(mul(dr_, kd), mul(fnp.sum(dr_), kt))
                        dA = fnp.subtract(dA, es("ia,ia,i->a", Wn, Wn, kr_))
                        fnp.subtract(w1, fnp.outer(mul(v_, 6.0 * eps_), m_), out=w1)
                if ns:  # old legs (dense, and the bases of compressed ones), u and Z^T rows to the activation side
                    mul(thin, VW11s[t], out=thin)
                    if GF is not None and ns > mc:  # r25a gf: G += lam (w21 / w11) G o H on the dense live slots
                        with _ns("gf"):
                            rho_ = fnp.divide(VW21s[t], fnp.maximum(VW11s[t], 1e-30))
                            Gd_ = Gt[mc:ns]
                            if GHL_t is not None:  # r30b q12: G o H from the taps times (LAMG rho), the einsum's order
                                Xf_ = mul(GHL_t, mul(LAMG[mc:ns][:, :, None], rho_), out=GHL_t)
                            else:
                                Xf_ = es("sji,sji,sj,i->sji", Gd_, Ht[mc:ns], LAMG[mc:ns], rho_)
                            if QXON and _QX.get('k', 4) >= 4:  # r25a qx tap 3: s1 |v1| of the increment (rows m, columns j)
                                v_ = fnp.ones((ns - mc, n), dtype=f32)
                                for _ in range(2):
                                    u_ = es("sji,si->sj", Xf_, v_)
                                    v_ = es("sji,sj->si", Xf_, u_)
                                    v_ = fnp.divide(v_, fnp.maximum(fnp.sqrt(es("si,si->s", v_, v_)), 1e-30)[:, None])
                                u_ = es("sji,si->sj", Xf_, v_)
                                es("s,si->i", fnp.sqrt(es("sj,sj->s", u_, u_)), fnp.abs(v_), out=QX[t, 3])
                            fnp.add(Gd_, Xf_, out=Gd_)
                    if T0 + 2 * ns * n > dstart:
                        Lt = LEGt[dstart:T0 + 2 * ns * n]
                        mul(Lt, VW11s[t], out=Lt)
                    if mc:
                        mul(Qv, VW11s[t], out=Qv)
                cp(UPs[ns], VW21s[t])  # u of the new source (its g / 4 is in c)
                if x38_:  # r38: the extraction's K3 pair (-6 eps m, w11) in the feed slot (the feed is 0 at lam' = 0)
                    mul(VX[t, V_PK1], -6.0 * eps_, out=UPs[ns])
                    cp(CSRC[ns], VW11s[t])
            # ---- the new source in the 2-leg form: E = I diag(xv) + H diag(e) + Lz Z^T (rows: birth index) ----
            with _ns("twoleg"):
                es("jj,x->j", w1, c_three, out=XV3s[ns])  # 3 xv
                mul(VW21s[t], 3.0, out=C3Vs[ns])  # 3 v
                if K4F is not None:  # r25a k4f: -scale w31 of the birth layer (sign: lam = -alpha / sigma)
                    mul(VW31s[t], -K4F, out=W3Fs[ns])
                if GF is not None and t:  # r25a gf: -scale alpha / sigma of the birth layer
                    mul(mul(a, SV[3]), -GF, out=LAMGs[ns])
                rd0, rd1 = us(es("xji,ji,x->xj", win[0:2], w0, c_12))  # per birth row: [H.H, 2 E.H]
                fnp.divide(rd1, fnp.maximum(rd0, 1e-30), out=C2Es[ns])  # 2 e (least squares)
                fnp.fill_diagonal(w1, 0.0)
                fnp.subtract(w1, es("ji,j,x->ji", w0, C2Es[ns], c_half), out=w1)  # residual R = E_off - e H
                Qy = fnp.linalg.qr(fnp.matmul(w1, Wn[:, :kq]))[0]  # range of R; test matrix: next layer's weights
                Ub, sb, Vtb = fnp.linalg.svd(fnp.matmul(Qy.T, w1), full_matrices=False, k=kz)
                fnp.matmul(Qy, Ub, out=LZs[ns])
                es("c,ci->ci", sb, Vtb, out=ZTs[ns])
                if dc_on:  # dc: this source's centers and their rows in LEGt
                    R_ = _sat_size('alive_' + _DC['theta'], t, n, max(1 << d, 32)) if t > 0 else n
                    if R_ < n:
                        ki_ = fnp.sort(fnp.argsort(a)[n - R_:]).astype(fnp.int32)
                        km_ = fnp.zeros((n,), dtype=f32)
                        fnp.put_along_axis(km_, ki_, fnp.ones((R_,), dtype=f32), axis=0)
                        DCK[ns] = km_
                        mul(w0, km_[:, None], out=w0)  # H^T rows
                        mul(C3V2E[ns], km_[None, :], out=C3V2E[ns])  # 3 v, 2 e
                        mul(XV3s[ns], km_, out=XV3s[ns])  # 3 xv
                        mul(LZs[ns], km_[:, None], out=LZs[ns])  # Lz rows
                        DCH[ns] = fnp.add(ki_, T0 + 2 * ns * n)
                        DCG[ns] = fnp.add(ki_, T0 + (2 * ns + 1) * n)
                        R2_ = -(-R_ // (1 << (d + 1))) * (1 << (d + 1))
                        DCHP[ns] = DCH[ns] if R2_ == R_ else fnp.add(
                            fnp.sort(fnp.argsort(a)[n - R2_:]).astype(fnp.int32), T0 + 2 * ns * n)
                    else:
                        DCH[ns] = fnp.arange(T0 + 2 * ns * n, T0 + (2 * ns + 1) * n, dtype=fnp.int32)
                        DCG[ns] = fnp.arange(T0 + (2 * ns + 1) * n, T0 + (2 * ns + 2) * n, dtype=fnp.int32)
                        DCHP[ns] = DCH[ns]
            ns += 1
            ord_p, dyn_a_prev = (ord_t, (a if t else None)) if early else (None, None)  # sat: the next push's inner set
            if QRE is not None and t:  # r25a qx2 tap 6: the carried energy to the activation side (legs x w11)
                mul(QX[t, QXX[6]], mul(VW11s[t], VW11s[t]), out=QRE)
            # compress every source that is now `age` layers old: rank-k basis of its normalized [H, G] legs
            if Smc:
                with _ns("compress"):
                    kc_pin = n  # r25b kc: the inner size the next push reads the bases with
                    if _R25B.get('kc'):
                        kc_pin = _r25_kc_pin(t, n, d, CF, hp, cin_p, ord_p, dc_g and any(
                            s_ in DCK for s_ in range(min(Smc, max(mc, t - age + 1)), ns)))
                    while mc < Smc and mc <= t - age:
                        X4 = LEG4s[mc]
                        if _WC is None:
                            nrm = fnp.sqrt(fnp.maximum(es("xji,xji->x", X4, X4), 1e-30))
                            es("xji,x->xji", X4, fnp.divide(1.0, nrm), out=SCR3)
                            Uc, sv, Vt = fnp.linalg.svd(SCR, full_matrices=False, k=k)
                            cp(QtS3s[mc], Vt)
                            if QRE is not None:  # r25a qx2 tap 6: the column energy this compression drops
                                fnp.add(QRE, fnp.maximum(fnp.subtract(es("ri,ri->i", SCR, SCR),
                                                                      es("c,ci,ci->i", mul(sv, sv), Vt, Vt)), 0.0), out=QRE)
                            es("xjc,c,x->xjc", Uc.reshape((2, n, k)), sv, nrm, out=CtS4s[mc])
                        else:  # r24 wc: rows weighted by (birth atom's T mass / row norm)^g
                            A4 = ASt4[mc]  # this layer's [AH, AG] rows of the source (its 3 T^T read them)
                            rl = fnp.sqrt(fnp.maximum(es("xji,xji->xj", X4, X4), 1e-30))
                            tm = es("xj,xj->j", rl, fnp.sqrt(es("xji,xji->xj", A4, A4)))
                            wr = fnp.divide(tm[None, :], rl)
                            wr = fnp.power(fnp.divide(wr, fnp.maximum(fnp.max(wr), 1e-30)), _WC['g'])
                            wr = fnp.maximum(wr, _WC['floor'])
                            es("xji,xj->xji", X4, wr, out=SCR3)
                            ki_ = (fnp.subtract(DCH[mc], T0 + 2 * mc * n) if (_R25B.get('kc') and mc in DCK)
                                   else None)  # r25b kc: the centers dc kept (rows of both legs)
                            if ki_ is None and kc_pin >= n:
                                Uc, sv, Vt = fnp.linalg.svd(SCR, full_matrices=False, k=k)
                                cp(QtS3s[mc], Vt)
                                if QRE is not None:  # r25a qx2 tap 6: the column energy this compression drops
                                    fnp.add(QRE, fnp.maximum(fnp.subtract(es("ri,ri->i", SCR, SCR),
                                                                          es("c,ci,ci->i", mul(sv, sv), Vt, Vt)), 0.0), out=QRE)
                                es("xjc,c,xj->xjc", Uc.reshape((2, n, k)), sv, fnp.divide(1.0, wr), out=CtS4s[mc])
                            else:  # r25b kc: the SVD on the kept rows and the columns the next push reads
                                _r25_kc(SCR, wr, k, n, ki_, ord_p[n - kc_pin:] if kc_pin < n else None,
                                        ord_p[:n - kc_pin] if kc_pin < n else None, QRE, QtS3s[mc], CtS4s[mc],
                                        _R25B['kc'] == 'B')
                        mc += 1
            if lc and t:  # this layer's corrected mean, carried into the next layer's readout only
                with _ns("lc"):
                    rich = None
                    if t in _LC_RICH:
                        soff2 = fnp.stack([es("ij->i", M3), es("ij->i", M0)])  # S0 of this layer: rows of S0^2, S0
                        rich = lc_rich(t, W, var, SV, lcpre, soff2)
                    _, dp = lc_step(t, W, a, SV, lcR5, dp, rich)
        if wm is not None:
            _wmsg(weights, VX, QB, SVB, wm, f32, self._parts, QX if QXON else None)
        return VX[:, V_PK1, :]


class Estimator(BaseEstimator):
    """Law-based augmented K=3 cumulant propagation at (n, depth) = (1024, 16); k3_fast's SIMPLE engine otherwise."""

    def __init__(self, dtype=None, force_aug: bool = False, lam=None, age=6, rank=256, sw_depth=5,
                 sw_depth_batched=5, kz=4, kq=32, lc: bool = True, law: bool = True, sw_inplace: bool = True,
                 sw_hi_push=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15), sw_hi_tt=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), sw_hi_batched=(), sw_setup_plan: bool = True,
                 gru: bool = True, gru_law: bool = False, gru_stack: bool = False) -> None:
        self._dtype = dtype
        self._force_aug = force_aug  # tests only: run the AUG engine at any shape (depth >= 2)
        self._lam = lam
        # cost machinery (ours, as estimators/k3_fast_sc2.py): compress sources `age` layers old to rank `rank`
        # (age None: never); Strassen-Winograd levels for the push / 3 T^T and for the per-source batched products
        self._cost = dict(age=age, rank=rank, sw_depth=sw_depth, sw_depth_batched=sw_depth_batched, kz=kz, kq=kq,
                          sw_inplace=sw_inplace, sw_hi_push=tuple(sw_hi_push), sw_hi_tt=tuple(sw_hi_tt),
                          sw_hi_batched=tuple(sw_hi_batched))
        self._lc = lc  # the learned mean correction (scored shape only)
        self._law = law  # the per-neuron K4 law (scored shape only)
        self._lcc = None  # its constant arrays, built once in setup()
        self._swp = None  # the scored shape's Strassen-Winograd frames and views, built once in setup()
        self._setup_plan = bool(sw_setup_plan)  # False: predict makes each call's views (no workspace kept between MLPs)
        self._gru = bool(gru)  # the W-message recurrent corrector (scored shape only); it replaces lc's final mean
        self._gru_law = bool(gru_law)  # research only: keep the per-neuron K4 law under the corrector
        self._gru_stack = bool(gru_stack)  # research only: keep lc's corrected mean (and the law) and add the corrector
        self._wm = None  # its arrays, loaded once in setup()
        self._parts = None  # research only: set to a dict to receive intermediate final means

    def setup(self, ctx: SetupContext) -> None:
        # sat: warm the ops the restriction adds (sort, search, read, scatter) on tiny arrays, once per worker
        w_ = fnp.array([0.5, -1.0, 2.0, 1.0], dtype=fnp.float32)
        o_ = fnp.argsort(w_)
        fnp.searchsorted(fnp.take(w_, o_), fnp.array([0.0, 1.5], dtype=fnp.float32)).tolist()
        z_ = fnp.zeros((2, 4), dtype=fnp.float32)
        fnp.put_along_axis(z_, o_[None, 2:], fnp.ones((2, 2), dtype=fnp.float32), axis=1)
        fnp.fill_diagonal(z_, fnp.abs(w_)[:2])
        # load the correction's constant tables once per worker (data only; off the FLOP budget and the residual)
        lc_used = self._lc and (not self._gru or self._gru_stack)
        self._lcc = _lc_consts(fnp.float32, _AUG_SHAPE[0]) if (lc_used and self._dtype is None) else None
        # the corrector's arrays (data next to this file; 0 FLOPs, off the residual)
        self._wm = None
        if self._gru and self._dtype is None:
            d = getattr(ctx, "submission_dir", None) or os.path.dirname(os.path.abspath(__file__))
            z = fnp.load(str(os.path.join(d, "wmsg.npz")))
            self._wm = {k: z[k] for k in _WM_KEYS + tuple(k for k in ("wqx", "qxc") if k in getattr(z, "files", z))}
        # the work arena of the scored shape, its frames and the views every Strassen-Winograd call of a predict
        # indexes (data structures; predict carves the frames again so each MLP is billed as before)
        self._swp = None
        if self._setup_plan and self._dtype is None:
            c = self._cost
            self._swp = _SwPlan(fnp.float32, _AUG_SHAPE[0], _AUG_SHAPE[1], c["age"], c["rank"], c["sw_depth"],
                                c["sw_depth_batched"], frozenset(int(t) for t in c["sw_hi_push"]),
                                frozenset(int(t) for t in c["sw_hi_tt"]),
                                frozenset(int(t) for t in c["sw_hi_batched"]), bool(c["sw_inplace"]), views=True,
                                cf=_covfold_layers(_AUG_SHAPE[0], _AUG_SHAPE[1], c["sw_depth"],
                                                   frozenset(int(t) for t in c["sw_hi_tt"])))  # the predict's plan key
            if _DC['theta'] is not None and _DC['gather'] and c["sw_depth"] > 0:  # dcfix: the gather buffers, touched now
                n_, S_, a_, k_ = _AUG_SHAPE[0], _AUG_SHAPE[1] - 1, c["age"], c["rank"]
                Smc_ = max(0, S_ - a_ - _R25B.get('s3', 0)) if (a_ is not None and a_ >= 1 and 0 < k_ < n_) else 0
                rows_ = (Smc_ * k_ if Smc_ else 0) + (2 * min(S_, a_ if a_ else S_) + 1 + 2 * _R25B.get('s3', 0)) * n_
                self._swp.dcx = fnp.zeros((2, rows_, n_), dtype=fnp.float32)
        return None

    def teardown(self) -> None:
        self._swp = None  # release the setup() workspace (its handles are freed with the next request)
        try:  # collect now so the client queues every handle of the plan for release
            import gc
            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        return None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return self.propagate(mlp.weights)

    def propagate(self, weights):
        shape = (weights[0].shape[0], len(weights))
        if (self._force_aug and shape[1] >= 2) or shape == _AUG_SHAPE:
            lcc, swp = (self._lcc, self._swp) if shape == _AUG_SHAPE else (None, None)
            return _AugEngine(self._dtype, self._lam, **self._cost, lc=self._lc, lcc=lcc,
                              law=self._law and (not self._gru or self._gru_law or self._gru_stack), swp=swp,
                              wm=self._wm, wm_stack=self._gru_stack,
                              parts=self._parts).propagate(weights)
        return _SimpleEngine(self._dtype).propagate(weights)
