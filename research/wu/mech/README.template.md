# The 3 T^T band: what the dropped third-cumulant entries are, and what they do (research/wu/mech)

Question (coordinator, 10-21): in the submitted engine (estimators/r38/r38_sub_x100_s2/estimator.py, submission 337488)
a restriction called the 3 T^T band skips part of a third-cumulant term; turning it off (make_variants.py
v_engine_nott) was measured worse. Are the dropped entries wrong against the truth (biased, sign-flipped), and how much
of the final-layer mean error do they account for?

Follow-up (coordinator, 10-22): the write-up describes submission 336955, the opr30 file (estimators/wu2/v_engine and
v_engine_nott are its cuts, the files behind kp_table.json), where the band gains much more. (1) Replicate the entry
statistics and the oracle there. (2) Find which change between opr30 and r38 removed the gain. Sections 8 and 9
answer it; the conclusion below covers both files, the short answer after it is the 10-21 one for r38.

Networks: public minis 0-3 with Monte Carlo truth (the r38 file: minis 0 and 1 for everything, 2 and 3 for the oracle
swap; the opr30 file and the r38 variants: minis 0-3); minis 0-15 for band on / off without truth (0-7 for the
extraction's parts). Training pool only: no held-out row and no corpus network was read. Line numbers are in
estimators/r38/r38_sub_x100_s2/estimator.py (sha256 0d69dc7d...) unless stated. Runs: r38 10-21 and 10-22 06:31-09:35
KST (after the 10-21 20:33 pause); opr30, r37 and the r38 variants 10-22 09:45-12:01 KST.

## Conclusion (both files)

1. **opr30: the dropped entries are right, neither sign-flipped nor biased.** At layers 2-10 they agree with the Monte
   Carlo truth as well as the kept entries do (median relative error 2-5%, sign agreement 1.00, corr 1.00, slope
   0.97-1.00, mean signed bias 0.00-0.01 of the mean absolute truth; minis 0-3). At layers 11-14 they are shrunk to
   0.52-0.63 of the truth by the s8 push restriction (no legs for neurons with a < -3), sign agreement 0.92-0.95. The
   rest of opr30's third cumulant is close to exact too (kept slope 0.94-1.00). r38's is not: every entry is shrunk,
   more with depth (kept slope 0.92 at t = 2 to 0.47 at t = 14 on average over minis 0-3, dropped 0.86 to 0.13). The
   shrinkage comes with the extraction, not with the law: r38 with the extraction undone (law still off) has opr30's
   slopes (kept 0.93-1.00 at every layer), r38 with the law back and the extraction on has r38's. Nor is it the
   input-norm scale mode taken out on purpose: against the Monte Carlo truth with the inputs put on the sphere (mini
   0), opr30's kept-entry slopes are 1.01-1.10, so the scale mode is up to about 10% of the third cumulant, while
   r38's fall from 1.02 at layer 2 to 0.54 at layer 14.
2. **The opr30 band gains by removing correct entries.** Zeroing the dropped entries (z) gives x0.871-0.938 of the
   band-off MSE, the submitted band x0.880-0.920; the truth on them (m) x1.002-1.015. The gain is in the entries
   dropped for their doubled index (|a_j| >= 3; zeroing only those x0.840-0.937, only the row drops x1.10-1.19),
   mostly at layers 2-10, and it stays when every other entry is exact (x0.83-0.91 with the truth on the kept entries,
   x0.85-1.00 with the truth everywhere). The truth on every off-diagonal entry is about neutral (dmm x0.85-1.08), the
   true diagonal helps (dd x0.72-0.81), and both together help more (ddmm x0.64-0.78): unlike r38 (x22-42), the opr30
   engine is not hurt by exact third cumulants.
3. **The scale-mode extraction took the gain.** Band on / off over public minis 0-15: opr30 x0.918 and r37 x0.919
   (band on better on 16 of 16 each), so R36 (rank 256) and L4 left the gain alone; r38 without the extraction (law
   off as submitted) x0.894 (16 of 16); r38 x0.983 (11 of 16); r38 with the law back and the extraction on x1.062
   (better on 0 of 16). The band helps in every engine without R38's extraction and in none with it, law on or off. Of
   the extraction's four parts it is mainly the one that takes the scale mode out of the next layer's K4 core rows
   (dA): with only that part off the band gains again (x0.915 on minis 0-7, better on 8 of 8, in an engine six times
   worse than r37), with only the feed pair off a little (x0.964, in an engine nearly as good as r38), with the C' or
   the E part off not at all (x0.992, x1.030).
4. **Why: the band's change stayed, the error it met moved.** The band changes the final means in nearly the same way
   in opr30 and r38 (cosine 0.87-0.90 on minis 0-3, a quarter of the band-off error's rms in both). In opr30 that
   change points toward the error (cosine 0.28-0.37), in r38 hardly (0.10-0.17); moved onto the other file's band-off
   means, either file's band change gains on opr30 (x0.88-0.94) and not on r38 (x0.95-1.03). Of the opr30 band's gross
   gain, 44-68% lies along what r38 does better than opr30 (the band's change has cosine 0.49-0.50 with it). In the
   engine without only the extraction's dA part, 87-92% of the band's gross gain lies along that part's own change,
   and every file's band change gains there as on opr30: the band offsets part of the error the engine makes while the
   next layer's K4 core rows carry the input-norm scale mode's share, and the extraction's dA part removes that error
   itself. The gain sits in the entries dropped for their doubled index in every file: zeroing only those gains
   x0.84-0.87 without the dA part, x0.84-0.94 in opr30, x0.93-1.01 in r38. Across all six engines measured the band's
   change is alike (cosine 0.83-0.94), and whichever file's change is used it gains on the band-off means of every
   engine whose K4 core rows keep the scale mode's share (x0.88-0.93) and on none whose rows do not (x0.99-1.00 on
   r38, x1.07-1.10 on r38 with the law back). The accuracy of the dropped entries is not what decides: they are right
   in opr30 and shrunk in r38, and the band's change is nearly the same in both.

## Answer in short (r38 file, submission 337488; 10-21)

1. **Not sign-flipped, shrunk like the rest.** At layers 2-10 the dropped entries agree with the Monte Carlo truth as
   well as the kept ones (minis 0-3): correlation 0.92-0.99 (kept 0.91-0.99), sign agreement 0.87-1.00 (kept
   0.89-1.00). Every off-diagonal entry of the engine's third cumulant is shrunk toward zero, the dropped ones a little
   more (engine / truth slope 0.47-0.87, kept 0.55-0.93, falling with depth). At layers 11-14 the band-off engine does
   not compute most dropped entries either (slope 0.11-0.23): the push of those layers (R25-B s8) writes no legs for
   neurons with a < -3, so their rows and columns of 3 T^T hold only the thin rows' share, band on or off.
2. **Small weight by construction, a few percent of the final MSE.** In C' a 3 T^T entry is weighted by w11 of its
   single index and w21 of its doubled index (line 1367): below Phi(-1.5) = 0.067 for a dropped row, below
   phi(3) / sigma = 0.0044 / sigma for a dropped column. Zeroing them (the clean band, all else as band off) moves the
   final-layer MSE by
   Z_RANGE;
   the submitted band (with s7 and the thin rows) by
   P_RANGE;
   over minis 0-15 the submitted band's on / off ratio is
   EXT_R38
   (the opr30 file's: x0.918, 16 of 16).
3. **Exact values make it worse.** The dropped entries set to their Monte Carlo truth: x1.156, x1.148, x1.182, x1.165 of
   the band-off MSE (minis 0-3). One layer at a time the truth there moves the final MSE by -1.6% to +3.5%, not
   additively: on mini 1 the truth at any single layer but one helps a little (x0.984-0.998; layer 12 x1.001), at all
   layers together it costs x1.12 of the clean band.
4. **Why: the engine is calibrated at its own third cumulants.** Every off-diagonal entry scaled by 0.9 costs x1.8-1.9,
   by 0.75 x7.0-7.4, by 1.25 x9.9-10.6, by 0 x82-90; every off-diagonal entry set to the truth x22-24 (that truth scaled
   to 0.75: x7.1-7.7); the true diagonal t3 alone x27-42. At one layer, the truth makes that layer's C' and the next layer's variance and t3
   more accurate, yet the next layer's mean MSE grows x1.3-2.9: 87-93% of the change comes through the next sigma, in a
   direction orthogonal to the remaining mean error. The engine's means are accurate because its variance error's
   mean shift (1.6-2.4e-4 rms) is cancelled by its other terms (correlation -0.75 to -0.80, leaving 5e-5 to 1.4e-4);
   any change that does not respect that cancellation adds in quadrature, exact or not. The band removes a small-weight
   slice of T inside this calibrated system; its sign of effect is network-dependent, and exactness is not the issue.

## 1. The band, exactly

**The slice.** `M1` is the second slot of the stack `M = [S0, 3T^T, 3T, Q, S0*3T^T, S0*3T, S0^3, J, I]` (line 2746),
the product of the source legs on the preactivation side of layer t with their coefficient rows, `M1 = L^T A`
(lines 3240-3324), plus the thin rows' product u r^T + Z AZ (line 3325). Off the diagonal

    M1[i, j] = 3 kappa3(z_i, z_j, z_j) + F[i, j],      z = z_t = h_{t-1} @ W_t (preactivation)

with row i the single index and column j the doubled index (line 3281: "rows (single index) by ta, columns (doubled
index) by tb"). F is the K4 star (3,1) fold (r25a k4f, lines 3208-3217), 2-6% of the K3 part's rms here. The diagonal
is 3 t3 = 3 kappa3(z_i, z_i, z_i) (lines 3326-3331). In the readable engine (research/r38/k3_acc_tc.py lines 200-242,
309) T[i, j] = kappa3(z_i, z_i, z_j) and M1 = 3 T^T. Verified: (M1 - F) / 3 correlates 0.85-0.99 with Monte Carlo
kappa3(z_i, z_j, z_j) on the kept entries of layers 2-14 (minis 0-3) and -0.01 to 0.02 with its transpose. (Our notes' working
description holds for the thresholds and for rows / columns only if T_ij is read as M1 / 3 = kappa3(z_i, z_j, z_j), the
single index in the row; for kappa3(z_i, z_i, z_j) the roles are swapped. And an entry is dropped when either condition
holds, not both; see the rule.)

**Where it enters.** The diagonal feeds the per-neuron algebra (MZ, line 2757; XA3, line 3425): pK1 = w01 + t3 w31 / 6
+ d4 w41 / 24. The off-diagonal is read only by the matrices step (lines 3433-3454): the diagonals of M0 and M1 are
zeroed (line 3435), M2 = M1^T (line 3440), M345 = [Q, S0*3T^T, S0*3T] (line 3441), and the box einsums (line 3452)
write the window [H'^T, E, C', R4]: the newborn source's legs, the activation covariance C' and the K4 rows. The
diagrams: `_diagrams` lines 1359-1378 (K_M1 with coefficient 1/6 is the covariance diagram T^T with 1/2 and weights
w_1p of the row and w_2q of the column, line 1367; K_M2 line 1368; S0*T^T lines 1376-1377), `_hprime_terms` (H' gets
(w21_i / 2) T_im and rho_m w11_i T_mi, lines 1385, 1387), `_window_terms` (E gets -w11_a^2 w11_b T, line 1414;
C' = P11, line 1430; R4 = K422 rows, line 1441).

**The rule.** `_SAT = {'in': 1.75, 'out': 4.0, 'ta': 1.5, 'tb': 3.0, 'tb_band': True, 'mode': 'bins', 'q': 32}`
(line 1606). At each layer 0 < t < 15 (lines 2963-3011), with a = mu / sigma the engine's own standardized
preactivation mean at that layer:

* counts from the sorted a (lines 2984-2989): c_ta = #{a >= -1.5}, c_tb = #{-3 <= a < 3};
* Pa, Pb = c_ta, c_tb rounded up to a multiple of 64 (`_sat_size`, lines 1759-1763; mult = max(2^dtt, 32), dtt = 6
  because sw_hi_tt = (1, ..., 14), lines 2981, 3601), lines 2999-3000;
* if Pa < 1024 or Pb < 1024: the cheapest pair of that layer's `_SAT_BINS["tt"]` bins (line 1610; fractions of n
  rounded up to 64 by `_sat_pad` / `_sat_bins`, lines 1615-1631) with Pa' >= Pa and Pb' >= Pb; if none covers, (1024,
  1024): no restriction (line 3005);
* kept rows Ia = the Pa neurons with the largest a (`ord_t[n - Pa:]`), kept columns Ib = the Pb neurons with the
  smallest |a| (`ordb_t[:Pb]`), lines 3009-3011, 3255-3256, 3282-3283.

The product then runs only on Ia x Ib (`_swk("tt", ...)`, lines 3265, 3306) and M1 is zero elsewhere (lines 3266-3275,
3307-3316) except the thin rows' product, added to every entry (line 3325; on minis 0 and 1 what it leaves on the
dropped entries is 11-21% of their band-off rms), and the diagonal, recomputed exactly for every neuron (line 3329).
The dropped set of layer t:

    D_t = {(i, j), i != j : i not in Ia  OR  j not in Ib}

every entry whose single index is among the 1024 - Pa lowest a (all a < -1.5) or whose doubled index is among the
1024 - Pb largest |a| (all |a| >= 3); the complement of a product set. Layers: t = 2..14 (the tt bins are empty at
t = 1, no source exists at t = 0, there is no 3 T^T at t = 15). On mini 0 the band acted at t = 3..14 (at t = 2 no bin
covered c_ta), on mini 1 at t = 2..14; at t <= 5 only rows are dropped (Pb = 1024). |D_t| grows from 65,472 (6% of the
off-diagonal) to 552,484 (53%).

**R25-B s7** (`_R25B['s7'] = 1`, line 1636; lines 3017-3019, 3169-3175, 3220-3230) belongs to the band: on layers with
an active column band and compressed sources (t >= 7 here) the compressed sources' legs are rebuilt and their blocks P
made on the band columns only, and their thin coefficient rows are zeroed off the band. v_engine_nott removes the tt
bins and s7.

## 2. Method

* Truth (mc_truth.py): 10^7 inputs per network as the organizers' sampler draws them (whestbench simulation.py: x ~
  N(0, I), float32 forward h_t = relu(h_{t-1} @ W_t), float64 accumulators), 200 batches of 50,000; per batch the
  unbiased k-statistic B / ((B - 1)(B - 2)) sum (z_i - zbar_i)(z_j - zbar_j)^2 with the batch's own means; estimate =
  batch mean, standard error = batch-means SE. Check: the Monte Carlo activation means match the dataset's 10^9-sample
  means at the expected noise (final layer MSE 8.7e-9 vs avg_variance / 10^7 = 7.9e-9 on mini 0). The truth's rms is
  32-100 times its standard error; 82-99% of entries are resolved (|TT_mc| > 3 SE). mc_cov.py: the activation
  covariance the same way (for the local test).
* Engine (make_mech.py): v_engine (band on), v_engine_nott (band off) and v_engine_nos7 cut from the r38 file with
  make_variants.py's own replacement lists, and the same files with three hooks (mech_*): a module global `_MECH`,
  None by default. The hooked band-off copy reproduces v_engine_nott bit for bit (pred sha16 f45127d45e8c2ab8 /
  3363c377e60c736c, same FLOPs), the hooked band-on copy v_engine. The hooks record, per layer, M1, the fold share F,
  the exact product of the same operands in float64 (Mx), the band sets the submitted rule picks at that layer's a,
  and C'; mech_run.py --spec / --override replace entries of M1 before anything reads them.

## 3. Are the dropped entries wrong? (per layer, kept vs dropped)

TT[i, j] = kappa3(z_i, z_j, z_j). Engine value = (Mx - F) / 3 of the band-off engine (Mx: the exact product; the
engine's own APA-level product differs by 0.5-0.9% rms and changes none of these numbers materially; results.json
has both). res: share of entries resolved (|TT_mc| > 3 SE); med: median |eng - mc| / |mc| over resolved; sign: sign
agreement over resolved; corr: Pearson over the group; slope: E[eng mc] / E[mc^2], noise-corrected (the engine's
scale relative to the truth); rows-only / cols-only: dropped because of the row (a_i < -1.5) or the column (|a_j| >= 3)
alone.

TABLE_ENTRIES

* Layers 2-10: dropped entries look like kept ones. The signed bias is the shrinkage of a signed quantity: the truth of
  an entry with a dead single index (a_i <= -1.5) is negative for essentially every j (mean / mean abs -0.90 to -1.00 on
  minis 0 and 1), so the dropped rows show a positive bias (+0.16 to +0.58 of the mean abs truth; +0.76 to +0.92 at
  t >= 11) and the kept entries (mostly positive truth, +0.44 to +0.74) a negative one (-0.002 to -0.49).
* Layers 11-14 (minis 0 and 1): split by the engine's a, rows with a < -3 have slope 0.04-0.14 and correlation 0.06-0.20 (rows with
  -3 <= a <= -1.5: 0.50-0.61 and 0.93-0.98; the a < -3 rows at t = 6-10: 0.49-0.63), columns with a <= -3 slope
  0.06-0.19. The push of layers 11-14 writes only the output columns of the top-Po neurons by a, Po >= #{a >= -3}
  rounded up to the s8 pairs (R25-B s8, lines 2995, 3099-3110), so the other neurons' legs are zero there.
* The fold share is 2-6% of the truth's rms; the Monte Carlo standard error 1-3%.

Diagonal t3 = kappa3(z_i, z_i, z_i), slope / corr against the truth, neurons inside vs outside the column band:

TABLE_DIAG

The diagonal is shrunk the same way. The submitted file (band on, with s7) is worse than band off and than v_engine_nos7
on the out-of-band neurons' t3 from t = 7 on: s7 leaves the compressed sources' share out of those neurons' t3 and
subtracts a fold share computed from leg columns it did not rebuild. Its effect on the final MSE is small and of either
sign (v_engine / v_engine_nos7: x1.007 on mini 0, x0.989 on mini 1).

## 4. How much of the final-layer error do they account for? (oracle swap, all band layers at once)

Band-off engine with entries of M1 replaced at every layer where the band acts (mech_run.py specs; B = v_engine_nott,
dkz = the clean band: dropped entries 0, kept entries and diagonal as the band-on code makes them, exactly). Rows s*,
k075, mm*, dd, ddmm are the scale test of section 6.

TABLE_SWAP

* (a) band on: production 0.973 / 1.016 of B; the clean band (z) 0.950 / 1.025. (b) band off: 1. (c) dropped entries
  = Monte Carlo truth (m): 1.156 / 1.148.
* Controls: doubling the Monte Carlo noise (dkmn vs dkm) +0.2-0.3%; the fold share on the dropped entries (mnf vs m,
  dkmnf vs dkm) 0.2-0.6%; the APA level and the exact diagonal (d, dkx, xall vs B) at most 1.3%.

The same on mini 2 and 3:

TABLE_SWAP23

## 5. One layer at a time

Base: the clean band at every band layer. At one layer L the dropped entries take the engine's exact values (x) or
the truth (m), or every off-diagonal entry takes the truth (t); the state up to layer L is identical, so the change of
C'_L and of layer L + 1 is local.

TABLE_LAYERS

TABLE_LOCAL

Where the next layer's mean moves (mech_decomp.py, truth on every entry of layer L, L = 2..13): the sigma part
phi(a) d sigma is 87-93% of the change's rms, the t3 part 6-11%, the rest 12-28%; the sigma part's best multiple toward
fixing the existing error is -0.08 to 0.14 (orthogonal). For the dropped entries alone (x, m) the change of C'_L is tiny
(relative error unchanged to 3-4 digits, cosine with the needed correction 0.003-0.17) and of the right direction
(best multiple 0.63-0.91 at L <= 5, 1.8-106 later), and t3 at L + 1 does not move. The full truth's change has a best
multiple of 2.0-2.3 at every layer: the right direction at about half the needed size.

Why a more accurate C' makes the means worse: the engine's next-layer variance is too low by 0.1-0.2% (layer 3) to
4.4-4.7% (layer 15) on average (0.5-7% rms; scale_check.py, 10^5 samples, agrees with mc_truth.py); a tenth of that is
the input-norm scale mode (sphere-normalized inputs: -0.1% to -4.2%) and conditioning on each sample's |h_{t-1}|
overshoots (+0.2% to +4.2%), so it is not the r38 scale-mode extraction alone. That variance error alone would move
the means by phi(a) dsigma = 1.6e-4 (layer 4) to 2.4e-4 (layer 13) rms, yet the engine's mean error is 5.0e-5 to
1.35e-4: the rest of its mean error correlates -0.75 to -0.80 with the sigma part and cancels most of it (the mean
error itself correlates +0.02 to +0.12 with it). Any change of C' that the cancelling terms do not follow adds its
mean effect in quadrature.

## 6. Scale test: the engine is calibrated at its own third cumulants

Band off, every off-diagonal entry of 3 T^T at layers 1-14 multiplied by s: final MSE / B = SCALE_LINE. A quadratic
bowl centred near s = 1 (the engine's own scale; the truth is 1.1-2 times larger). The truth on every off-diagonal entry
costs x22-24, scaled to 0.75 or 0.5 still x7 or x10-13, so the pattern matters, not only the size. The Monte Carlo
diagonal t3 alone (dd) costs DD_LINE. The engine's mean equations work with its own t3 and T. (The 10-21 version
of this note guessed that the K4 law's fitted lam' table carries this calibration. It cannot in r38: R38 multiplies
the lam' rows by zero wherever the engine reads them (LMK4, LMK6, lines 2710-2711; section 9). The opr30 engine has
the same bowl around its own scale (section 8), and there its own scale is the truth's.)

## 7. Band on / off across public minis 0-15 (no truth needed)

TABLE_EXT

On the same sixteen public networks the band gains much more in the opr30 file, whose cuts kp_table.json measured
(band on / off geometric mean x0.918, band on better on 16 of 16, like the corpus's x1.101 the other way round), than
in the submitted r38 file (x0.983, 11 of 16; range x0.927-1.024). The r38 engine itself is 2-18% better than the opr30
engine on every one of these networks. Section 9 finds which change took the gain.

## 8. The opr30 file (submission 336955), where the band gains x0.918

**Files.** make_cuts.py applies make_mech.py's first three hooks to estimators/wu2/v_engine and v_engine_nott, the
make_variants.py cuts of the opr30 file (VARIANTS.json base ..._opr30, the files kp_table.json measured):
est_opr30/mech_prod (band on) and mech_nott (band off). The fourth hook (C') has its anchor in r38's extraction block,
so there is no C' recording and no local test for opr30. With the identity spec both copies reproduce the cuts bit for
bit on minis 0-3: the same float64 predictions (sha16 3e1727c41f20155e, 40f13f50fa1c1c01, f88aa8c47a774e51,
73af83b6eeac425c band off; 5c3dd2cde9da7323, a41887e497977bf7, 477289afe5710574, eeb091395a2bd09b band on) and final
MSE as the unmodified cuts (runs/opr30_rec.jsonl against runs/ext_opr30_v_engine{,_nott}.jsonl). The band rule is the
one of section 1: the `_SAT` and `_SAT_BINS` lines are the same and so are s7 and s8 (opr30 to r38 changes R36, L4
and R38 only, section 9). The Monte Carlo truth of section 2 serves every file: it belongs to the network.

**Kept vs dropped, per layer** (exact engine values, fold removed, against the Monte Carlo truth; means over minis
0-3; med: median relative error over resolved entries; sign: sign agreement over resolved; corr: Pearson; slope:
E[eng mc] / E[mc^2], noise-corrected; bias: mean signed error over the mean absolute truth):

TABLE_OPR30_LAYERS

The same for the r38 file (section 3's runs), for contrast:

TABLE_R38_LAYERS

* The opr30 engine's third cumulant is close to exact. Kept entries at every layer: median relative error 4-6%, sign
  agreement 0.98-1.00, corr 0.97-1.00, slope 0.94-1.00, mean signed bias -0.03 to 0.00 of the mean absolute truth
  (r38: slope 0.92 at t = 2 falling to 0.47 at t = 14, means over minis 0-3).
* Layers 2-10: the dropped entries are as exact as the kept ones: median relative error 2-5%, sign agreement 1.00,
  corr 1.00, slope 0.97-1.00, bias 0.00 to +0.01, for entries dropped because of the row (a_i < -1.5; slope
  0.98-1.00) and because of the column (|a_j| >= 3; 0.95-0.98) alike. Not sign-flipped and not biased.
* Layers 11-14: shrunk to 0.52-0.63 of the truth (rows only 0.64-0.73, columns only 0.52-0.65, both 0.33-0.47), sign
  agreement 0.92-0.95, corr 0.80-0.85, median relative error 11-25%, bias +0.11 to +0.19 (the shrinkage of mostly
  negative truths). The cause is the one of section 3: the s8 push writes no legs for neurons with a < -3. Shrunk,
  not flipped; r38's dropped entries there are at 0.11-0.23 of the truth with sign agreement 0.57-0.77.

opr30 per network (columns as in section 3):

TABLE_OPR30_ENTRIES

Diagonal t3, slope / corr against the truth:

TABLE_OPR30_DIAG

Band off, t3 is close to exact inside the column band (slope 0.95-1.00; 0.86 on mini 0 at t = 11) and outside it at
t = 5-10 (0.97-0.99); at t = 11-14 the out-of-band t3 is shrunk like the entries (0.50-0.62). With the band on, s7
degrades the out-of-band t3 from t = 7 (0.93-0.94) to 0.68-0.75 at t = 10 and 0.27-0.31 at t = 14, as in r38.

**Oracle swap** (B = wu2 v_engine_nott; entries of M1 replaced at every band layer, specs as in sections 4 and 6):

TABLE_OPR30_SWAP

* (a) band on x0.880-0.920 of B; (z) the clean band x0.871-0.938; (c) the dropped entries set to their truth (m)
  x1.002-1.015. The engine's dropped entries are close to the truth already, so m stays near B, and taking them out
  is what gains.
* Where the clean band gains: zeroing the dropped entries at layers 2-10 only (zlo) gives x0.892-0.949, at layers
  11-14 only (zhi) x0.932-0.941, both (z) x0.871-0.938: most of it comes from layers 2-10, where the dropped entries
  are exact, and the two do not add (zlo x zhi would be x0.84-0.88). By why an entry is dropped: zeroing only the
  entries whose doubled index is out (zcol, |a_j| >= 3) gives x0.840-0.937, more than the whole clean band, while
  zeroing only those whose single index is out (zrow, a_i < -1.5) costs x1.099-1.190; the clean band nets the two.
* With the truth: dmm (every off-diagonal entry = truth + fold, diagonal exact) x0.852-1.076, about neutral on
  average; dd (the diagonal t3 = truth) x0.724-0.811, a gain; with the truth on the kept entries and the dropped ones
  zeroed (mz) x0.712-0.892, which is x0.83-0.91 of dmm, so zeroing the dropped entries pays even when every other
  off-diagonal entry is exact; the truth everywhere, the diagonal included (ddmm, layers 1-14), x0.636-0.783, and with
  the dropped entries zeroed on top (ddmz) x0.548-0.742, x0.85-1.00 of ddmm. In r38 the same settings cost x22-24
  (dmm), x27-42 (dd) and x26-38 (ddmm).
* Scale test: s090 x2.13-2.21, s125 x13.4-17.7. The same bowl around s = 1 as in r38 (x1.8-1.9 and x9.9-10.6 there);
  in opr30 s = 1 is also about the truth's scale.
* By layer (the band-on over band-off MSE of each layer's means, mech_run.py layer_mse): in both files the band costs
  up to 4% at layers 3-9 (the clean band up to 6%) and gains at the last layers; at layer 15 opr30's band gives
  x0.880-0.920 (z: x0.871-0.938), r38's x0.973-1.016 (z: x0.950-1.025). What zeroing at layers 2-10 gains (zlo) shows
  in the means only from layer 11 on (mini 0: x1.04 at layer 6, x0.99 at layer 10, x0.95 at 11, x0.90 at 15).

## 9. Which change removed the gain

**The chain.** From the diffs of the estimator files:

* r37 = opr30 + R36 + L4. R36: compression rank 192 -> 256 (`Estimator.__init__` default). L4: `_R25B['s1']` 6 -> 4,
  the fold-layer push inner restriction from layer 4 (was 6), with bins for layers 4 and 5.
* r38 = r37 + R38 (research/r38/make_r38.py: the scale-mode extraction, with the lam' law off), two changes. **Law
  off**: LMK4 and LMK6 (lines 2710-2711) zero the lam'_t, lam'_{t+1} rows of FF where the per-neuron algebra reads
  them (XA3 at layers 1-14, line 3425; the last layer's readout, line 3393). With them the off-diagonal K4 law leaves
  the engine: d4 = g dA + g lam' sigma^2 keeps only g dA (line 1285), and the newborn source's K3 feed c = (g / 4)
  lam' w11 (V_C, line 1265; CSRC, line 2781) and the (3,1) slice lam' w31, lam' w21 (lines 1266-1267) are 0. The lam'
  table itself (LAM_REFIT) is unchanged, and in the submitted file it still reaches the correctors' features, which
  the v_engine cuts switch off. **Extraction**: `_X38` (line 1659; lines 3463-3475, 3501-3503) takes the input-norm
  scale mode out of C', the K4 core rows (dA: the next layer's preactivation K4 diagonal, line 3461, which enters
  every mean and Hermite coefficient there through d4) and the newborn's K3, and writes the extraction's K3 pair (-6
  eps m, w11) into the feed slot the law left empty.

make_cuts.py cuts r37 as r38 was cut (est_r37/v_engine is byte-identical to estimators/r38/r38_eng_base) and r38 with
each half of R38 undone (est_r38x: lawon = LMK4 and LMK6 all ones; noext = `_X38['on']` False; lawon_noext = both).
r38 with both halves undone reproduces r37 bit for bit on mini 0 (pred sha16 63fe5f85a7e185fd band on,
6658b425761157ba band off, the same final MSE; 67,594 more FLOPs for the unit masks), so R38 is exactly these two
changes.

**Band on / off, per public mini** (final-layer MSE of each file's v_engine over its v_engine_nott, ablate.py):

TABLE_GAIN

R36 and L4 kept the gain: r37 x0.919, band on better on 16 of 16, as opr30 (x0.918). R38 took it, and of R38's two
halves it is the extraction. Without the extraction the band gains with the law on (r37, x0.919) and with it off
(x0.894, 16 of 16); with the extraction it does not, with the law off (r38 as submitted, x0.983, better on 11 of 16)
or on (x1.062, band on worse on 16 of 16). Taking the law out moves the ratio the other way, by x0.93-0.97. The 2 x 2
in short (band on / off, geometric mean over minis 0-15; band on better on k of 16):

| | law on | law off |
|---|---|---|
| no extraction | r37 x0.919 (16 of 16) | x0.894 (16 of 16) |
| extraction | x1.062 (0 of 16) | r38 x0.983 (11 of 16) |

Each half alone is a much worse engine: the law off alone costs x2.15 of r37's band-on MSE (x2.21 band off), like
kp_table.json's v_engine_nolaw on the opr30 file (x2.12); the extraction on top of the law costs x1.74 (band on,
x1.51 band off). Together they are 5% (band on) and 11% (band off) better than r37. So the extraction takes over
what the fitted law did and does more (inferred from these ratios); the band's gain does not survive it.

**Which part of the extraction** (the r38 file with one of the extraction's four parts off, or the whole extraction at
half strength; band on / off, public minis 0-7; make_cuts.py est_r38x noxc, noxa, noxw, noxf, x05):

TABLE_XPART

Three of the four parts do not come apart as engines: with the C', the dA or the E part off and the other three on,
the engine is far worse (band-on MSE over r37's x17.8-27.5, x5.0-7.7 and x15.7-21.7), as R38's own notes found for
each part alone in the readable engine (research/r38/make_r38.py: every part alone x7 to x39 worse). The feed pair is
small: without it the engine stays near r38 (x0.91-1.04 of r37's band-on MSE on minis 0-7, r38 x0.942). The band's
ratio sorts the engines mainly by the dA part. With the dA part off, so that the next layer's K4 core rows keep the
scale mode's share, the band gains x0.915, better on 8 of 8, as in the engines without any extraction (x0.887-0.919 on
these minis: r38 without it, opr30, r37). With the dA part on it gains little or nothing: x0.964 without the feed pair
(better than r38's own ratio on 8 of 8, in an engine nearly as good as r38), x0.992 without the C' part (a 1% move in
an engine 22 times worse), x1.030 without the E part (worse on 8 of 8), x0.989 for r38 itself on these minis. The
whole extraction at half strength (eps_t x 0.5, law off; band-on MSE x1.17-1.38 of r37's) keeps most of the gain:
x0.912, better on 8 of 8. The gain goes only when the scale mode is (nearly) all taken out, not in proportion to how
much is.

**What the change did to the third cumulant** (each file's band-off engine against the Monte Carlo truth, minis 0-3;
kept slope / dropped slope (dropped corr), means over the four networks):

TABLE_XFILE

Diagonal t3, slope inside / outside the column band:

TABLE_XFILE_DIAG

r37 and r38 without the extraction have opr30's third cumulant at every layer; r38 with the law back has r38's. The
extraction is what moves the engine's T and t3 away from the Gaussian-input truth: by about 8% at layer 2 and by about
half at layer 14. Is r38's lower third cumulant the extraction taking the scale mode's share out on purpose? Only at
the first layers. With the inputs put on the sphere of radius sqrt(n) (mc_sphere.py: the same Gaussian draws with the
common factor |x| / sqrt(n) divided out; mini 0, 10^7 samples; sphere.md), the truth's third cumulant is up to about
10% smaller: opr30, r37 and r38 without the extraction have kept-entry slopes 1.01-1.10 on it (dropped entries and t3
up to 1.17 at the first layers). r38's kept-entry slopes on it are 1.02 at layer 2 and 0.96 at layer 3, then fall to
0.54 at layer 14 (dropped 0.18, t3 0.29): the extraction takes out about the scale mode's share at the first layers,
then more at every layer, about half of the third cumulant by the last one. Which of its parts does that was not
measured. Slopes on the sphere-input truth, mini 0 (kept / dropped / t3; sphere.md has the same against the Gaussian
truth):

TABLE_SPHERE

**The clean band and the truth on the dropped entries in each file** (minis 0, 1, 2, 3; each over its own B):

TABLE_VAR

The clean band moves the MSE as the submitted band does in every file (z / B: r37 x0.862-0.983, r38 without the
extraction x0.820-0.886, r38 with the law back x1.015-1.085). The truth on the dropped entries costs a little where
the engine's entries are close to the Gaussian-input truth (opr30 x1.002-1.015, r37 x1.012-1.034, r38 without the
extraction x1.017-1.024), much in r38 (x1.148-1.182), and is mixed in r38 with the law back (x0.941-1.031).

**Which dropped entries carry the gain** (the clean band split by why an entry is dropped: zrow zeroes only the entries
whose single index is out of the band, a_i < -1.5; zcol only those whose doubled index is out, |a_j| >= 3; minis
0-3, each over its own B):

TABLE_ZSPLIT

The pattern is the same in all three: the entries dropped for their doubled index carry the gain and those dropped for
their single index cost. What the dA part changes is the size of the first. Zeroing only the column drops gives
x0.840-0.937 in opr30 and x0.838-0.874 in r38 without only the dA part, but x0.932-1.008 in r38; zeroing only the row
drops costs x1.099-1.190, x1.046-1.073 and x1.027-1.066.

**Does the band do the extraction's job?** (mech_overlap.py: final-layer means of each file's band-off engine B and
band-on engine P on minis 0-3 against the dataset's means; e = truth - B is the band-off error and d = P - B what the
band changes, so the band changes the MSE by (|d|^2 - 2 d.e) / n.)

The band's change, file by file (on / off from the recorded means; cos(d, e) > 0: the band moves the means toward the
truth):

TABLE_OVERLAP_FILES

How alike the band's changes are across files (cosine of d):

TABLE_OVERLAP_PAIRS

Transfer: one file's band change put on another file's band-off means (MSE over that file's band-off MSE, geometric
mean over minis 0-3; the diagonal is each file's own band on / off):

TABLE_OVERLAP_TRANSFER

Splits. Between two band-off engines e splits exactly into the second engine's change X and the error it leaves: for
the engine without the extraction (noext) X = B(r38) - B(noext), the extraction's own change, and e = X + e38. The
band's MSE change then splits into a part along X, a part along the error left, and |d|^2. Rows: noext by the whole
extraction; r37 by the extraction at the law on (X = B(lawon) - B(r37)) and by all of R38; opr30 by everything from
opr30 to r38; r38 without the extraction's dA part (noxa) by the dA part's own change (X = B(r38) - B(noxa)); noext
by the other three parts (X = B(noxa) - B(noext)). Values x 1e-9; negative = the band lowers the MSE.

TABLE_OVERLAP

**What this says.** The band changes the final means in nearly the same way in opr30 and in r38: the cosine between
the two changes is 0.87-0.90 on minis 0-3, and each is about a quarter of its file's band-off error (rms). What
differs is the error that change meets. In opr30 it points toward the band-off error (cosine 0.28-0.37), in r38 hardly
(0.10-0.17). Put on the other file's band-off means, r38's band change gains on opr30 (x0.920-0.935) and opr30's gains
little or nothing on r38 (x0.952-1.034): the error moved, not the band. Along the change from opr30's band-off engine
to r38's (cosine with the band's change 0.49-0.50) the opr30 band takes 2.0-2.7e-9 off the MSE, along the error r38
leaves 1.3-2.6e-9, and its own size costs 1.5-1.9e-9. r38 has made the first part already; the second about pays for
the cost, which is the x0.95-1.03 above. The engine without only the extraction's dA part (noxa) shows where that
error sits. Its band change is the same again (cosine 0.83-0.91 with opr30's and r38's), and 87-92% of its gross gain
lies along the dA part's own change (X = B(r38) - B(noxa); cosine with the band's change 0.41-0.68; 9.0-14.0e-9 of MSE
along it, 1.1-1.7e-9 along the error r38 leaves, for a cost of 1.0-1.4e-9). Every file's band change gains on noxa's
band-off means (x0.885-0.917) as on opr30's (x0.901-0.929) and not on r38's (x0.989-0.994). So the band offsets part
of an error that the dA part removes outright: the error the engine makes while the next layer's K4 core rows carry
the input-norm scale mode's share. All six engines say the same. Their band changes are alike (cosine 0.83-0.94
between any two), and the transfer table sorts by column, not by row: on the band-off means of the four engines whose
K4 core rows keep the scale mode's share (opr30, r37, r38 without the extraction, r38 without only its dA part) every
file's band change gains (x0.88-0.93); on r38's it is neutral (x0.99-1.00); on those of r38 with the law back, which
has the dA part and the law's K4 terms on top, it costs (x1.07-1.10). Without the extraction, 76-88% of the band's
gross gain lies along the whole extraction's change (cosine 0.34-0.54) and none along the change of its other three
parts (C', E, feed pair: cosine -0.27 to -0.56, the band works against them, and its gain lies along the error they
leave, which is what the dA part removes).

## 10. Verified directly vs inferred

Measured directly: the band sets per layer (the submitted rule applied to the engine's own a, recorded by the hook);
the orientation of M1 (Monte Carlo correlation); the bit-for-bit match of every hooked copy with the cut it comes from
(r38: est/; opr30: est_opr30/; r37 and the r38 variants: est_r37/, est_r38x/; minis 0-3) and of r38 with both halves
of R38 undone against r37 (mini 0); every engine-vs-truth statistic of sections 3, 5, 8 and 9; every final-layer MSE
(one deterministic run per setting and network; band on / off on minis 0-15, the extraction's parts on minis 0-7); the
variance deficit and the cancellation correlations of section 5; the alignments and the transfer test of section 9
(final means, minis 0-3); every file's third cumulant against the sphere-input truth (mini 0).

Read from the code, consistent with the measurements but not traced value by value: that the a < -3 legs at layers
11-14 are exactly zero (s8 output columns; the measured slopes collapse exactly there and only there, in both files);
that s7's t3 defect comes from the missing compressed share and the stale-column fold subtraction (the measured
out-of-band t3 degrades exactly on s7 layers, and v_engine_nos7 does not show it); the C' weights of the dropped
entries (from the diagram table); what R36, L4 and R38 change (the diffs), which terms LMK4 / LMK6 switch off
(`_vec_rows`, lines 1255-1268, and `_expand`, line 1285) and what each of the extraction's parts takes out (lines
3463-3475 and 3501-3503, and the header of research/r38/make_r38.py).

Inferred, not tested: what, inside the cumulants, the band's change and the extraction's change share (measured: the
alignment of their effects on the final means and the transfer test of section 9; not traced: through which terms);
why the dA part in particular (the 2 x 2 and the part runs are measured; the path from the K4 core rows' scale-mode
share to the error the band offsets is not traced); that the extraction does what the fitted law did (from the 2 x 2
ratios only); that kp_table.json's corpus result (opr30 file, untouched corpus) has the mechanism measured here on
public minis. The truth-based results stand on four networks (minis 0-3), the band on / off ratios on sixteen (eight
for the extraction's parts). The production band keeps the thin rows' share on the dropped entries and the clean band
(z, dkz) zeroes it; in r38 the two differ by x1.017 / x1.002 (v_engine_nos7 vs z), measured only in aggregate.

## 11. Surprises

1. kp_table.json's v_engine_nott row (x1.276 adjusted, x1.101 MSE, 48 of 48 worse) was measured on estimators/wu2,
   which make_variants.py cut from the opr30 file (VARIANTS.json base ..._opr30, submission 336955), not from the r38
   file graded as 337488. On public minis 0-15 the opr30 cuts show the
   large gain (x0.918, 16 of 16), the r38 cuts a small one (x0.983, 11 of 16): about four fifths of the band's benefit
   is gone in the submitted file.
2. Exact values hurt in r38: the truth on the dropped entries costs 15-18% of final MSE on four networks; the truth on
   every off-diagonal entry x22-24; the true diagonal alone x27-42; at one layer x1.2-2.6; each of these makes the
   touched layer's C' and the next layer's t3 and variance more accurate. In opr30 the same truths cost about nothing
   or help (section 8).
3. At layers 11-14 the band-off engine does not compute the entries of neurons with a < -3 either: the push writes
   only the output columns with a >= -3. One place in the code says otherwise: `_R25B['s8']['out']` is 3.5 and the
   header (lines 39-40) says "output size (alive 3.5 at t)", but line 2995 uses cs_[3], the count below -_SAT['tb'],
   so since b30 moved tb from 3.5 to 3.0 the s8 output threshold is -3.0.
4. s7 degrades the submitted file's t3 for out-of-band neurons from layer 7 on (slope 0.09 vs 0.21 at t = 14 on mini
   0); v_engine_nos7 is clean. Small and of either sign on the final MSE. opr30 shows the same.
5. The true kappa3(z_i, z_j, z_j) is negative for practically every j when the single index is dead (a_i <= -1.5); the
   dropped entries are larger, in standardized units, than the kept ones (rms 0.022-0.046 vs 0.015-0.037).
6. The engine's next-layer variance is too low by up to 4.7% at layer 15 on average; a tenth of it is the input-norm
   scale mode.
7. The extraction, not the law-off, took the band's gain. With the extraction on, the band does not help whether the
   law is on (x1.062, worse on 16 of 16) or off (x0.983); with it off, the band helps whether the law is on (x0.919)
   or off (x0.894, the largest gain measured).
8. Each half of R38 alone is a much worse engine (law off x2.15, the extraction with the law x1.74 of r37's band-on
   MSE); together they are 5% better band on and 11% band off.
9. The band changes the final means in nearly the same way in every engine measured (cosine 0.83-0.94 between any
   two of six); what changed between the files is the error that change meets.
10. Of the extraction's four parts the one that decides the band's gain is the one in the next layer's K4 core rows
    (dA): with only that part off the band gains as much as without any extraction (x0.915), in an engine six times
    worse than r37, and 87-92% of that gain lies along what the dA part itself changes in the final means.
11. In opr30 the band's gain is all in the entries dropped for their doubled index (|a_j| >= 3; zeroing only those
    x0.840-0.937); zeroing only those dropped for their single index (a_i < -1.5) costs x1.10-1.19. And it stays with
    exact cumulants: with the truth everywhere, zeroing the dropped entries still gives x0.85-1.00.
12. r38's third cumulant is shrunk against the Gaussian-input truth (section 3) because of the extraction: with only
    the extraction undone the engine has opr30's slopes again. With the scale mode taken out of the truth (inputs on
    the sphere, mini 0) the shrinkage stays: the scale mode is up to about 10% of the third cumulant, r38's is about
    half of the truth's at layer 14.

## 12. Reproduce (Mac, ~/nova-arc)

Heavy runs go through research/r25a/mac_job25.sh (each engine run has a ~6 GB footprint; the scripts' headers show
the call). Order:

    cd ~/nova-arc
    .venv-fs0123/bin/python research/wu/mech/make_mech.py --base estimators/r38/r38_sub_x100_s2 --out research/wu/mech/est
    for i in 0 1 2 3; do .venv-fs0123/bin/python research/wu/mech/mc_truth.py --id $i --n 10000000 --batch 50000 --out research/wu/mech/mc; done
    for i in 0 1; do .venv-fs0123/bin/python research/wu/mech/mc_cov.py --id $i --n 10000000 --batch 50000 --out research/wu/mech/mc; done
    bash research/wu/mech/run_ref.sh      # band on / off / no s7 with ablate.py, and the recording runs of the hooked copies
    bash research/wu/mech/run_swap.sh     # oracle swap, mini 0 and 1
    bash research/wu/mech/run_layers.sh   # one-layer swaps with C' kept
    bash research/wu/mech/run_scale.sh    # scaled off-diagonal, Monte Carlo diagonal
    bash research/wu/mech/run_more.sh     # mini 2 and 3
    bash research/wu/mech/run_ext.sh      # band on / off of r38 and opr30 on minis 0-15
    .venv-fs0123/bin/python research/wu/mech/make_cuts.py   # second round: est_opr30, est_r37, est_r38x, CUTS.json
    bash research/wu/mech/run_round2.sh   # opr30 recordings and oracle (minis 0-3); band on / off of r37 and the r38 variants (minis 0-15)
    bash research/wu/mech/run_round2b.sh  # opr30: where the clean band's gain comes from (minis 0-3)
    bash research/wu/mech/run_round2c.sh  # r37 and the r38 variants: recordings, z and m (minis 0-3)
    bash research/wu/mech/run_round2d.sh  # the r38 file with one of the extraction's four parts off: band on / off (minis 0-7)
    bash research/wu/mech/run_round2e.sh  # band-on final means of r37 and the r38 variants, then mech_overlap.py
    bash research/wu/mech/run_round2f.sh  # the extraction at half strength (minis 0-7); zrow / zcol for r38 and r38 without dA
    .venv-fs0123/bin/python research/wu/mech/mech_overlap.py  # overlap.json, overlap.md (needs run_round2c.sh and run_round2e.sh)
    .venv-fs0123/bin/python research/wu/mech/mc_sphere.py --id 0 --n 10000000 --batch 50000 --out research/wu/mech/mc
    .venv-fs0123/bin/python research/wu/mech/sphere_cmp.py    # sphere.json, sphere.md (needs mc_sphere.py and the recordings)
    .venv-fs0123/bin/python research/wu/mech/mech_report.py   # results.json, tables.md, tables_compact.md
    .venv-fs0123/bin/python research/wu/mech/mech_local.py    # local.json, local.md
    .venv-fs0123/bin/python research/wu/mech/mech_decomp.py   # decomp.json, decomp.md
    .venv-fs0123/bin/python research/wu/mech/scale_check.py   # the variance against scale-normalized Monte Carlo
    .venv-fs0123/bin/python research/wu/mech/fill_readme.py   # this file from README.template.md

## 13. Files

research/wu/mech/ (Mac ~/nova-arc and the PC mirror; the large arrays stay on the Mac):

* scripts: make_mech.py, make_cuts.py, mc_truth.py, mc_cov.py, mech_run.py, mech_report.py, mech_local.py,
  mech_decomp.py, mech_overlap.py, mc_sphere.py, sphere_cmp.py, scale_check.py, fill_readme.py, run_ref.sh,
  run_swap.sh, run_layers.sh, run_scale.sh, run_more.sh, run_ext.sh, run_resume.sh, run_round2.sh, run_round2b.sh,
  run_round2c.sh, run_round2d.sh, run_round2e.sh, run_round2f.sh
* results: results.json (entry statistics per network, layer and group: `entries` r38, `entries_opr30`, `entries_r37`,
  `entries_r38noext`, `entries_r38lawon`; every run's final and per-layer MSE and band sets; `overlap`:
  mech_overlap.py's numbers), tables.md (all tables, both engine values), tables_compact.md, local.json / local.md,
  decomp.json / decomp.md, overlap.json / overlap.md, sphere.json / sphere.md, runs/*.jsonl (one line per run and
  network; second round: opr30_*.jsonl, r37_*.jsonl, r38x_*.jsonl, r38_zsplit.jsonl), mc/*.json (Monte Carlo run
  records), est/MECH.json, est/MECH_v1.json and CUTS.json (every cut's replacements and sha256), README.template.md
* Mac only: est/ (the six r38 cuts; MECH_v1.json is the first build without the C' hook, which run_ref.sh and
  run_swap.sh used; the hook is a None-guarded call and the rebuilt band-off copy reproduces the same predictions,
  runs/verify.jsonl), est_opr30/, est_r37/, est_r38x/ (the second round's cuts), mc/*.npz (truth arrays, ~100 MB per
  network and kind; mcs_mini0.npz: the sphere-input truth), rec/ (r38 recordings of M1, F, Mx, C' per layer, ~3.8 GB),
  rec_opr30/, rec_r37/, rec_r38noext/, rec_r38lawon/ (the second round's recordings, ~3.1 GB), pred_r37/,
  pred_r38noext/, pred_r38lawon/, pred_r38noxa/ (final means only), logs/
