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
   x0.950, x1.025, x0.981, x0.984 (minis 0, 1, 2, 3);
   the submitted band (with s7 and the thin rows) by
   x0.973, x1.016, x0.989, x0.977;
   over minis 0-15 the submitted band's on / off ratio is
   geometric mean x0.983, band on better on 11 of 16
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

| mini | t | Pa / Pb | dropped | kept res | kept med | kept sign | kept corr | kept slope | drop res | drop med | drop sign | drop corr | drop slope | rows-only slope / corr | cols-only slope / corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 1024 / 1024 | 0 | 0.84 | 0.11 | 1.00 | 0.99 | 0.92 | - | - | - | - | - | - / - | - / - |
| 0 | 3 | 960 / 1024 | 65472 | 0.87 | 0.16 | 0.99 | 0.99 | 0.87 | 0.98 | 0.23 | 0.98 | 0.99 | 0.80 | 0.80 / 0.99 | - / - |
| 0 | 4 | 896 / 1024 | 130944 | 0.89 | 0.22 | 0.98 | 0.98 | 0.81 | 0.98 | 0.30 | 0.97 | 0.99 | 0.74 | 0.74 / 0.99 | - / - |
| 0 | 5 | 896 / 1024 | 130944 | 0.91 | 0.27 | 0.96 | 0.98 | 0.77 | 0.99 | 0.36 | 0.96 | 0.98 | 0.69 | 0.69 / 0.98 | - / - |
| 0 | 6 | 832 / 960 | 249634 | 0.92 | 0.30 | 0.95 | 0.97 | 0.74 | 0.97 | 0.38 | 0.94 | 0.97 | 0.66 | 0.65 / 0.97 | 0.70 / 0.97 |
| 0 | 7 | 832 / 960 | 249640 | 0.93 | 0.33 | 0.94 | 0.96 | 0.71 | 0.98 | 0.42 | 0.94 | 0.96 | 0.62 | 0.62 / 0.97 | 0.66 / 0.96 |
| 0 | 8 | 768 / 896 | 360136 | 0.94 | 0.38 | 0.93 | 0.95 | 0.66 | 0.97 | 0.46 | 0.91 | 0.95 | 0.59 | 0.59 / 0.96 | 0.62 / 0.95 |
| 0 | 9 | 832 / 896 | 302839 | 0.95 | 0.41 | 0.92 | 0.95 | 0.63 | 0.98 | 0.49 | 0.91 | 0.95 | 0.56 | 0.56 / 0.96 | 0.59 / 0.95 |
| 0 | 10 | 832 / 832 | 356059 | 0.95 | 0.43 | 0.91 | 0.94 | 0.61 | 0.98 | 0.51 | 0.91 | 0.95 | 0.53 | 0.53 / 0.94 | 0.56 / 0.94 |
| 0 | 11 | 768 / 896 | 360127 | 0.96 | 0.47 | 0.89 | 0.90 | 0.55 | 0.98 | 0.83 | 0.76 | 0.56 | 0.23 | 0.23 / 0.36 | 0.28 / 0.68 |
| 0 | 12 | 768 / 768 | 458371 | 0.96 | 0.49 | 0.89 | 0.92 | 0.54 | 0.98 | 0.82 | 0.77 | 0.61 | 0.21 | 0.21 / 0.28 | 0.30 / 0.70 |
| 0 | 13 | 704 / 704 | 552484 | 0.97 | 0.51 | 0.89 | 0.91 | 0.51 | 0.98 | 0.79 | 0.75 | 0.63 | 0.22 | 0.22 / 0.34 | 0.31 / 0.72 |
| 0 | 14 | 704 / 704 | 552472 | 0.97 | 0.53 | 0.88 | 0.90 | 0.49 | 0.98 | 0.91 | 0.67 | 0.52 | 0.16 | 0.14 / 0.22 | 0.25 / 0.65 |
| 1 | 2 | 960 / 1024 | 65472 | 0.82 | 0.12 | 1.00 | 0.99 | 0.92 | 0.96 | 0.16 | 0.99 | 0.99 | 0.86 | 0.86 / 0.99 | - / - |
| 1 | 3 | 960 / 1024 | 65472 | 0.87 | 0.17 | 0.99 | 0.99 | 0.87 | 0.98 | 0.23 | 0.98 | 0.99 | 0.80 | 0.80 / 0.99 | - / - |
| 1 | 4 | 896 / 1024 | 130944 | 0.89 | 0.21 | 0.98 | 0.98 | 0.82 | 0.97 | 0.30 | 0.96 | 0.99 | 0.74 | 0.74 / 0.99 | - / - |
| 1 | 5 | 896 / 1024 | 130944 | 0.91 | 0.27 | 0.96 | 0.97 | 0.77 | 0.99 | 0.36 | 0.95 | 0.98 | 0.68 | 0.68 / 0.98 | - / - |
| 1 | 6 | 832 / 960 | 249631 | 0.93 | 0.31 | 0.95 | 0.97 | 0.73 | 0.97 | 0.40 | 0.93 | 0.97 | 0.65 | 0.65 / 0.97 | 0.69 / 0.97 |
| 1 | 7 | 832 / 896 | 302849 | 0.94 | 0.35 | 0.93 | 0.96 | 0.69 | 0.97 | 0.43 | 0.92 | 0.96 | 0.61 | 0.61 / 0.97 | 0.65 / 0.96 |
| 1 | 8 | 832 / 896 | 302857 | 0.94 | 0.39 | 0.92 | 0.95 | 0.65 | 0.98 | 0.47 | 0.92 | 0.95 | 0.57 | 0.57 / 0.95 | 0.61 / 0.94 |
| 1 | 9 | 768 / 832 | 409243 | 0.95 | 0.43 | 0.90 | 0.93 | 0.62 | 0.97 | 0.51 | 0.89 | 0.94 | 0.55 | 0.54 / 0.94 | 0.58 / 0.93 |
| 1 | 10 | 768 / 832 | 409236 | 0.95 | 0.45 | 0.90 | 0.92 | 0.59 | 0.97 | 0.54 | 0.87 | 0.93 | 0.52 | 0.52 / 0.94 | 0.55 / 0.92 |
| 1 | 11 | 768 / 768 | 458356 | 0.96 | 0.49 | 0.89 | 0.91 | 0.56 | 0.98 | 0.85 | 0.72 | 0.58 | 0.22 | 0.20 / 0.37 | 0.31 / 0.69 |
| 1 | 12 | 768 / 768 | 458369 | 0.96 | 0.52 | 0.87 | 0.90 | 0.53 | 0.98 | 0.88 | 0.69 | 0.56 | 0.19 | 0.17 / 0.30 | 0.30 / 0.69 |
| 1 | 13 | 704 / 704 | 552460 | 0.96 | 0.54 | 0.87 | 0.88 | 0.49 | 0.98 | 0.85 | 0.68 | 0.58 | 0.21 | 0.19 / 0.39 | 0.31 / 0.70 |
| 1 | 14 | 704 / 704 | 552488 | 0.97 | 0.57 | 0.86 | 0.87 | 0.47 | 0.98 | 0.94 | 0.63 | 0.44 | 0.14 | 0.12 / 0.20 | 0.23 / 0.62 |
| 2 | 2 | 960 / 1024 | 65472 | 0.82 | 0.11 | 1.00 | 0.99 | 0.93 | 0.95 | 0.16 | 1.00 | 0.99 | 0.87 | 0.87 / 0.99 | - / - |
| 2 | 3 | 960 / 1024 | 65472 | 0.87 | 0.17 | 0.99 | 0.99 | 0.87 | 0.98 | 0.25 | 0.98 | 0.99 | 0.79 | 0.79 / 0.99 | - / - |
| 2 | 4 | 960 / 1024 | 65472 | 0.90 | 0.23 | 0.98 | 0.98 | 0.81 | 0.99 | 0.29 | 0.98 | 0.99 | 0.75 | 0.75 / 0.99 | - / - |
| 2 | 5 | 896 / 960 | 188253 | 0.91 | 0.27 | 0.96 | 0.98 | 0.76 | 0.97 | 0.33 | 0.96 | 0.98 | 0.70 | 0.70 / 0.98 | 0.73 / 0.97 |
| 2 | 6 | 832 / 960 | 249629 | 0.93 | 0.32 | 0.95 | 0.97 | 0.71 | 0.97 | 0.40 | 0.94 | 0.97 | 0.64 | 0.64 / 0.97 | 0.67 / 0.97 |
| 2 | 7 | 832 / 896 | 302843 | 0.94 | 0.36 | 0.94 | 0.96 | 0.67 | 0.97 | 0.44 | 0.93 | 0.96 | 0.60 | 0.60 / 0.96 | 0.63 / 0.96 |
| 2 | 8 | 832 / 896 | 302834 | 0.94 | 0.41 | 0.92 | 0.94 | 0.62 | 0.97 | 0.49 | 0.91 | 0.95 | 0.56 | 0.56 / 0.95 | 0.58 / 0.94 |
| 2 | 9 | 768 / 832 | 409254 | 0.95 | 0.44 | 0.90 | 0.93 | 0.59 | 0.98 | 0.53 | 0.90 | 0.94 | 0.51 | 0.51 / 0.93 | 0.54 / 0.93 |
| 2 | 10 | 768 / 832 | 409254 | 0.96 | 0.49 | 0.89 | 0.91 | 0.55 | 0.98 | 0.57 | 0.88 | 0.92 | 0.47 | 0.47 / 0.92 | 0.50 / 0.91 |
| 2 | 11 | 768 / 768 | 458357 | 0.96 | 0.51 | 0.88 | 0.90 | 0.52 | 0.98 | 0.87 | 0.69 | 0.56 | 0.20 | 0.18 / 0.30 | 0.29 / 0.68 |
| 2 | 12 | 768 / 768 | 458367 | 0.96 | 0.57 | 0.85 | 0.88 | 0.48 | 0.98 | 0.92 | 0.64 | 0.49 | 0.16 | 0.14 / 0.25 | 0.25 / 0.65 |
| 2 | 13 | 704 / 704 | 552492 | 0.96 | 0.58 | 0.85 | 0.85 | 0.46 | 0.98 | 0.89 | 0.64 | 0.51 | 0.17 | 0.16 / 0.30 | 0.27 / 0.67 |
| 2 | 14 | 704 / 704 | 552476 | 0.97 | 0.60 | 0.84 | 0.85 | 0.45 | 0.98 | 0.97 | 0.57 | 0.38 | 0.11 | 0.08 / 0.21 | 0.22 / 0.61 |
| 3 | 2 | 960 / 1024 | 65472 | 0.83 | 0.12 | 1.00 | 0.99 | 0.92 | 0.96 | 0.17 | 0.99 | 0.99 | 0.86 | 0.86 / 0.99 | - / - |
| 3 | 3 | 960 / 1024 | 65472 | 0.88 | 0.17 | 0.99 | 0.99 | 0.86 | 0.98 | 0.23 | 0.98 | 0.99 | 0.80 | 0.80 / 0.99 | - / - |
| 3 | 4 | 896 / 1024 | 130944 | 0.90 | 0.23 | 0.98 | 0.98 | 0.80 | 0.99 | 0.29 | 0.97 | 0.98 | 0.74 | 0.74 / 0.98 | - / - |
| 3 | 5 | 896 / 960 | 188256 | 0.92 | 0.28 | 0.96 | 0.98 | 0.76 | 0.97 | 0.34 | 0.96 | 0.98 | 0.69 | 0.69 / 0.98 | 0.73 / 0.98 |
| 3 | 6 | 832 / 960 | 249634 | 0.93 | 0.32 | 0.95 | 0.97 | 0.72 | 0.97 | 0.40 | 0.94 | 0.97 | 0.64 | 0.64 / 0.97 | 0.68 / 0.97 |
| 3 | 7 | 832 / 896 | 302851 | 0.94 | 0.35 | 0.93 | 0.96 | 0.69 | 0.97 | 0.43 | 0.93 | 0.96 | 0.62 | 0.61 / 0.97 | 0.66 / 0.96 |
| 3 | 8 | 832 / 896 | 302852 | 0.94 | 0.38 | 0.92 | 0.95 | 0.66 | 0.98 | 0.47 | 0.92 | 0.95 | 0.57 | 0.57 / 0.95 | 0.61 / 0.95 |
| 3 | 9 | 768 / 832 | 409254 | 0.95 | 0.43 | 0.91 | 0.94 | 0.62 | 0.97 | 0.50 | 0.90 | 0.94 | 0.55 | 0.55 / 0.94 | 0.58 / 0.93 |
| 3 | 10 | 768 / 832 | 409244 | 0.95 | 0.45 | 0.90 | 0.92 | 0.58 | 0.98 | 0.54 | 0.88 | 0.92 | 0.50 | 0.51 / 0.93 | 0.52 / 0.92 |
| 3 | 11 | 768 / 768 | 458367 | 0.96 | 0.50 | 0.88 | 0.91 | 0.54 | 0.98 | 0.86 | 0.70 | 0.56 | 0.20 | 0.18 / 0.31 | 0.30 / 0.69 |
| 3 | 12 | 768 / 768 | 458368 | 0.96 | 0.52 | 0.87 | 0.89 | 0.52 | 0.98 | 0.89 | 0.67 | 0.53 | 0.19 | 0.16 / 0.26 | 0.29 / 0.69 |
| 3 | 13 | 768 / 704 | 507478 | 0.96 | 0.54 | 0.87 | 0.89 | 0.50 | 0.98 | 0.88 | 0.66 | 0.52 | 0.17 | 0.12 / 0.23 | 0.29 / 0.69 |
| 3 | 14 | 768 / 704 | 507496 | 0.97 | 0.56 | 0.85 | 0.87 | 0.48 | 0.98 | 0.96 | 0.60 | 0.41 | 0.11 | 0.07 / 0.11 | 0.23 / 0.62 |

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

| mini | t | band off: in | band off: out | band on (submitted): in | band on: out | band on without s7: out |
|---|---|---|---|---|---|---|
| 0 | 2 | 0.89 / 1.00 | - | 0.89 / 1.00 | - | - |
| 0 | 3 | 0.82 / 0.99 | - | 0.82 / 0.99 | - | - |
| 0 | 4 | 0.75 / 0.99 | - | 0.75 / 0.99 | - | - |
| 0 | 5 | 0.70 / 0.98 | - | 0.70 / 0.98 | - | - |
| 0 | 6 | 0.68 / 0.98 | 0.60 / 0.99 | 0.68 / 0.98 | 0.60 / 0.99 | 0.60 / 0.99 |
| 0 | 7 | 0.64 / 0.97 | 0.58 / 0.99 | 0.64 / 0.97 | 0.54 / 0.99 | 0.58 / 0.99 |
| 0 | 8 | 0.61 / 0.96 | 0.53 / 0.99 | 0.61 / 0.96 | 0.42 / 0.97 | 0.53 / 0.99 |
| 0 | 9 | 0.58 / 0.95 | 0.52 / 0.99 | 0.59 / 0.95 | 0.38 / 0.95 | 0.52 / 0.99 |
| 0 | 10 | 0.58 / 0.95 | 0.47 / 0.98 | 0.59 / 0.95 | 0.31 / 0.92 | 0.47 / 0.98 |
| 0 | 11 | 0.47 / 0.88 | 0.26 / 0.93 | 0.47 / 0.88 | 0.15 / 0.82 | 0.26 / 0.93 |
| 0 | 12 | 0.53 / 0.93 | 0.25 / 0.84 | 0.53 / 0.93 | 0.13 / 0.71 | 0.25 / 0.84 |
| 0 | 13 | 0.53 / 0.92 | 0.26 / 0.81 | 0.53 / 0.92 | 0.13 / 0.61 | 0.26 / 0.81 |
| 0 | 14 | 0.51 / 0.89 | 0.21 / 0.82 | 0.50 / 0.89 | 0.09 / 0.54 | 0.20 / 0.82 |
| 1 | 2 | 0.88 / 0.99 | - | 0.88 / 0.99 | - | - |
| 1 | 3 | 0.81 / 0.99 | - | 0.81 / 0.99 | - | - |
| 1 | 4 | 0.75 / 0.99 | - | 0.75 / 0.99 | - | - |
| 1 | 5 | 0.70 / 0.98 | - | 0.70 / 0.98 | - | - |
| 1 | 6 | 0.66 / 0.97 | 0.60 / 0.99 | 0.67 / 0.97 | 0.60 / 0.99 | 0.60 / 0.99 |
| 1 | 7 | 0.65 / 0.96 | 0.56 / 0.99 | 0.65 / 0.96 | 0.52 / 0.99 | 0.56 / 0.99 |
| 1 | 8 | 0.60 / 0.96 | 0.52 / 0.99 | 0.61 / 0.96 | 0.41 / 0.97 | 0.52 / 0.99 |
| 1 | 9 | 0.59 / 0.94 | 0.47 / 0.98 | 0.59 / 0.94 | 0.32 / 0.91 | 0.47 / 0.98 |
| 1 | 10 | 0.55 / 0.92 | 0.47 / 0.98 | 0.55 / 0.92 | 0.28 / 0.88 | 0.46 / 0.98 |
| 1 | 11 | 0.54 / 0.92 | 0.24 / 0.82 | 0.54 / 0.92 | 0.11 / 0.64 | 0.24 / 0.82 |
| 1 | 12 | 0.52 / 0.89 | 0.24 / 0.83 | 0.51 / 0.89 | 0.08 / 0.55 | 0.24 / 0.83 |
| 1 | 13 | 0.51 / 0.88 | 0.24 / 0.79 | 0.51 / 0.88 | 0.07 / 0.36 | 0.24 / 0.79 |
| 1 | 14 | 0.49 / 0.87 | 0.16 / 0.74 | 0.49 / 0.87 | 0.03 / 0.21 | 0.16 / 0.74 |
| 2 | 2 | 0.89 / 1.00 | - | 0.89 / 1.00 | - | - |
| 2 | 3 | 0.81 / 0.99 | - | 0.81 / 0.99 | - | - |
| 2 | 4 | 0.75 / 0.99 | - | 0.75 / 0.99 | - | - |
| 2 | 5 | 0.72 / 0.98 | 0.65 / 1.00 | 0.72 / 0.98 | 0.65 / 1.00 | - |
| 2 | 6 | 0.66 / 0.98 | 0.59 / 0.99 | 0.66 / 0.98 | 0.59 / 0.99 | - |
| 2 | 7 | 0.63 / 0.96 | 0.56 / 0.99 | 0.63 / 0.96 | 0.52 / 0.99 | - |
| 2 | 8 | 0.59 / 0.95 | 0.51 / 0.99 | 0.59 / 0.95 | 0.39 / 0.96 | - |
| 2 | 9 | 0.56 / 0.93 | 0.46 / 0.98 | 0.56 / 0.93 | 0.28 / 0.90 | - |
| 2 | 10 | 0.51 / 0.92 | 0.42 / 0.98 | 0.50 / 0.92 | 0.21 / 0.82 | - |
| 2 | 11 | 0.52 / 0.90 | 0.23 / 0.81 | 0.51 / 0.90 | 0.08 / 0.48 | - |
| 2 | 12 | 0.46 / 0.87 | 0.20 / 0.81 | 0.46 / 0.87 | 0.05 / 0.33 | - |
| 2 | 13 | 0.48 / 0.86 | 0.19 / 0.74 | 0.47 / 0.86 | 0.02 / 0.14 | - |
| 2 | 14 | 0.44 / 0.84 | 0.16 / 0.73 | 0.44 / 0.84 | 0.02 / 0.10 | - |
| 3 | 2 | 0.87 / 0.99 | - | 0.87 / 0.99 | - | - |
| 3 | 3 | 0.81 / 0.99 | - | 0.81 / 0.99 | - | - |
| 3 | 4 | 0.74 / 0.99 | - | 0.74 / 0.99 | - | - |
| 3 | 5 | 0.71 / 0.98 | 0.66 / 1.00 | 0.71 / 0.98 | 0.66 / 1.00 | - |
| 3 | 6 | 0.67 / 0.98 | 0.59 / 0.99 | 0.67 / 0.98 | 0.59 / 0.99 | - |
| 3 | 7 | 0.64 / 0.96 | 0.57 / 0.99 | 0.64 / 0.96 | 0.52 / 0.99 | - |
| 3 | 8 | 0.61 / 0.95 | 0.53 / 0.99 | 0.61 / 0.95 | 0.42 / 0.97 | - |
| 3 | 9 | 0.58 / 0.94 | 0.49 / 0.98 | 0.58 / 0.94 | 0.32 / 0.93 | - |
| 3 | 10 | 0.56 / 0.93 | 0.44 / 0.98 | 0.56 / 0.93 | 0.25 / 0.88 | - |
| 3 | 11 | 0.52 / 0.91 | 0.24 / 0.81 | 0.52 / 0.91 | 0.12 / 0.65 | - |
| 3 | 12 | 0.50 / 0.90 | 0.23 / 0.81 | 0.50 / 0.90 | 0.08 / 0.52 | - |
| 3 | 13 | 0.50 / 0.88 | 0.22 / 0.77 | 0.50 / 0.88 | 0.07 / 0.42 | - |
| 3 | 14 | 0.46 / 0.86 | 0.16 / 0.75 | 0.46 / 0.86 | 0.04 / 0.33 | - |

The diagonal is shrunk the same way. The submitted file (band on, with s7) is worse than band off and than v_engine_nos7
on the out-of-band neurons' t3 from t = 7 on: s7 leaves the compressed sources' share out of those neurons' t3 and
subtracts a fold share computed from leg columns it did not rebuild. Its effect on the final MSE is small and of either
sign (v_engine / v_engine_nos7: x1.007 on mini 0, x0.989 on mini 1).

## 4. How much of the final-layer error do they account for? (oracle swap, all band layers at once)

Band-off engine with entries of M1 replaced at every layer where the band acts (mech_run.py specs; B = v_engine_nott,
dkz = the clean band: dropped entries 0, kept entries and diagonal as the band-on code makes them, exactly). Rows s*,
k075, mm*, dd, ddmm are the scale test of section 6.

| run | mini 0 (x 1e-8) | mini 1 | / B mini 0 | / B mini 1 | / dkz mini 0 | / dkz mini 1 |
|---|---|---|---|---|---|---|
| v_engine: band on, as submitted (a) | 2.0801 | 1.9832 | 0.973 | 1.016 | 1.019 | 0.982 |
| v_engine_nos7: band on without s7 | 2.0659 | 2.0051 | 0.967 | 1.027 | 1.012 | 0.993 |
| v_engine_nott: band off (b) = B | 2.1374 | 1.9528 | 1.000 | 1.000 | 1.047 | 0.967 |
| z: dropped entries 0 (clean band) | 2.0309 | 2.0014 | 0.950 | 1.025 | 0.995 | 0.992 |
| dz: z + diagonal exact | 2.0411 | 2.0077 | 0.955 | 1.028 | 1.000 | 0.995 |
| dkz: z + diagonal and kept entries exact | 2.0419 | 2.0185 | 0.955 | 1.034 | 1.000 | 1.000 |
| d: band off, diagonal exact | 2.1581 | 1.9570 | 1.010 | 1.002 | 1.057 | 0.970 |
| dkx: band off, exact product (no APA) on band layers | 2.1643 | 1.9669 | 1.013 | 1.007 | 1.060 | 0.974 |
| xall: band off, exact product on layers 1-14 | 2.1381 | 1.9576 | 1.000 | 1.002 | 1.047 | 0.970 |
| enf: dropped = engine minus fold share | 2.1488 | 1.9551 | 1.005 | 1.001 | 1.052 | 0.969 |
| dkxnf: dkx, dropped minus fold share | 2.1630 | 1.9699 | 1.012 | 1.009 | 1.059 | 0.976 |
| m: dropped = Monte Carlo truth + fold (c) | 2.4712 | 2.2420 | 1.156 | 1.148 | 1.210 | 1.111 |
| mnf: dropped = Monte Carlo truth | 2.4599 | 2.2280 | 1.151 | 1.141 | 1.205 | 1.104 |
| dkm: dkz with dropped = truth + fold | 2.4606 | 2.2524 | 1.151 | 1.153 | 1.205 | 1.116 |
| dkmnf: dkz with dropped = truth | 2.4565 | 2.2450 | 1.149 | 1.150 | 1.203 | 1.112 |
| dkmn: dkm with the Monte Carlo noise doubled | 2.4663 | 2.2586 | 1.154 | 1.157 | 1.208 | 1.119 |
| dmm: every off-diagonal entry = truth + fold | 49.8337 | 43.9136 | 23.315 | 22.487 | 24.406 | 21.755 |
| s125: band off, off-diagonal x 1.25 (layers 1-14) | 22.6646 | 19.3532 | 10.604 | 9.910 | 11.100 | 9.588 |
| s090: band off, off-diagonal x 0.9 | 3.8686 | 3.7554 | 1.810 | 1.923 | 1.895 | 1.860 |
| s075: band off, off-diagonal x 0.75 | 15.0127 | 14.4671 | 7.024 | 7.408 | 7.352 | 7.167 |
| s050: band off, off-diagonal x 0.5 | 52.0106 | 50.6506 | 24.333 | 25.937 | 25.472 | 25.093 |
| s000: band off, off-diagonal x 0 | 175.0296 | 175.7565 | 81.888 | 90.002 | 85.719 | 87.072 |
| k075: band off, kept entries x 0.75 | 11.3475 | 13.3525 | 5.309 | 6.838 | 5.557 | 6.615 |
| mm075: dmm x 0.75 | 16.3543 | 13.7913 | 7.651 | 7.062 | 8.009 | 6.832 |
| mm050: dmm x 0.5 | 21.3812 | 24.7894 | 10.003 | 12.694 | 10.471 | 12.281 |
| dd: band off, diagonal = Monte Carlo 3 t3 (layers 1-14) | 58.2962 | 82.6597 | 27.274 | 42.329 | 28.550 | 40.950 |
| ddmm: dd + every off-diagonal entry = truth + fold | 54.5703 | 73.6268 | 25.531 | 37.703 | 26.725 | 36.475 |

* (a) band on: production 0.973 / 1.016 of B; the clean band (z) 0.950 / 1.025. (b) band off: 1. (c) dropped entries
  = Monte Carlo truth (m): 1.156 / 1.148.
* Controls: doubling the Monte Carlo noise (dkmn vs dkm) +0.2-0.3%; the fold share on the dropped entries (mnf vs m,
  dkmnf vs dkm) 0.2-0.6%; the APA level and the exact diagonal (d, dkx, xall vs B) at most 1.3%.

The same on mini 2 and 3:

| run | mini 2 (x 1e-8) | mini 3 | / B mini 2 | / B mini 3 |
|---|---|---|---|---|
| v_engine: band on, as submitted | 2.0560 | 2.1639 | 0.989 | 0.977 |
| v_engine_nott: band off = B | 2.0798 | 2.2138 | 1.000 | 1.000 |
| z: dropped entries 0 | 2.0405 | 2.1791 | 0.981 | 0.984 |
| dkz: clean band | 2.0436 | 2.1659 | 0.983 | 0.978 |
| dkx: band off, exact product | 2.0928 | 2.2126 | 1.006 | 0.999 |
| m: dropped = truth + fold | 2.4578 | 2.5787 | 1.182 | 1.165 |
| dkm: dkz with dropped = truth + fold | 2.4424 | 2.5628 | 1.174 | 1.158 |
| dkmnf: dkz with dropped = truth | 2.4277 | 2.5678 | 1.167 | 1.160 |
| dmm: every off-diagonal entry = truth + fold | 47.7670 | 52.8143 | 22.967 | 23.856 |

## 5. One layer at a time

Base: the clean band at every band layer. At one layer L the dropped entries take the engine's exact values (x) or
the truth (m), or every off-diagonal entry takes the truth (t); the state up to layer L is identical, so the change of
C'_L and of layer L + 1 is local.

One-layer swaps, final-layer MSE / clean band:

| L | dropped (mini 0 / 1) | x: engine values, mini 0 | mini 1 | m: truth on dropped, mini 0 | mini 1 | t: truth on all, mini 0 | mini 1 |
|---|---|---|---|---|---|---|---|
| 2 | 0 / 65472 | 1.000 | 0.986 | 1.000 | 0.991 | 1.000 | 1.286 |
| 3 | 65472 / 65472 | 1.008 | 0.998 | 1.005 | 0.993 | 1.356 | 1.428 |
| 4 | 130944 / 130944 | 0.996 | 0.991 | 0.999 | 0.991 | 1.403 | 1.705 |
| 5 | 130944 / 130944 | 1.011 | 0.995 | 1.004 | 0.997 | 2.150 | 1.777 |
| 6 | 249634 / 249631 | 1.006 | 0.996 | 1.016 | 0.998 | 2.401 | 1.687 |
| 7 | 249640 / 302849 | 1.008 | 0.990 | 1.013 | 0.992 | 2.235 | 1.659 |
| 8 | 360136 / 302857 | 1.014 | 0.991 | 1.021 | 0.992 | 2.628 | 2.228 |
| 9 | 302839 / 409243 | 1.019 | 0.981 | 1.035 | 0.987 | 2.092 | 2.088 |
| 10 | 356059 / 409236 | 1.017 | 0.989 | 1.032 | 0.984 | 2.003 | 1.816 |
| 11 | 360127 / 458356 | 1.002 | 0.994 | 0.996 | 0.993 | 1.874 | 1.738 |
| 12 | 458371 / 458369 | 1.004 | 0.997 | 1.015 | 1.001 | 1.718 | 1.736 |
| 13 | 552484 / 552460 | 0.998 | 0.996 | 0.997 | 0.998 | 1.361 | 1.491 |
| 14 | 552472 / 552488 | 0.999 | 0.998 | 1.001 | 0.995 | 1.229 | 1.237 |

Truth on every off-diagonal entry of layer L, local effect (clean band -> truth):

| mini | L | C'_L rel err | var(L+1) rms rel err | var(L+1) mean rel err | t3(L+1) rel rms err | MSE of layer L+1 means |
|---|---|---|---|---|---|---|
| 0 | 2 | 0.0250 -> 0.0250 | 0.0021 -> 0.0021 | -0.0016 -> -0.0016 | 0.2093 -> 0.2093 | 2.280e-09 -> 2.280e-09 |
| 1 | 2 | 0.0246 -> 0.0206 | 0.0021 -> 0.0020 | -0.0016 -> -0.0016 | 0.2145 -> 0.2072 | 1.808e-09 -> 4.207e-09 |
| 0 | 3 | 0.0388 -> 0.0329 | 0.0039 -> 0.0036 | -0.0029 -> -0.0029 | 0.2761 -> 0.2653 | 3.196e-09 -> 6.558e-09 |
| 1 | 3 | 0.0387 -> 0.0328 | 0.0038 -> 0.0035 | -0.0027 -> -0.0028 | 0.2786 -> 0.2689 | 2.539e-09 -> 6.264e-09 |
| 0 | 4 | 0.0583 -> 0.0514 | 0.0061 -> 0.0058 | -0.0043 -> -0.0044 | 0.3278 -> 0.3189 | 4.439e-09 -> 1.084e-08 |
| 1 | 4 | 0.0555 -> 0.0486 | 0.0062 -> 0.0059 | -0.0044 -> -0.0045 | 0.3297 -> 0.3191 | 3.311e-09 -> 9.440e-09 |
| 0 | 5 | 0.0724 -> 0.0647 | 0.0089 -> 0.0085 | -0.0062 -> -0.0062 | 0.3667 -> 0.3599 | 5.303e-09 -> 1.158e-08 |
| 1 | 5 | 0.0774 -> 0.0697 | 0.0096 -> 0.0092 | -0.0068 -> -0.0068 | 0.3824 -> 0.3734 | 4.512e-09 -> 9.666e-09 |
| 0 | 6 | 0.0950 -> 0.0868 | 0.0118 -> 0.0115 | -0.0083 -> -0.0084 | 0.4019 -> 0.3970 | 6.166e-09 -> 1.524e-08 |
| 1 | 6 | 0.1099 -> 0.1025 | 0.0130 -> 0.0126 | -0.0092 -> -0.0092 | 0.4169 -> 0.4098 | 5.852e-09 -> 1.230e-08 |
| 0 | 7 | 0.1194 -> 0.1109 | 0.0164 -> 0.0159 | -0.0113 -> -0.0113 | 0.4510 -> 0.4366 | 7.421e-09 -> 1.523e-08 |
| 1 | 7 | 0.1347 -> 0.1272 | 0.0181 -> 0.0175 | -0.0122 -> -0.0121 | 0.4598 -> 0.4533 | 7.425e-09 -> 1.446e-08 |
| 0 | 8 | 0.1481 -> 0.1398 | 0.0194 -> 0.0187 | -0.0135 -> -0.0135 | 0.4712 -> 0.4685 | 9.800e-09 -> 1.685e-08 |
| 1 | 8 | 0.1560 -> 0.1483 | 0.0213 -> 0.0209 | -0.0147 -> -0.0148 | 0.5064 -> 0.5017 | 8.249e-09 -> 1.451e-08 |
| 0 | 9 | 0.1736 -> 0.1659 | 0.0242 -> 0.0239 | -0.0162 -> -0.0165 | 0.5066 -> 0.5055 | 1.117e-08 -> 1.876e-08 |
| 1 | 9 | 0.1810 -> 0.1736 | 0.0257 -> 0.0254 | -0.0173 -> -0.0175 | 0.5300 -> 0.5265 | 1.057e-08 -> 1.826e-08 |
| 0 | 10 | 0.1914 -> 0.1834 | 0.0308 -> 0.0302 | -0.0207 -> -0.0208 | 0.6925 -> 0.6815 | 1.306e-08 -> 1.951e-08 |
| 1 | 10 | 0.2131 -> 0.2056 | 0.0347 -> 0.0337 | -0.0235 -> -0.0232 | 0.7210 -> 0.7097 | 1.310e-08 -> 1.975e-08 |
| 0 | 11 | 0.2335 -> 0.2262 | 0.0405 -> 0.0394 | -0.0270 -> -0.0268 | 0.7213 -> 0.7078 | 1.597e-08 -> 2.436e-08 |
| 1 | 11 | 0.2549 -> 0.2484 | 0.0425 -> 0.0420 | -0.0284 -> -0.0286 | 0.7346 -> 0.7255 | 1.588e-08 -> 2.096e-08 |
| 0 | 12 | 0.2710 -> 0.2640 | 0.0465 -> 0.0457 | -0.0319 -> -0.0320 | 0.7286 -> 0.7219 | 1.621e-08 -> 2.320e-08 |
| 1 | 12 | 0.2886 -> 0.2820 | 0.0469 -> 0.0468 | -0.0319 -> -0.0322 | 0.7431 -> 0.7322 | 1.819e-08 -> 2.501e-08 |
| 0 | 13 | 0.2998 -> 0.2935 | 0.0529 -> 0.0525 | -0.0354 -> -0.0357 | 0.7796 -> 0.7721 | 1.768e-08 -> 2.242e-08 |
| 1 | 13 | 0.3219 -> 0.3159 | 0.0608 -> 0.0603 | -0.0399 -> -0.0398 | 0.8103 -> 0.7957 | 1.926e-08 -> 2.490e-08 |
| 0 | 14 | 0.3304 -> 0.3245 | 0.0627 -> 0.0627 | -0.0429 -> -0.0433 | - | 2.042e-08 -> 2.509e-08 |
| 1 | 14 | 0.3604 -> 0.3543 | 0.0691 -> 0.0687 | -0.0467 -> -0.0471 | - | 2.019e-08 -> 2.497e-08 |

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

Band off, every off-diagonal entry of 3 T^T at layers 1-14 multiplied by s: final MSE / B = s = 0: x81.9 / x90.0; s = 0.5: x24.3 / x25.9; s = 0.75: x7.0 / x7.4; s = 0.9: x1.8 / x1.9; s = 1.25: x10.6 / x9.9 (mini 0 / 1); s = 1 is B itself. A quadratic
bowl centred near s = 1 (the engine's own scale; the truth is 1.1-2 times larger). The truth on every off-diagonal entry
costs x22-24, scaled to 0.75 or 0.5 still x7 or x10-13, so the pattern matters, not only the size. The Monte Carlo
diagonal t3 alone (dd) costs x27 / x42 of B, and with the truth off the diagonal as well (ddmm) x26 / x38. The engine's mean equations work with its own t3 and T. (The 10-21 version
of this note guessed that the K4 law's fitted lam' table carries this calibration. It cannot in r38: R38 multiplies
the lam' rows by zero wherever the engine reads them (LMK4, LMK6, lines 2710-2711; section 9). The opr30 engine has
the same bowl around its own scale (section 8), and there its own scale is the truth's.)

## 7. Band on / off across public minis 0-15 (no truth needed)

| mini | r38 band on (x 1e-8) | r38 band off | r38 on / off | r38 clean band / off | opr30 band on | opr30 band off | opr30 on / off |
|---|---|---|---|---|---|---|---|
| 0 | 2.0801 | 2.1374 | 0.973 | 0.955 | 2.2533 | 2.5604 | 0.880 |
| 1 | 1.9832 | 1.9528 | 1.016 | 1.034 | 2.3388 | 2.5420 | 0.920 |
| 2 | 2.0560 | 2.0798 | 0.989 | 0.983 | 2.4933 | 2.7715 | 0.900 |
| 3 | 2.1639 | 2.2138 | 0.977 | 0.978 | 2.4920 | 2.7603 | 0.903 |
| 4 | 1.9427 | 1.9609 | 0.991 | 0.975 | 2.0743 | 2.2974 | 0.903 |
| 5 | 1.6854 | 1.6501 | 1.021 | 1.008 | 1.7861 | 1.8456 | 0.968 |
| 6 | 1.4506 | 1.4167 | 1.024 | 1.018 | 1.6433 | 1.6875 | 0.974 |
| 7 | 1.8643 | 2.0108 | 0.927 | 0.926 | 2.1306 | 2.4677 | 0.863 |
| 8 | 2.0374 | 2.0975 | 0.971 | 0.983 | 2.2993 | 2.4015 | 0.957 |
| 9 | 2.1925 | 2.3004 | 0.953 | 0.940 | 2.3668 | 2.6646 | 0.888 |
| 10 | 1.7305 | 1.7937 | 0.965 | 0.960 | 1.9297 | 2.1202 | 0.910 |
| 11 | 1.9768 | 1.9309 | 1.024 | 1.014 | 2.1638 | 2.2059 | 0.981 |
| 12 | 1.7231 | 1.7704 | 0.973 | 0.968 | 1.8539 | 1.9848 | 0.934 |
| 13 | 2.1192 | 2.0729 | 1.022 | 1.018 | 2.1528 | 2.3269 | 0.925 |
| 14 | 2.1073 | 2.2712 | 0.928 | 0.923 | 2.3373 | 2.6788 | 0.873 |
| 15 | 1.8412 | 1.8679 | 0.986 | 0.965 | 2.0901 | 2.2691 | 0.921 |
| all | | | geometric mean 0.983, band on better on 11 of 16 | geometric mean 0.977, band on better on 11 of 16 | | | geometric mean 0.918, band on better on 16 of 16 |

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

| t | kept med | kept sign | kept corr | kept slope | kept bias | dropped med | dropped sign | dropped corr | dropped slope | dropped bias |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.06 | 1.00 | 1.00 | 1.00 | 0.00 | 0.04 | 1.00 | 1.00 | 1.00 | +0.01 |
| 3 | 0.05 | 1.00 | 1.00 | 1.00 | 0.00 | 0.03 | 1.00 | 1.00 | 1.00 | 0.00 |
| 4 | 0.05 | 1.00 | 1.00 | 0.99 | 0.00 | 0.03 | 1.00 | 1.00 | 0.99 | +0.01 |
| 5 | 0.05 | 1.00 | 1.00 | 0.99 | 0.00 | 0.03 | 1.00 | 1.00 | 0.99 | 0.00 |
| 6 | 0.05 | 1.00 | 1.00 | 0.99 | 0.00 | 0.03 | 1.00 | 1.00 | 0.99 | +0.01 |
| 7 | 0.05 | 1.00 | 1.00 | 0.99 | 0.00 | 0.03 | 1.00 | 1.00 | 0.98 | +0.01 |
| 8 | 0.05 | 1.00 | 1.00 | 0.99 | 0.00 | 0.04 | 1.00 | 1.00 | 0.98 | +0.01 |
| 9 | 0.05 | 1.00 | 1.00 | 0.99 | -0.01 | 0.04 | 1.00 | 1.00 | 0.97 | +0.01 |
| 10 | 0.05 | 1.00 | 1.00 | 0.98 | -0.01 | 0.05 | 1.00 | 1.00 | 0.97 | +0.01 |
| 11 | 0.06 | 0.99 | 0.99 | 0.97 | -0.01 | 0.14 | 0.94 | 0.83 | 0.60 | +0.15 |
| 12 | 0.06 | 0.99 | 1.00 | 0.98 | -0.01 | 0.14 | 0.94 | 0.83 | 0.59 | +0.13 |
| 13 | 0.06 | 0.99 | 0.99 | 0.97 | -0.02 | 0.13 | 0.94 | 0.83 | 0.60 | +0.13 |
| 14 | 0.06 | 0.99 | 1.00 | 0.98 | -0.01 | 0.21 | 0.93 | 0.80 | 0.54 | +0.15 |

The same for the r38 file (section 3's runs), for contrast:

| t | kept med | kept sign | kept corr | kept slope | kept bias | dropped med | dropped sign | dropped corr | dropped slope | dropped bias |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.11 | 1.00 | 0.99 | 0.92 | -0.01 | 0.16 | 0.99 | 0.99 | 0.86 | +0.16 |
| 3 | 0.17 | 0.99 | 0.99 | 0.87 | -0.03 | 0.24 | 0.98 | 0.99 | 0.80 | +0.23 |
| 4 | 0.22 | 0.98 | 0.98 | 0.81 | -0.08 | 0.30 | 0.97 | 0.99 | 0.74 | +0.29 |
| 5 | 0.27 | 0.96 | 0.98 | 0.77 | -0.09 | 0.35 | 0.96 | 0.98 | 0.69 | +0.31 |
| 6 | 0.31 | 0.95 | 0.97 | 0.73 | -0.16 | 0.40 | 0.94 | 0.97 | 0.65 | +0.32 |
| 7 | 0.35 | 0.94 | 0.96 | 0.69 | -0.17 | 0.43 | 0.93 | 0.96 | 0.61 | +0.30 |
| 8 | 0.39 | 0.92 | 0.95 | 0.65 | -0.22 | 0.47 | 0.91 | 0.95 | 0.57 | +0.31 |
| 9 | 0.43 | 0.91 | 0.94 | 0.61 | -0.27 | 0.51 | 0.90 | 0.94 | 0.54 | +0.30 |
| 10 | 0.45 | 0.90 | 0.92 | 0.58 | -0.29 | 0.54 | 0.89 | 0.93 | 0.51 | +0.30 |
| 11 | 0.49 | 0.89 | 0.91 | 0.54 | -0.34 | 0.85 | 0.72 | 0.57 | 0.21 | +0.42 |
| 12 | 0.52 | 0.87 | 0.90 | 0.52 | -0.36 | 0.88 | 0.69 | 0.55 | 0.19 | +0.39 |
| 13 | 0.54 | 0.87 | 0.89 | 0.49 | -0.42 | 0.85 | 0.68 | 0.56 | 0.20 | +0.35 |
| 14 | 0.57 | 0.86 | 0.87 | 0.47 | -0.43 | 0.95 | 0.62 | 0.44 | 0.13 | +0.40 |

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

| mini (opr30) | t | Pa / Pb | dropped | kept res | kept med | kept sign | kept corr | kept slope | drop res | drop med | drop sign | drop corr | drop slope | rows-only slope / corr | cols-only slope / corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 1024 / 1024 | 0 | 0.84 | 0.06 | 1.00 | 1.00 | 1.00 | - | - | - | - | - | - / - | - / - |
| 0 | 3 | 960 / 1024 | 65472 | 0.87 | 0.05 | 1.00 | 1.00 | 1.00 | 0.98 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 0 | 4 | 896 / 1024 | 130944 | 0.89 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 0 | 5 | 896 / 1024 | 130944 | 0.91 | 0.05 | 1.00 | 1.00 | 0.99 | 0.99 | 0.02 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 0 | 6 | 832 / 960 | 249634 | 0.92 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.97 / 1.00 |
| 0 | 7 | 832 / 960 | 249640 | 0.93 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.03 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.97 / 1.00 |
| 0 | 8 | 768 / 896 | 360136 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 0 | 9 | 832 / 896 | 302839 | 0.95 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 0 | 10 | 832 / 832 | 356059 | 0.95 | 0.06 | 1.00 | 1.00 | 0.99 | 0.98 | 0.05 | 1.00 | 1.00 | 0.97 | 0.99 / 1.00 | 0.96 / 1.00 |
| 0 | 11 | 768 / 896 | 360127 | 0.96 | 0.06 | 0.98 | 0.97 | 0.94 | 0.98 | 0.15 | 0.95 | 0.80 | 0.60 | 0.67 / 0.66 | 0.54 / 0.75 |
| 0 | 12 | 768 / 768 | 458371 | 0.96 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.13 | 0.94 | 0.83 | 0.59 | 0.69 / 0.74 | 0.59 / 0.77 |
| 0 | 13 | 704 / 704 | 552484 | 0.97 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.11 | 0.95 | 0.84 | 0.61 | 0.72 / 0.79 | 0.64 / 0.78 |
| 0 | 14 | 704 / 704 | 552472 | 0.97 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.19 | 0.92 | 0.80 | 0.53 | 0.66 / 0.78 | 0.55 / 0.73 |
| 1 | 2 | 960 / 1024 | 65472 | 0.82 | 0.06 | 1.00 | 1.00 | 1.00 | 0.96 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 1 | 3 | 960 / 1024 | 65472 | 0.87 | 0.05 | 1.00 | 1.00 | 1.00 | 0.98 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 1 | 4 | 896 / 1024 | 130944 | 0.89 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 1 | 5 | 896 / 1024 | 130944 | 0.91 | 0.05 | 1.00 | 1.00 | 0.99 | 0.99 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 1 | 6 | 832 / 960 | 249631 | 0.93 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.97 / 1.00 |
| 1 | 7 | 832 / 896 | 302849 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.97 / 1.00 |
| 1 | 8 | 832 / 896 | 302857 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 1 | 9 | 768 / 832 | 409243 | 0.95 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 1 | 10 | 768 / 832 | 409236 | 0.95 | 0.05 | 1.00 | 1.00 | 0.98 | 0.97 | 0.05 | 1.00 | 1.00 | 0.97 | 0.99 / 1.00 | 0.96 / 1.00 |
| 1 | 11 | 768 / 768 | 458356 | 0.96 | 0.05 | 1.00 | 1.00 | 0.98 | 0.98 | 0.14 | 0.94 | 0.84 | 0.60 | 0.69 / 0.76 | 0.60 / 0.77 |
| 1 | 12 | 768 / 768 | 458369 | 0.96 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.14 | 0.93 | 0.84 | 0.60 | 0.70 / 0.76 | 0.60 / 0.78 |
| 1 | 13 | 704 / 832 | 462421 | 0.96 | 0.06 | 0.99 | 0.99 | 0.96 | 0.98 | 0.14 | 0.94 | 0.82 | 0.60 | 0.70 / 0.79 | 0.54 / 0.72 |
| 1 | 14 | 704 / 704 | 552488 | 0.97 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.19 | 0.93 | 0.81 | 0.55 | 0.67 / 0.79 | 0.55 / 0.73 |
| 2 | 2 | 960 / 1024 | 65472 | 0.82 | 0.06 | 1.00 | 1.00 | 1.00 | 0.95 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 2 | 3 | 960 / 1024 | 65472 | 0.87 | 0.05 | 1.00 | 1.00 | 1.00 | 0.98 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 2 | 4 | 960 / 1024 | 65472 | 0.90 | 0.05 | 1.00 | 1.00 | 0.99 | 0.99 | 0.02 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | - / - |
| 2 | 5 | 896 / 960 | 188253 | 0.91 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.98 / 1.00 |
| 2 | 6 | 832 / 960 | 249629 | 0.93 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.97 / 1.00 |
| 2 | 7 | 832 / 896 | 302843 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.97 / 1.00 |
| 2 | 8 | 832 / 896 | 302834 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 2 | 9 | 768 / 832 | 409254 | 0.95 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.04 | 1.00 | 1.00 | 0.97 | 0.99 / 1.00 | 0.95 / 1.00 |
| 2 | 10 | 768 / 832 | 409254 | 0.96 | 0.05 | 1.00 | 1.00 | 0.98 | 0.98 | 0.05 | 1.00 | 1.00 | 0.97 | 0.98 / 1.00 | 0.95 / 1.00 |
| 2 | 11 | 768 / 768 | 458357 | 0.96 | 0.05 | 1.00 | 1.00 | 0.98 | 0.98 | 0.13 | 0.94 | 0.84 | 0.61 | 0.69 / 0.75 | 0.62 / 0.78 |
| 2 | 12 | 768 / 768 | 458367 | 0.96 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.14 | 0.94 | 0.83 | 0.59 | 0.68 / 0.73 | 0.59 / 0.77 |
| 2 | 13 | 704 / 704 | 552492 | 0.96 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.11 | 0.94 | 0.85 | 0.63 | 0.73 / 0.80 | 0.65 / 0.80 |
| 2 | 14 | 704 / 704 | 552476 | 0.97 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.20 | 0.93 | 0.81 | 0.55 | 0.67 / 0.79 | 0.56 / 0.73 |
| 3 | 2 | 960 / 1024 | 65472 | 0.83 | 0.06 | 1.00 | 1.00 | 1.00 | 0.96 | 0.04 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 3 | 3 | 960 / 1024 | 65472 | 0.88 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 3 | 4 | 896 / 1024 | 130944 | 0.90 | 0.05 | 1.00 | 1.00 | 0.99 | 0.99 | 0.02 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | - / - |
| 3 | 5 | 896 / 960 | 188256 | 0.92 | 0.04 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.98 / 1.00 |
| 3 | 6 | 832 / 960 | 249634 | 0.93 | 0.04 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.97 / 1.00 |
| 3 | 7 | 832 / 896 | 302851 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.03 | 1.00 | 1.00 | 0.99 | 0.99 / 1.00 | 0.97 / 1.00 |
| 3 | 8 | 832 / 896 | 302852 | 0.94 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 3 | 9 | 768 / 832 | 409254 | 0.95 | 0.05 | 1.00 | 1.00 | 0.99 | 0.97 | 0.04 | 1.00 | 1.00 | 0.98 | 0.99 / 1.00 | 0.96 / 1.00 |
| 3 | 10 | 768 / 832 | 409244 | 0.95 | 0.06 | 1.00 | 1.00 | 0.99 | 0.98 | 0.05 | 1.00 | 1.00 | 0.97 | 0.99 / 1.00 | 0.95 / 1.00 |
| 3 | 11 | 768 / 768 | 458367 | 0.96 | 0.05 | 1.00 | 1.00 | 0.99 | 0.98 | 0.13 | 0.94 | 0.83 | 0.59 | 0.68 / 0.73 | 0.60 / 0.77 |
| 3 | 12 | 768 / 768 | 458368 | 0.96 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.13 | 0.94 | 0.83 | 0.60 | 0.68 / 0.75 | 0.61 / 0.78 |
| 3 | 13 | 768 / 832 | 409239 | 0.96 | 0.06 | 0.99 | 0.99 | 0.97 | 0.98 | 0.17 | 0.93 | 0.80 | 0.55 | 0.66 / 0.73 | 0.52 / 0.71 |
| 3 | 14 | 768 / 704 | 507496 | 0.97 | 0.06 | 0.99 | 1.00 | 0.98 | 0.98 | 0.25 | 0.93 | 0.80 | 0.52 | 0.64 / 0.74 | 0.54 / 0.73 |

Diagonal t3, slope / corr against the truth:

| mini (opr30) | t | band off: in | band off: out | band on (submitted): in | band on: out | band on without s7: out |
|---|---|---|---|---|---|---|
| 0 | 2 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 0 | 3 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 0 | 4 | 0.99 / 1.00 | - | 0.99 / 1.00 | - | - |
| 0 | 5 | 0.99 / 1.00 | - | 0.99 / 1.00 | - | - |
| 0 | 6 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.98 / 1.00 | - |
| 0 | 7 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.94 / 1.00 | - |
| 0 | 8 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.86 / 1.00 | - |
| 0 | 9 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.81 / 1.00 | - |
| 0 | 10 | 0.99 / 1.00 | 0.97 / 1.00 | 0.99 / 1.00 | 0.75 / 0.99 | - |
| 0 | 11 | 0.86 / 0.94 | 0.59 / 0.97 | 0.86 / 0.94 | 0.43 / 0.96 | - |
| 0 | 12 | 0.98 / 1.00 | 0.59 / 0.90 | 0.98 / 1.00 | 0.40 / 0.89 | - |
| 0 | 13 | 0.99 / 1.00 | 0.62 / 0.88 | 0.99 / 1.00 | 0.40 / 0.86 | - |
| 0 | 14 | 0.99 / 1.00 | 0.54 / 0.89 | 0.98 / 1.00 | 0.31 / 0.87 | - |
| 1 | 2 | 0.99 / 1.00 | - | 0.99 / 1.00 | - | - |
| 1 | 3 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 1 | 4 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 1 | 5 | 0.99 / 1.00 | - | 1.00 / 1.00 | - | - |
| 1 | 6 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | - |
| 1 | 7 | 1.00 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 0.94 / 1.00 | - |
| 1 | 8 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.86 / 1.00 | - |
| 1 | 9 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.79 / 0.99 | - |
| 1 | 10 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.72 / 0.99 | - |
| 1 | 11 | 0.99 / 1.00 | 0.58 / 0.90 | 0.98 / 1.00 | 0.40 / 0.89 | - |
| 1 | 12 | 0.99 / 1.00 | 0.61 / 0.90 | 0.99 / 1.00 | 0.37 / 0.89 | - |
| 1 | 13 | 0.95 / 0.98 | 0.58 / 0.96 | 0.95 / 0.98 | 0.33 / 0.92 | - |
| 1 | 14 | 0.98 / 1.00 | 0.51 / 0.86 | 0.98 / 1.00 | 0.27 / 0.83 | - |
| 2 | 2 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 2 | 3 | 1.00 / 1.00 | - | 1.00 / 1.00 | - | - |
| 2 | 4 | 0.99 / 1.00 | - | 1.00 / 1.00 | - | - |
| 2 | 5 | 0.99 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | - |
| 2 | 6 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | - |
| 2 | 7 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.94 / 1.00 | - |
| 2 | 8 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.85 / 1.00 | - |
| 2 | 9 | 0.99 / 1.00 | 0.97 / 1.00 | 0.99 / 1.00 | 0.75 / 0.99 | - |
| 2 | 10 | 0.99 / 1.00 | 0.97 / 1.00 | 0.99 / 1.00 | 0.68 / 0.99 | - |
| 2 | 11 | 0.99 / 1.00 | 0.62 / 0.91 | 0.99 / 1.00 | 0.40 / 0.90 | - |
| 2 | 12 | 0.98 / 1.00 | 0.59 / 0.90 | 0.98 / 1.00 | 0.34 / 0.89 | - |
| 2 | 13 | 0.98 / 1.00 | 0.60 / 0.87 | 0.98 / 1.00 | 0.31 / 0.84 | - |
| 2 | 14 | 0.98 / 1.00 | 0.54 / 0.89 | 0.98 / 1.00 | 0.27 / 0.83 | - |
| 3 | 2 | 0.99 / 1.00 | - | 0.99 / 1.00 | - | - |
| 3 | 3 | 0.99 / 1.00 | - | 0.99 / 1.00 | - | - |
| 3 | 4 | 0.99 / 1.00 | - | 1.00 / 1.00 | - | - |
| 3 | 5 | 1.00 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | - |
| 3 | 6 | 1.00 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | - |
| 3 | 7 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.93 / 1.00 | - |
| 3 | 8 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.85 / 1.00 | - |
| 3 | 9 | 0.99 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.77 / 1.00 | - |
| 3 | 10 | 0.99 / 1.00 | 0.97 / 1.00 | 0.99 / 1.00 | 0.71 / 0.99 | - |
| 3 | 11 | 0.99 / 1.00 | 0.58 / 0.89 | 0.99 / 1.00 | 0.40 / 0.89 | - |
| 3 | 12 | 0.99 / 1.00 | 0.59 / 0.89 | 0.99 / 1.00 | 0.37 / 0.87 | - |
| 3 | 13 | 0.96 / 0.98 | 0.52 / 0.94 | 0.95 / 0.98 | 0.30 / 0.91 | - |
| 3 | 14 | 1.00 / 1.00 | 0.50 / 0.86 | 0.99 / 1.00 | 0.28 / 0.84 | - |

Band off, t3 is close to exact inside the column band (slope 0.95-1.00; 0.86 on mini 0 at t = 11) and outside it at
t = 5-10 (0.97-0.99); at t = 11-14 the out-of-band t3 is shrunk like the entries (0.50-0.62). With the band on, s7
degrades the out-of-band t3 from t = 7 (0.93-0.94) to 0.68-0.75 at t = 10 and 0.27-0.31 at t = 14, as in r38.

**Oracle swap** (B = wu2 v_engine_nott; entries of M1 replaced at every band layer, specs as in sections 4 and 6):

| run | mini 0 (x 1e-8) | mini 1 (x 1e-8) | mini 2 (x 1e-8) | mini 3 (x 1e-8) | / B mini 0 | / B mini 1 | / B mini 2 | / B mini 3 |
|---|---|---|---|---|---|---|---|---|
| a: band on, as submitted (wu2 v_engine) | 2.2533 | 2.3388 | 2.4933 | 2.4920 | 0.880 | 0.920 | 0.900 | 0.903 |
| B: band off (wu2 v_engine_nott) | 2.5604 | 2.5420 | 2.7715 | 2.7603 | 1.000 | 1.000 | 1.000 | 1.000 |
| z: dropped entries 0 (clean band) | 2.2349 | 2.3844 | 2.4150 | 2.4885 | 0.873 | 0.938 | 0.871 | 0.902 |
| m: dropped = Monte Carlo truth + fold | 2.5977 | 2.5793 | 2.8137 | 2.7656 | 1.015 | 1.015 | 1.015 | 1.002 |
| dmm: every off-diagonal entry = truth + fold | 2.1812 | 2.3243 | 2.7163 | 2.9694 | 0.852 | 0.914 | 0.980 | 1.076 |
| s090: band off, off-diagonal x 0.9 (layers 1-14) | 5.6555 | 5.4022 | 6.0242 | 5.9960 | 2.209 | 2.125 | 2.174 | 2.172 |
| s125: band off, off-diagonal x 1.25 | 45.3328 | 38.1982 | 37.9140 | 37.0437 | 17.706 | 15.027 | 13.680 | 13.420 |
| dd: band off, diagonal = Monte Carlo 3 t3 (layers 1-14) | 2.0606 | 2.0382 | 2.0072 | 2.2400 | 0.805 | 0.802 | 0.724 | 0.811 |
| mz: kept entries = truth + fold, dropped 0, diagonal exact | 1.8220 | 2.1106 | 2.4249 | 2.4617 | 0.712 | 0.830 | 0.875 | 0.892 |
| zlo: dropped entries 0 at layers 2-10 only | 2.2918 | 2.4116 | 2.4727 | 2.5179 | 0.895 | 0.949 | 0.892 | 0.912 |
| zhi: dropped entries 0 at layers 11-14 only | 2.4092 | 2.3704 | 2.5995 | 2.5863 | 0.941 | 0.932 | 0.938 | 0.937 |
| zrow: 0 only where the row is out (a_i < -1.5) | 3.0467 | 2.9416 | 3.0472 | 3.1342 | 1.190 | 1.157 | 1.099 | 1.135 |
| zcol: 0 only where the column is out (|a_j| >= 3) | 2.1505 | 2.3825 | 2.3414 | 2.4654 | 0.840 | 0.937 | 0.845 | 0.893 |
| ddmm: dd + every off-diagonal entry = truth + fold | 1.6283 | 1.8935 | 1.8330 | 2.1623 | 0.636 | 0.745 | 0.661 | 0.783 |
| ddmz: dd + kept = truth + fold, dropped 0 | 1.4032 | 1.8862 | 1.6678 | 1.8354 | 0.548 | 0.742 | 0.602 | 0.665 |

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

| mini | opr30 | r37 = opr30 + rank 256 + L4 | r38 with both halves of R38 undone (mini 0) | r38 without the extraction (law off) | r38 with the law back (extraction on) | r38 as submitted (law off, extraction on) |
|---|---|---|---|---|---|---|
| 0 | 0.880 | 0.882 | 0.882 | 0.885 | 1.074 | 0.973 |
| 1 | 0.920 | 0.950 | - | 0.900 | 1.088 | 1.016 |
| 2 | 0.900 | 0.891 | - | 0.922 | 1.023 | 0.989 |
| 3 | 0.903 | 0.883 | - | 0.865 | 1.080 | 0.977 |
| 4 | 0.903 | 0.918 | - | 0.892 | 1.070 | 0.991 |
| 5 | 0.968 | 0.974 | - | 0.887 | 1.101 | 1.021 |
| 6 | 0.974 | 0.994 | - | 0.904 | 1.092 | 1.024 |
| 7 | 0.863 | 0.867 | - | 0.841 | 1.059 | 0.927 |
| 8 | 0.957 | 0.939 | - | 0.935 | 1.040 | 0.971 |
| 9 | 0.888 | 0.886 | - | 0.889 | 1.033 | 0.953 |
| 10 | 0.910 | 0.897 | - | 0.900 | 1.011 | 0.965 |
| 11 | 0.981 | 0.959 | - | 0.917 | 1.073 | 1.024 |
| 12 | 0.934 | 0.942 | - | 0.907 | 1.043 | 0.973 |
| 13 | 0.925 | 0.925 | - | 0.900 | 1.087 | 1.022 |
| 14 | 0.873 | 0.891 | - | 0.855 | 1.057 | 0.928 |
| 15 | 0.921 | 0.919 | - | 0.908 | 1.066 | 0.986 |
| geometric mean of on / off | 0.918 | 0.919 | 0.882 | 0.894 | 1.062 | 0.983 |
| band on better | 16 of 16 | 16 of 16 | 1 of 1 | 16 of 16 | 0 of 16 | 11 of 16 |
| band-on MSE, geometric mean (x 1e-8) | 2.136 | 2.024 | 2.132 (1 net) | 4.345 | 3.521 | 1.924 |

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

| mini | r38 as submitted | without the C' part | without the dA (K4 rows) part | without the E (K3 leg) part | without the feed pair | the whole extraction at half strength | without the extraction (all four) |
|---|---|---|---|---|---|---|---|
| 0 | 0.973 | 0.999 | 0.946 | 1.022 | 0.949 | 0.894 | 0.885 |
| 1 | 1.016 | 0.975 | 0.914 | 1.039 | 0.982 | 0.919 | 0.900 |
| 2 | 0.989 | 0.995 | 0.893 | 1.032 | 0.969 | 0.925 | 0.922 |
| 3 | 0.977 | 0.999 | 0.913 | 1.030 | 0.943 | 0.891 | 0.865 |
| 4 | 0.991 | 0.990 | 0.923 | 1.029 | 0.969 | 0.909 | 0.892 |
| 5 | 1.021 | 0.996 | 0.891 | 1.021 | 0.987 | 0.943 | 0.887 |
| 6 | 1.024 | 0.991 | 0.925 | 1.040 | 0.995 | 0.945 | 0.904 |
| 7 | 0.927 | 0.996 | 0.919 | 1.027 | 0.916 | 0.872 | 0.841 |
| geometric mean of on / off | 0.989 | 0.992 | 0.915 | 1.030 | 0.964 | 0.912 | 0.887 |
| band on better | 5 of 8 | 8 of 8 | 8 of 8 | 0 of 8 | 8 of 8 | 8 of 8 | 8 of 8 |
| band-on MSE, geometric mean (x 1e-8) | 1.890 | 45.019 | 12.272 | 35.295 | 1.946 | 2.477 | 4.221 |

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

| t | opr30 | r37 = opr30 + rank 256 + L4 | r38 without the extraction (law off) | r38 with the law back (extraction on) | r38 as submitted |
|---|---|---|---|---|---|
| 2 | 1.00 / 1.00 (1.00) | 1.00 / 1.00 (1.00) | 1.00 / 0.99 (1.00) | 0.92 / 0.86 (0.99) | 0.92 / 0.86 (0.99) |
| 3 | 1.00 / 1.00 (1.00) | 1.00 / 1.00 (1.00) | 0.99 / 0.99 (1.00) | 0.87 / 0.80 (0.99) | 0.87 / 0.80 (0.99) |
| 4 | 0.99 / 0.99 (1.00) | 0.99 / 0.99 (1.00) | 0.99 / 0.99 (1.00) | 0.81 / 0.74 (0.99) | 0.81 / 0.74 (0.99) |
| 5 | 0.99 / 0.99 (1.00) | 0.99 / 0.99 (1.00) | 0.99 / 0.99 (1.00) | 0.77 / 0.69 (0.98) | 0.77 / 0.69 (0.98) |
| 6 | 0.99 / 0.99 (1.00) | 0.99 / 0.99 (1.00) | 0.99 / 0.98 (1.00) | 0.73 / 0.65 (0.97) | 0.73 / 0.65 (0.97) |
| 7 | 0.99 / 0.98 (1.00) | 0.99 / 0.98 (1.00) | 0.99 / 0.98 (1.00) | 0.69 / 0.62 (0.96) | 0.69 / 0.61 (0.96) |
| 8 | 0.99 / 0.98 (1.00) | 0.99 / 0.98 (1.00) | 0.98 / 0.97 (1.00) | 0.65 / 0.58 (0.95) | 0.65 / 0.57 (0.95) |
| 9 | 0.99 / 0.97 (1.00) | 0.99 / 0.98 (1.00) | 0.98 / 0.97 (1.00) | 0.62 / 0.55 (0.94) | 0.61 / 0.54 (0.94) |
| 10 | 0.98 / 0.97 (1.00) | 0.99 / 0.97 (1.00) | 0.98 / 0.96 (1.00) | 0.59 / 0.51 (0.93) | 0.58 / 0.51 (0.93) |
| 11 | 0.97 / 0.60 (0.83) | 0.97 / 0.60 (0.83) | 0.97 / 0.59 (0.83) | 0.54 / 0.21 (0.57) | 0.54 / 0.21 (0.57) |
| 12 | 0.98 / 0.59 (0.83) | 0.98 / 0.60 (0.83) | 0.98 / 0.59 (0.83) | 0.52 / 0.19 (0.55) | 0.52 / 0.19 (0.55) |
| 13 | 0.97 / 0.60 (0.83) | 0.98 / 0.60 (0.83) | 0.97 / 0.59 (0.83) | 0.50 / 0.20 (0.57) | 0.49 / 0.20 (0.56) |
| 14 | 0.98 / 0.54 (0.80) | 0.98 / 0.54 (0.80) | 0.97 / 0.53 (0.80) | 0.48 / 0.13 (0.45) | 0.47 / 0.13 (0.44) |

Diagonal t3, slope inside / outside the column band:

| t | opr30 | r37 = opr30 + rank 256 + L4 | r38 without the extraction (law off) | r38 with the law back (extraction on) | r38 as submitted |
|---|---|---|---|---|---|
| 2 | 1.00 / - | 1.00 / - | 1.00 / - | 0.88 / - | 0.88 / - |
| 3 | 1.00 / - | 1.00 / - | 0.99 / - | 0.81 / - | 0.81 / - |
| 4 | 0.99 / - | 0.99 / - | 0.99 / - | 0.75 / - | 0.75 / - |
| 5 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.98 | 0.71 / 0.65 | 0.71 / 0.65 |
| 6 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.98 | 0.67 / 0.60 | 0.67 / 0.60 |
| 7 | 0.99 / 0.98 | 0.99 / 0.98 | 0.99 / 0.98 | 0.64 / 0.57 | 0.64 / 0.57 |
| 8 | 0.99 / 0.98 | 0.99 / 0.98 | 0.99 / 0.97 | 0.61 / 0.52 | 0.60 / 0.52 |
| 9 | 0.99 / 0.98 | 0.99 / 0.98 | 0.99 / 0.97 | 0.58 / 0.49 | 0.58 / 0.49 |
| 10 | 0.99 / 0.97 | 0.99 / 0.98 | 0.98 / 0.96 | 0.55 / 0.45 | 0.55 / 0.45 |
| 11 | 0.96 / 0.59 | 0.96 / 0.59 | 0.95 / 0.59 | 0.51 / 0.24 | 0.51 / 0.24 |
| 12 | 0.99 / 0.59 | 0.99 / 0.59 | 0.98 / 0.59 | 0.51 / 0.23 | 0.50 / 0.23 |
| 13 | 0.97 / 0.58 | 0.97 / 0.58 | 0.96 / 0.58 | 0.51 / 0.23 | 0.50 / 0.23 |
| 14 | 0.99 / 0.52 | 0.99 / 0.52 | 0.98 / 0.52 | 0.48 / 0.18 | 0.48 / 0.17 |

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

| t | opr30 | r37 | r38 without the extraction | r38 with the law back | r38 |
|---|---|---|---|---|---|
| 2 | 1.10 / - / 1.16 | 1.10 / - / 1.16 | 1.10 / - / 1.16 | 1.02 / - / 1.04 | 1.02 / - / 1.04 |
| 3 | 1.09 / 1.17 / 1.15 | 1.09 / 1.17 / 1.15 | 1.09 / 1.17 / 1.14 | 0.96 / 0.94 / 0.95 | 0.96 / 0.94 / 0.95 |
| 4 | 1.09 / 1.14 / 1.14 | 1.09 / 1.14 / 1.14 | 1.09 / 1.14 / 1.13 | 0.90 / 0.86 / 0.86 | 0.90 / 0.86 / 0.86 |
| 5 | 1.08 / 1.13 / 1.12 | 1.08 / 1.13 / 1.12 | 1.07 / 1.13 / 1.12 | 0.86 / 0.79 / 0.80 | 0.86 / 0.79 / 0.80 |
| 6 | 1.07 / 1.10 / 1.11 | 1.07 / 1.10 / 1.11 | 1.07 / 1.10 / 1.10 | 0.81 / 0.74 / 0.74 | 0.81 / 0.74 / 0.74 |
| 7 | 1.07 / 1.09 / 1.10 | 1.07 / 1.09 / 1.10 | 1.06 / 1.09 / 1.10 | 0.78 / 0.70 / 0.70 | 0.77 / 0.70 / 0.70 |
| 8 | 1.07 / 1.08 / 1.09 | 1.07 / 1.08 / 1.09 | 1.06 / 1.08 / 1.09 | 0.73 / 0.66 / 0.64 | 0.73 / 0.66 / 0.64 |
| 9 | 1.06 / 1.07 / 1.08 | 1.06 / 1.07 / 1.08 | 1.06 / 1.06 / 1.08 | 0.69 / 0.63 / 0.62 | 0.69 / 0.62 / 0.61 |
| 10 | 1.06 / 1.06 / 1.08 | 1.06 / 1.06 / 1.08 | 1.05 / 1.05 / 1.07 | 0.67 / 0.59 / 0.58 | 0.67 / 0.59 / 0.57 |
| 11 | 1.01 / 0.65 / 0.79 | 1.01 / 0.66 / 0.80 | 1.00 / 0.65 / 0.79 | 0.60 / 0.25 / 0.40 | 0.60 / 0.25 / 0.40 |
| 12 | 1.06 / 0.64 / 0.76 | 1.06 / 0.64 / 0.77 | 1.05 / 0.64 / 0.76 | 0.59 / 0.24 / 0.36 | 0.59 / 0.24 / 0.36 |
| 13 | 1.05 / 0.67 / 0.76 | 1.06 / 0.67 / 0.77 | 1.05 / 0.66 / 0.76 | 0.56 / 0.25 / 0.36 | 0.56 / 0.25 / 0.35 |
| 14 | 1.05 / 0.58 / 0.68 | 1.05 / 0.58 / 0.69 | 1.04 / 0.57 / 0.68 | 0.54 / 0.18 / 0.30 | 0.54 / 0.18 / 0.29 |

**The clean band and the truth on the dropped entries in each file** (minis 0, 1, 2, 3; each over its own B):

| file | band on / off (minis 0, 1, 2, 3) | z / B | m / B |
|---|---|---|---|
| opr30 | 0.880, 0.920, 0.900, 0.903 | 0.873, 0.938, 0.871, 0.902 | 1.015, 1.015, 1.015, 1.002 |
| r37 = opr30 + rank 256 + L4 | 0.882, 0.950, 0.891, 0.883 | 0.871, 0.983, 0.862, 0.878 | 1.034, 1.017, 1.015, 1.012 |
| r38 without the extraction (law off) | 0.885, 0.900, 0.922, 0.865 | 0.848, 0.878, 0.886, 0.820 | 1.017, 1.019, 1.024, 1.021 |
| r38 with the law back (extraction on) | 1.074, 1.088, 1.023, 1.080 | 1.075, 1.085, 1.015, 1.063 | 0.969, 0.990, 1.031, 0.941 |
| r38 as submitted | 0.973, 1.016, 0.989, 0.977 | 0.950, 1.025, 0.981, 0.984 | 1.156, 1.148, 1.182, 1.165 |

The clean band moves the MSE as the submitted band does in every file (z / B: r37 x0.862-0.983, r38 without the
extraction x0.820-0.886, r38 with the law back x1.015-1.085). The truth on the dropped entries costs a little where
the engine's entries are close to the Gaussian-input truth (opr30 x1.002-1.015, r37 x1.012-1.034, r38 without the
extraction x1.017-1.024), much in r38 (x1.148-1.182), and is mixed in r38 with the law back (x0.941-1.031).

**Which dropped entries carry the gain** (the clean band split by why an entry is dropped: zrow zeroes only the entries
whose single index is out of the band, a_i < -1.5; zcol only those whose doubled index is out, |a_j| >= 3; minis
0-3, each over its own B):

| file | z / B (minis 0, 1, 2, 3) | zrow / B | zcol / B |
|---|---|---|---|
| opr30 | 0.873, 0.938, 0.871, 0.902 | 1.190, 1.157, 1.099, 1.135 | 0.840, 0.937, 0.845, 0.893 |
| r38 without the extraction's dA part | 0.934, 0.901, 0.873, 0.906 | 1.073, 1.069, 1.046, 1.071 | 0.874, 0.840, 0.838, 0.848 |
| r38 as submitted | 0.950, 1.025, 0.981, 0.984 | 1.066, 1.053, 1.049, 1.027 | 0.932, 1.008, 0.964, 0.989 |

The pattern is the same in all three: the entries dropped for their doubled index carry the gain and those dropped for
their single index cost. What the dA part changes is the size of the first. Zeroing only the column drops gives
x0.840-0.937 in opr30 and x0.838-0.874 in r38 without only the dA part, but x0.932-1.008 in r38; zeroing only the row
drops costs x1.099-1.190, x1.046-1.073 and x1.027-1.066.

**Does the band do the extraction's job?** (mech_overlap.py: final-layer means of each file's band-off engine B and
band-on engine P on minis 0-3 against the dataset's means; e = truth - B is the band-off error and d = P - B what the
band changes, so the band changes the MSE by (|d|^2 - 2 d.e) / n.)

The band's change, file by file (on / off from the recorded means; cos(d, e) > 0: the band moves the means toward the
truth):

| file | on / off (minis 0, 1, 2, 3) | cos(d, e) | rms d / rms e |
|---|---|---|---|
| opr30 | 0.880, 0.920, 0.900, 0.903 | 0.37, 0.28, 0.33, 0.32 | 0.249, 0.275, 0.231, 0.248 |
| r37 | 0.882, 0.950, 0.891, 0.883 | 0.36, 0.23, 0.34, 0.36 | 0.255, 0.284, 0.254, 0.259 |
| noext | 0.885, 0.900, 0.922, 0.865 | 0.45, 0.36, 0.32, 0.49 | 0.155, 0.189, 0.170, 0.167 |
| lawon | 1.074, 1.088, 1.023, 1.080 | -0.12, -0.15, 0.02, -0.17 | 0.176, 0.183, 0.170, 0.160 |
| noxa | 0.946, 0.914, 0.893, 0.913 | 0.40, 0.49, 0.62, 0.56 | 0.075, 0.098, 0.093, 0.083 |
| r38 | 0.973, 1.016, 0.989, 0.977 | 0.17, 0.10, 0.14, 0.16 | 0.212, 0.266, 0.232, 0.219 |

How alike the band's changes are across files (cosine of d):

| band change of | opr30 | r37 | noext | lawon | noxa | r38 |
|---|---|---|---|---|---|---|
| opr30 | 1 | 0.92-0.94 | 0.90-0.92 | 0.87-0.89 | 0.83-0.89 | 0.86-0.90 |
| r37 | 0.92-0.94 | 1 | 0.90-0.93 | 0.88-0.89 | 0.86-0.91 | 0.87-0.91 |
| noext | 0.90-0.92 | 0.90-0.93 | 1 | 0.88-0.90 | 0.87-0.91 | 0.88-0.90 |
| lawon | 0.87-0.89 | 0.88-0.89 | 0.88-0.90 | 1 | 0.88-0.91 | 0.88-0.91 |
| noxa | 0.83-0.89 | 0.86-0.91 | 0.87-0.91 | 0.88-0.91 | 1 | 0.88-0.91 |
| r38 | 0.86-0.90 | 0.87-0.91 | 0.88-0.90 | 0.88-0.91 | 0.88-0.91 | 1 |

Transfer: one file's band change put on another file's band-off means (MSE over that file's band-off MSE, geometric
mean over minis 0-3; the diagonal is each file's own band on / off):

| band change of (row), applied to (column) | opr30 | r37 | noext | lawon | noxa | r38 |
|---|---|---|---|---|---|---|
| opr30 | 0.901 | 0.904 | 0.884 | 1.091 | 0.885 | 0.994 |
| r37 | 0.908 | 0.901 | 0.884 | 1.096 | 0.884 | 0.996 |
| noext | 0.922 | 0.922 | 0.892 | 1.097 | 0.898 | 1.004 |
| lawon | 0.924 | 0.924 | 0.909 | 1.066 | 0.916 | 0.986 |
| noxa | 0.929 | 0.927 | 0.909 | 1.075 | 0.916 | 0.991 |
| r38 | 0.929 | 0.931 | 0.915 | 1.071 | 0.917 | 0.989 |

Splits. Between two band-off engines e splits exactly into the second engine's change X and the error it leaves: for
the engine without the extraction (noext) X = B(r38) - B(noext), the extraction's own change, and e = X + e38. The
band's MSE change then splits into a part along X, a part along the error left, and |d|^2. Rows: noext by the whole
extraction; r37 by the extraction at the law on (X = B(lawon) - B(r37)) and by all of R38; opr30 by everything from
opr30 to r38; r38 without the extraction's dA part (noxa) by the dA part's own change (X = B(r38) - B(noxa)); noext
by the other three parts (X = B(noxa) - B(noext)). Values x 1e-9; negative = the band lowers the MSE.

| mini | split | band MSE change | along the change | along the error left | d^2 | cos(d, change) | cos(d, error left) | cos(change, e) | rms change / rms e |
|---|---|---|---|---|---|---|---|---|---|
| 0 | noext by extraction (law off) | -6.563 | -6.000 | -1.938 | 1.374 | 0.515 | 0.179 | 0.803 | 0.658 |
| 0 | r37 by extraction (law on) | -2.862 | -6.123 | +1.684 | 1.578 | 0.496 | -0.120 | 0.358 | 0.999 |
| 0 | r37 by R38 | -2.862 | -2.052 | -2.388 | 1.578 | 0.575 | 0.206 | 0.345 | 0.289 |
| 0 | opr30 by opr30 -> r38 | -3.071 | -2.043 | -2.619 | 1.591 | 0.494 | 0.225 | 0.417 | 0.324 |
| 0 | noxa by the dA part | -9.433 | -9.041 | -1.359 | 0.967 | 0.413 | 0.149 | 0.942 | 0.843 |
| 0 | noext by the extraction without its dA part | -6.563 | +6.766 | -14.704 | 1.374 | -0.315 | 0.475 | -0.242 | 1.214 |
| 1 | noext by extraction (law off) | -5.025 | -5.998 | -0.825 | 1.798 | 0.436 | 0.070 | 0.784 | 0.724 |
| 1 | r37 by extraction (law on) | -1.156 | -5.872 | +2.861 | 1.855 | 0.426 | -0.170 | 0.220 | 1.056 |
| 1 | r37 by R38 | -1.156 | -2.022 | -0.989 | 1.855 | 0.516 | 0.082 | 0.402 | 0.300 |
| 1 | opr30 by opr30 -> r38 | -2.033 | -2.690 | -1.261 | 1.918 | 0.492 | 0.103 | 0.492 | 0.392 |
| 1 | noxa by the dA part | -12.356 | -12.681 | -1.057 | 1.382 | 0.521 | 0.102 | 0.932 | 0.865 |
| 1 | noext by the extraction without its dA part | -5.025 | +6.940 | -13.763 | 1.798 | -0.270 | 0.428 | -0.009 | 1.355 |
| 2 | noext by extraction (law off) | -4.141 | -4.315 | -1.352 | 1.526 | 0.335 | 0.120 | 0.781 | 0.717 |
| 2 | r37 by extraction (law on) | -2.789 | -4.124 | -0.326 | 1.661 | 0.306 | 0.021 | 0.332 | 1.032 |
| 2 | r37 by R38 | -2.789 | -2.585 | -1.865 | 1.661 | 0.558 | 0.159 | 0.445 | 0.355 |
| 2 | opr30 by opr30 -> r38 | -2.781 | -2.673 | -1.589 | 1.481 | 0.495 | 0.143 | 0.507 | 0.421 |
| 2 | noxa by the dA part | -14.070 | -14.018 | -1.184 | 1.132 | 0.675 | 0.122 | 0.920 | 0.852 |
| 2 | noext by the extraction without its dA part | -4.141 | +12.350 | -18.017 | 1.526 | -0.559 | 0.637 | 0.012 | 1.231 |
| 3 | noext by extraction (law off) | -7.920 | -7.569 | -1.980 | 1.629 | 0.538 | 0.165 | 0.793 | 0.720 |
| 3 | r37 by extraction (law on) | -3.008 | -7.946 | +3.216 | 1.721 | 0.541 | -0.191 | 0.279 | 1.105 |
| 3 | r37 by R38 | -3.008 | -2.756 | -1.974 | 1.721 | 0.657 | 0.160 | 0.378 | 0.315 |
| 3 | opr30 by opr30 -> r38 | -2.683 | -2.583 | -1.804 | 1.704 | 0.486 | 0.147 | 0.449 | 0.388 |
| 3 | noxa by the dA part | -13.676 | -13.018 | -1.742 | 1.084 | 0.567 | 0.178 | 0.929 | 0.877 |
| 3 | noext by the extraction without its dA part | -7.920 | +10.908 | -20.457 | 1.629 | -0.478 | 0.638 | -0.140 | 1.166 |

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
