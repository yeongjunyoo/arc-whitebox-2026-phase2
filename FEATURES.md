# The corrector of submission 336955: code reference

`estimator.py` below is `estimators/k3_fast_aug_sc_2l_..._opr30/estimator.py`. Drafted from the code by a GPT model and checked by us against the code (two tap exponents and the FLOP count corrected).

This describes the corrector of the graded `estimator.py` together with its training path. The estimator has 16 weight matrices, so its corrector updates layers `t=1..15`; layer 0 initializes zero hidden state and zero correction. Line references are to the files of this repository. Sigma powers: in the estimator `SV[k] = sigma^(k-4)` for k = 0..8 (`_EXP2` and `HE`).

## 1. Inputs per layer

### Training feature vector and indexing

`lc3_wmsg.load_mlp` concatenates 14 base features, the 16 neighbour features that remain after `DROP = ["soffd"]`, and `t/15`; the result is transposed to `(15,n,F)` (training code `research/lc3_wmsg.py:50-63`, `research/r25a/gru_lean.py:80-85`). Thus the learned feature vector `X_t` has 31 columns. `WMsg.forward` standardizes all columns as `(X-mu)/sd` before adding two message columns; `mu` is the featurewise training mean and `sd=std+1e-6` (`research/lc3_wmsg.py:66-80`, `research/r25a/gru_lean.py:283-287`). The fitted mean and scale themselves are folded into the estimator's projections; they are not stored as `mu` and `sd` arrays in the shipped archive.

| Training column(s) | Feature | Exact construction / symbols | Normalization and inference counterpart |
|---|---|---|---|
| 0 | `a` | `m_t/sigma_t`, where `m` is the preactivation mean and `sigma=s` its standard deviation. | Standardized by training `mu/sd`; formed inside inference feature fold `FB` (`research/lc_chain.py:61-69`, `estimator.py:2546-2552`). |
| 1 | `s` | `sigma_t`. | Standardized by `mu/sd`; inference has sigma in `SVB` and feature projections (`research/lc_chain.py:61-69`, `estimator.py:2547-2552`). |
| 2-3 | `Phi`, `phi` | Standard normal CDF and density at `a`. | Standardized by `mu/sd`; `QB` stores `phi,Phi` (`research/lc_chain.py:63-69`, `estimator.py:2532-2533`). |
| 4-5 | `k3`, `k4` | `t3/sigma^3`, `d4/sigma^4`; `t3,d4` are the engine's third and fourth preactivation cumulants. | Standardized by `mu/sd`; inference derives them through the `FB` projection (`research/lc_chain.py:65-68`, `estimator.py:2546-2552`). |
| 6 | `kA` | Cached fourth-law coefficient `kA` (training cache quantity; its internal interpretation is not needed by the feature builder). | Standardized by `mu/sd`; inference source is an engine feature row, folded through `fT` (`research/lc_chain.py:65-69`, `estimator.py:2549-2552`). |
| 7-12 | `R3n`, `R4n`, `E33n`, `E34n`, `E44n`, `W5n` | Respectively `w[3]*t3/(6*sigma)`, `w[4]*d4/(24*sigma)`, `w[6]*t3^2/(72*sigma)`, `w[7]*t3*d4/(144*sigma)`, `w[8]*d4^2/(1152*sigma)`, and `w[5]*sigma^4`; `w[k]` are the Wick readout coefficients returned by `lc_fit.wick(a,sigma)`. | Standardized by `mu/sd`; generated in inference by the `hT/kT/fT` table contractions (`research/lc_chain.py:63-69`, `estimator.py:2548-2552`). |
| 13 | `p` | Base engine readout mean divided by `sigma`, `pred_t/sigma_t`. | Standardized by `mu/sd`; source is the engine mean row (`research/lc_chain.py:65-69`, `estimator.py:2546-2552`). |
| 14-29 | Neighbour features (listed next) | One feature for each `NB` name except `soffd`, in the order in `research/lc3_curve.py:48-49`. | Standardized by `mu/sd`. Inference computes power-weighted sums as `VV -> SS -> AT`; its source excludes `soffd` (`research/lc3_wmsg.py:50-60`, `estimator.py:2553-2592`). |
| 14 | `dvar` | `S2/sigma_t^2`, where `S2=(W_t∘W_t)^T k2_{t-1}`. | Same standardization; inference's first neighbour sum and `b2/FAC` fold (`research/lc3_curve.py:137-154`, `estimator.py:2564-2592`). |
| 15 | `neff` | `log(S2^2/Q4)/log(n)`, `Q4=(W_t∘W_t∘W_t∘W_t)^T(k2_{t-1}^2)`. | Same standardization; inference uses `AT` log channels and the `SS` powers (`research/lc3_curve.py:139-154`, `estimator.py:2564-2592`). |
| 16-17 | `k3d`, `k3o` | `S3/sigma_t^3`; `t3_t/sigma_t^3 - S3/sigma_t^3`, with `S3=(W_t^3)^T k3_{t-1}` (elementwise power). | Same standardization; inference builds the corresponding `W^3` weighted neighbour sum then applies powers of sigma (`research/lc3_curve.py:139-154`, `estimator.py:2564-2592`). |
| 18-19 | `k4d`, `k4o` | `S4/sigma_t^4`; `d4_t/sigma_t^4-S4/sigma_t^4`, with `S4=(W_t^4)^T k4_{t-1}`. | Same standardization; `k4o` is residual to the streamed fourth-cumulant contribution (`research/lc3_curve.py:139-154`, `estimator.py:2564-2592`). |
| 20-21 | `k5d`, `k6d` | `S5/sigma_t^5`, `S6/sigma_t^6`, with `S5=(W_t^5)^T k5_{t-1}`, `S6=(W_t^6)^T k6_{t-1}`. | Same standardization; powers and scaling are in `SS/FAC/AT` (`research/lc3_curve.py:139-154`, `estimator.py:2564-2592`). |
| 22-23 | `R5n`, `R6n` | `w[5]*S5/(120*sigma_t)` and `w[6]*S6/(720*sigma_t)`. | Same standardization; `FB[1:,11:13]` is additionally multiplied by `sigma^-5` and `sigma^-6` in the inference fold (`research/lc3_curve.py:143-154`, `estimator.py:2585-2592`). |
| 24-27 | `xa`, `xR4n`, `xk4`, `xPhi` | `(W_t^2)^T a_{t-1}/cn2`; `(W_t^2)^T pR4n_{t-1}/cn2`; `(W_t^2)^T (d4_{t-1}/sigma_{t-1}^4)/cn2`; `(W_t^2)^T Phi(a_{t-1})/cn2`; `cn2=sum_i W_{it}^2`. At `t=1`, preceding activation has `a=0`, `Phi=.5`, and the prior correction/readout terms are initialized as specified in the training builder. | Same standardization; `AT` combines `SS` with inverse `cn2` factors. (`research/lc3_curve.py:125-155`, `estimator.py:2553-2592`). |
| 28-29 | `cn2`, `cn4` | `cn2=sum_i W_{it}^2`; `cn4=(sum_i W_{it}^4)/cn2^2 * n`. | Same standardization; `cn2` is also the denominator for columns 24-27 (`research/lc3_curve.py:142-154`, `estimator.py:2585-2592`). |
| 30 | depth | `t/15` for `t=1..15`. | Standardized with the same learned `mu/sd`. It is a constant across neurons in a layer (`research/lc3_wmsg.py:53-63`). |

The source enum and formulas for all neighbour features are `research/lc3_curve.py:48-49,120-156`. At layer 1, the builder uses `W[0]` to set prior `sigma`, `a=0`, `Phi=.5`, and zero prior cumulant/readout residual terms (`research/lc3_curve.py:125-136`). The inference table computes ReLU cumulants of a Gaussian using `Phi,phi,a,var`, with the coefficient tensor `relu`, and sums them through elementwise powers of `W` (`estimator.py:2558-2563`).

### Structure taps appended by the shipped v2 model

The requested training command selects QX tap IDs `0,1,2,4,5,6,7,8,9,10,11`; the estimator's `_QX` mapping says these IDs occupy QX rows `0..10` respectively (`estimator.py:1653`). The base loader first orders base columns 0-13, neighbour columns 14-29, then depth column 30; the QX wrapper appends taps after that, producing QX columns 31-41 (`research/lc3_wmsg.py:53-60`, `research/r25a/gru_lean.py:45-60`). Each QX feature is transformed as `arcsinh(QX[t,id]/qxc[t,id])`; `qxc` is the training-pool median of nonzero absolute values per layer and tap, and then the transformed values are included in the same featurewise `mu/sd` fit (`research/r25a/gru_lean.py:45-60,63-75,270-285`). Inference applies the stored per-layer `qxc` before `wqx` (`estimator.py:2595-2598`).

| Appended column | Tap ID / name | Definition in estimator | Training / inference locations |
|---|---|---|---|
| 31 | 0: star diagonal ratio | Prefer `DGF_t/LAM_t` when available and nonzero; fallback `sum_s,i G[s,j,i]*H[s,j,i]^3*W3F[s,j]`. | `estimator.py:3327-3333`; training cache appends selected QX with arcsinh (`research/r25a/gru_lean.py:53-60`). |
| 32 | 1: G-H shape | `sum_s,i (G*H)[s,j,i] * H[s,j,i] * LAMG[s,j]`. | `estimator.py:3334-3343`; training transform as above. |
| 33 | 2: G-G shape | `sum_s,i (G*H)[s,j,i] * G[s,j,i] * LAMG[s,j]`. | `estimator.py:3334-3343`; training transform as above. |
| 34 | 4: T row contraction | `sum_k M1[j,k]^2/sigma_k^4 * sigma_j^-2 - (diag(M1)_j)^2/sigma_j^6`, where `M1` is the engine's `3T^T` matrix and `SV` contains sigma powers. | `estimator.py:3348-3354`; selected tap is arcsinh scaled during training. |
| 35 | 5: T column contraction | `sum_i M1[i,k]^2/sigma_i^2 * sigma_k^-4 - (diag(M1)_k)^2/sigma_k^6`. | `estimator.py:3348-3354`; same training transform. |
| 36 | 6: compressed-leg energy carry | Previous compression residual propagated as `(W_t∘W_t)^T QRE`; this is an implementation residual quantity, not the full uncompressed energy. | `estimator.py:3344-3348`, initialization at `2800-2803`, update at `3521-3523`; same training transform. |
| 37 | 7: diagonal preactivation correlation residual | `sum_a M0[j,a]^2/sigma_a^2 * sigma_j^-2 - (diag(M0)_j/sigma_j^2)^2`, with `M0` the engine covariance-like matrix. | `estimator.py:3355-3357`; same training transform. |
| 38 | 8: signed T column sum | `sum_i M1[i,k]*sigma_i^-1 * sigma_k^-2 - diag(M1)_k*sigma_k^-3`. | `estimator.py:3358-3359`; same training transform. |
| 39 | 9: off-diagonal T energy | `sum_i M1[i,k]^2 * w11_i^2 * sigma_i^-2 * sigma_k^-4` (the code uses `VW11s[t]` and the corresponding sigma-power rows). | `estimator.py:3430-3432`; same training transform. |
| 40 | 10: signed preactivation correlation sum | `sum_a M0[j,a]*sigma_a^-1 * sigma_j^-1 - diag(M0)_j*sigma_j^-2`. | `estimator.py:3360-3361`; same training transform. |
| 41 | 11: carried tap-5 energy | `(W_t∘W_t)^T QX[t-1,5]`. | `estimator.py:3362-3363`; same training transform. |

`G,H,LAMG,DGF,M0,M1` denote internal estimator source legs, source coefficients, and contractions; their full probabilistic interpretation is not uniquely recoverable from the tap expressions alone. The code does deterministically define the expressions above. Training taps are caches of the shipped estimator output, not independently reimplemented formulas (`research/r25a/gru_dump.py:115-123,187-193`). Tap 3 is computed by the estimator (`estimator.py:3467-3474`) but is not selected in the shipped training command or the `_QX` index map.

### Two recurrent message columns (not part of `X`)

At `t>1`, training defines `push_t=W_t^T dp_{t-1}`, `m1=push_t/sigma_t`, `m2=(W_t∘W_t)^T|dp_{t-1}|/sigma_t`, and the hidden carry `Hm=W_t^T H_{t-1}/sqrt(2)`; the GRU input concatenates standardized `X_t`, `1e3*m1`, and `1e3*m2` (`research/lc3_wmsg.py:85-103`). At `t=1`, both messages and push are zero and `Hm=0` (`research/lc3_wmsg.py:82-90`). In inference the same raw message numerators are built at `estimator.py:2615-2623`; the sigma scales and `1e3` message gain are folded into `msc`/`wbig`, and hidden carry scaling into that recurrent projection. There are no standalone `m1`, `m2`, or `sqrt(2)` constants in the packed arrays. The precise folded arithmetic can be recovered only as the products shown in `_wmsg`, not as a named scale array (`estimator.py:2599,2623`).

## 2. Recurrence and output

The training recurrence is an ordinary PyTorch `GRUCell` with one shared 32-unit state per neuron; weights are shared across layers, while input projections and the chain coefficient are layer-specific (`research/lc3_wmsg.py:66-76,85-105`). In equations:

```text
push_t = W_t^T dp_(t-1)
Hm_t   = W_t^T H_(t-1) / sqrt(2)
H_t    = GRUCell([standardized X_t, 1e3 push_t/sigma_t,
                  1e3 (W_t∘W_t)^T |dp_(t-1)|/sigma_t], Hm_t)
u_t    = 1e-4 sigma_t * head(H_t)
dp_t   = D_t * push_t + u_t
```

At `t=1`, `push=0`, `Hm=0` and `dp_0=0`; it therefore has `dp_1=u_1`. This is explicit in the training loop (`research/lc3_wmsg.py:82-103`). The estimator's algebraically folded GRU computes gate projections in `PRE`, uses `wbig` over current messages, constants, and the 32-wide carried state, applies tanh/update operations, and then combines the output and carry (`estimator.py:2593-2630`). `cm` encodes the update mix; `wbig` is the projection for the packed `m_in`/`SC` path (`estimator.py:2603-2630`).

The local head is a linear scalar output and is multiplied by `1e-4*sigma` in training (`research/lc3_wmsg.py:72-75,100-103`). In inference, `CO` projects two rows (`FB[:,15:17]`) into per-layer `COs`; its contraction with the concatenated incoming `push` and 32 state values computes the chain contribution plus the local head output. The estimator comment states the decomposition as `D push + 1e-4 sigma (w . h' + b)` (`estimator.py:2599-2600,2610-2614,2629-2630`).

`D_t` is the fixed-engine derivative of the mean readout with respect to preactivation mean, holding sigma and cumulants fixed:

```text
D_t = w_1 + w_4*t3_t/6 + w_5*d4_t/24
```

Here `w_k` is the kth Gaussian-ReLU Wick coefficient; equivalently this is the derivative of `w_0 + w_3*t3/6 + w_4*d4/24`. The training cache computes exactly this `D` from `lf.wick` (`research/lc3_curve.py:197-201`), while the inference's last two `FB` rows are contracted by the learned fold to form `D push` (`estimator.py:2599-2600,2629-2630`). The `D` values are not separately serialized. The final correction is added in place to the final engine mean row `VX[L,V_PK1]`, which is the returned prediction mean (`estimator.py:2631-2634`).

## 3. Packed `wmsg.npz` arrays

The key inventory and shapes are in `the key list of `wmsg.npz``; `setup()` loads `_WM_KEYS` and conditionally `wqx,qxc` (`estimator.py:2527-2528,3612-3616`). The table distinguishes directly named roles from opaque fitted coefficient tables.

| Array | Shape / dtype | Contents and use |
|---|---|---|
| `hT` | `(11,3,7)` float32 | Builds basis rows from `phi,Phi,1` and powers `a^0..a^6` in `Bas` (`estimator.py:2548-2552`). |
| `kT` | `(8,5,5)` float32 | Combines five selected engine quantities and their products into eight moment/law basis rows `K` (`estimator.py:2549-2552`). |
| `fT` | `(17,11,8)` float32 | Projects the `Bas` and `K` bases to the 17 `FB` rows consumed by feature, output, and chain projections (`estimator.py:2551-2552,2594,2600`). |
| `sel5` | `(5,5,9)` float32 | Selects/scales five engine fields and sigma powers into `D5`; its semantic dimensions are in the einsum and code comment (`estimator.py:2549`). |
| `pexp` | `(7,1)` float32 | Exponents applied to `a` for `Ap`; key-list preview is truncated, actual shape has seven entries (`estimator.py:2548`, ``wmsg.npz``). |
| `selp` | `(17,4)` float32 | Selects 4 of the 17 `FB` rows for the first four previous-layer neighbour atoms (`estimator.py:2554-2557`). |
| `e4` | `(4,7,1)` float32 | Powers of `[Phi,phi,a,var]` used to make Gaussian ReLU raw moment/cumulant basis (`estimator.py:2558-2563`). |
| `selq` | `(6,7)` float32 | Selects/combines the raw-moment basis for six ReLU cumulants (`estimator.py:2560-2563`). |
| `relu` | `(6,7,7,7)` float64 | Fixed coefficient table converting powers of `Phi,phi,a,var` to cumulants `k2..k6` (`estimator.py:2560-2563`). |
| `e6` | `(6,1)` float32 | Exponents `0,-1,-1.5,-2,-2.5,-3` on variance for sigma scaling in `FAC` (`estimator.py:2586-2589`, ``wmsg.npz``). |
| `e2` | `(2,1)` float32 | Exponents `-1,-2` for `cn2` scaling factors (`estimator.py:2586-2589`). |
| `b2` | `(15,13,10)` float32 | Selects/combines 13 neighbour sums with 10 scaling factors into 13 `AT` channels (`estimator.py:2590-2592`). |
| `wfb` | `(15,16,96)` float32 | Layer-specific projection of 16 `FB` feature rows to the 96 packed gate channels; normalization and input weights are folded in (`estimator.py:2593-2598`). |
| `wat` | `(15,96)` float32 | Projection of 15 `AT` neighbour channels into those same 96 gate channels, with their feature scaling folded in (`estimator.py:2591-2594`). |
| `wqx` | `(11,96)` float32 | Projects the 11 selected `arcsinh(QX/qxc)` taps into the gate channels (`estimator.py:2595-2598`). |
| `qxc` | `(15,11)` float32 | Per-layer/tap arcsinh scales, calculated from training dump medians of nonzero absolute taps (`research/r25a/gru_lean.py:63-75,270-272`; inference `estimator.py:2595-2598`). |
| `msc` | `(2,67)` float32 | Projects sigma/message scale factors (`SVB[:,3:5]`) to 67 channels consumed with the packed input/message state by `wbig` (`estimator.py:2599,2623`). |
| `wbig` | `(128,67)` float32 | Main packed gate projection for 67 message/state channels into the 128-channel intermediate used for the gates (`estimator.py:2603-2625`). The rows are split as `x_z/2`, `x_r/2`, `n_in+n_h/2`, `n_h/2` (`2605`). |
| `cm` | `(2,2)` float32 | Gate interpolation coefficients used to combine `Hm` and candidate `NN`; the trained values are approximately `[[.5,-.5],[1/sqrt(8),1/sqrt(8)]]` by the flattened inventory (`estimator.py:2629`, ``wmsg.npz``). |
| `mc` | `(2,98)` float32 | Projection of `FB` output/carry basis rows into two per-neuron coefficients used with the packed 98-vector `m_j` to produce `dp` (`estimator.py:2600,2610-2614,2630`). |

`hT` through `b2` are fixed basis/selection tables, not trainable recurrent weights. `wfb`, `wat`, `wqx`, `msc`, `wbig`, `cm`, and `mc` are the folded learned projections/fold parameters. The original `mu/sd`, PyTorch GRUCell's separate `weight_ih`, `weight_hh`, and biases, and linear head arrays are not present as separate keys: the estimator's precomputed projections encode their combined effect (`estimator.py:2593-2630`, training architecture `research/lc3_wmsg.py:66-76`).

## 4. FLOPs

All corrector tensor operations use `fnp` inside `_ns("wmsg")`, so flopscope counts them during prediction; loading `wmsg.npz` in `setup()` is free (`estimator.py:2540-2543,3611-3616`). Measured with flopscope 0.12.3 (mean over public mini networks 0-2), the corrector costs 2.19 GFLOPs per network, 0.7% of the bill (write-up section 7.4). The largest terms per layer at n = 1024: the recurrent messages `W^T [h', dp]` (33 columns, about 66 n^2 FLOPs) and `(W o W)^T |dp|` (about 3 n^2) for t >= 2; the neighbour sums, including forming `W o W` and `W^4` (about 29 n^2); the packed gate projection (about 2 x 128 x 67 n); and the float64 ReLU-cumulant table and the feature folds, which are per-neuron but not small.

## 5. Differences from write-up section 4

1. **The feature counts agree, but the code makes depth placement explicit.** The write-up lists 14 own features, 11 structure taps, 16 neighbour features, and includes depth with the messages (write-up section 4). The training tensor is ordered as 14 base + 16 neighbour + depth + 11 appended taps, for 42 feature columns before the two messages; inference projects the feature groups separately into gate inputs (`research/lc3_wmsg.py:53-63`, `research/r25a/gru_lean.py:45-60`, `estimator.py:2593-2598`).
2. **“W^o2 through W^o6” is not one uniform set of neighbour products.** Code uses `W^2` for six channels and `W^4` for three, plus `W^3`, `W^5`, and `W^6` for remaining cumulants; the exact assignments are `SS` and the contractions at `estimator.py:2564-2584`. Thus the prose shorthand at the write-up's section 4 hides the distinct channel mapping.
3. **The recurrence is conceptually accurate but its input scaling is packed.** Section 4 states `Hm=W^T H/sqrt(2)` and messages divided by sigma and multiplied by `1e3` (write-up section 4). Training does exactly that (`research/lc3_wmsg.py:91-103`); inference has the same raw contractions but folds the scales into `msc/wbig` rather than visibly multiplying by `1e3` or dividing by sigma (`estimator.py:2599,2620-2630`).
4. **The stated “weights ... with input standardisation folded in” is directionally right but incomplete.** Standardization is folded into layer projections, alongside message scaling, gate packing, and the head/carry projection. The archive has no `mu`, `sd`, vanilla GRU `weight_ih/weight_hh`, or standalone head arrays (the write-up's section 4; `estimator.py:2593-2630`, key inventory `the key list of `wmsg.npz``).
5. **The chain is mean-only as stated, but the exact inference route is folded.** `dp_t=D_t*push_t+u_t`, `D_t` as the full derivative, and final addition to the mean agree with the write-up (write-up section 4). The estimator obtains these through `COs` and the packed head/carry contraction, rather than exposing a `D_t` array or separate `u_t` vector (`estimator.py:2599-2600,2629-2634`).

