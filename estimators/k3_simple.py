# Attribution: this estimator re-implements, in flopscope, the factored k_max=3 "SIMPLE" cumulant
# propagation of ARC's reference implementation
#   alignment-research-center/mlp_cumulant_propagation, commit 93d091a
#   (src/mlp_kprop/kprop_harmonic.py, factor_k3.py, wick.py, harmonic.py, diagslice.py, cumulants.py),
#   MIT License, Copyright (c) 2026 Alignment Research Center,
# and the companion paper arXiv:2605.05179. The per-layer equations below were derived by reading that
# code and are checked against it numerically by research/k3_port_check.py. No other participant's code
# was read or used.
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
"""Factored K=3 SIMPLE cumulant propagation for He-init ReLU MLPs (x ~ N(0, I), no biases).

Reference call this reproduces:
    mlp_kprop(mlp, {1: 0, 2: I}, k_max=3, kind=SIMPLE, factor=True, use_avg_metric=True)
and returns K[f"act{l}"][1] for every layer l.

State after layer t (activation side), competition convention z = W^T h:
    mu  (n)      mean,           C (n, n) covariance,
    c4  (scalar) core of the r=2 radial 4th cumulant, K4 = c4 * Sym(g I (x) g I), g = 2 (= E[W^T W]),
    K3  (n,n,n)  third cumulant, held as a list of "sources", one per earlier layer s:
        K3 = sum_s Sym( sum_m cv_m H[:,m] (x) G[:,m] (x) H[:,m]  +  E[:,m] (x) G[:,m] (x) G[:,m] ).
        At creation G = I, H = diag(w11) S0, E = diag(x) + 3 Y^T, cv = 3 w21 (see below); every later
        layer maps all three matrices by diag(w11) W^T. The reference holds the same tensor as one
        FactoredTensor whose rank grows by 2n per layer; sharing the identity slot (G) between its two
        rank-n pieces is what makes the exact computation fit the 2^41 budget.

Per layer, with m = W^T mu, S = W^T C W, S0 = offdiag(S), Q = S0*S0, t3_i = K3pre_iii,
T_ij = K3pre_iij (i != j), k4d = 4 c4, k4o = 4 c4 / 3, w_kp = E[d^k ReLU(Z)^p], Z ~ N(m, diag S):
    pK1  = w01 + t3 w31/6 + k4d w41/24        (likewise pK2, pK3, pK4 with w_k2, w_k3, w_k4)
    pK11 = w11 w11' S0 + 1/2 (X + X^T) + w21 w21' (Q/2 + k4o/4),         X = T^T * (w11 w21')
    pK21 = w12 w11' S0 + 1/2 T^T*(w12 w21') + 1/2 T*(w22 w11') + w22 w21' (Q/2 + k4o/4)
    pK22 = w12 w12' S0 + 1/2 (Z + Z^T) + w22 w22' (Q/2 + k4o/4),         Z = T^T * (w12 w22')
    mu'  = pK1,  C' = offdiag(pK11) + diag(pK2 - pK1^2)
    K3_3 = pK3 - 3 pK2 pK1 + 2 pK1^3,  K3_21 = pK21 - 2 pK1_i pK11_ij
    K4_4 = pK4 - 4 pK1 pK3 - 3 pK2^2 + 12 pK1^2 pK2 - 6 pK1^4
    K4_22 = pK22 - 2 (pK21_ij pK1_j + pK21_ji pK1_i) + 4 pK1_i pK1_j pK11_ij - 2 pK11_ij^2
    c4'  = 3 (sum K4_4 + sum_{i!=j} K4_22) / (n (n + 2))
    new source: x = K3_3 - w11^3 t3,  Y = K3_21 - (w11^2 w11') * T - (w11^2 w21') * Q

All (n, n) work goes through a pool of reusable buffers (out=): allocating and freeing thousands of 4 MB
temporaries per predict() costs about 0.6 s of unmetered residual time on Windows, over the 400 ms cap.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

try:  # flops.namespace is an in-process (starter kit) profiling helper; the grader's flopscope client raises on access
    _ns = flops.namespace
except AttributeError:
    import contextlib

    def _ns(name: str):
        return contextlib.nullcontext()

_G2 = 4.0  # g^2 with g = mlp.init_scale = 2.0, the reference's average metric E[W W^T] for He-init ReLU layers


def _matmul_units(n: int, depth: int, max_sources: int | None) -> float:
    """Number of (n,n)@(n,n) products predict() performs (for the budget guard)."""
    units = 0.0
    for t in range(depth):
        last = t == depth - 1
        live = t if max_sources is None else min(t, max_sources)
        units += 1.5 if 0 < t < depth - 1 else 1.0  # covariance W^T C W (symmetric einsum) or one product
        if live:
            units += 3 * live - 1  # propagation; the newest source's G is W^T itself
            if not last:
                units += 3 * live  # (2,1) slice of the preactivation third cumulant
    return units


class _Pool:
    """Reusable plain (n, n) work buffers."""

    def __init__(self, n: int, dt) -> None:
        self.n, self.dt, self.free = n, dt, []

    def get(self):
        return self.free.pop() if self.free else fnp.empty((self.n, self.n), dtype=self.dt)

    def put(self, *arrs) -> None:
        self.free.extend(a for a in arrs if a is not None)


class Estimator(BaseEstimator):
    """ARC factored K=3 SIMPLE cumulant propagation (exact port; see module docstring)."""

    def __init__(self, dtype=None, max_sources: int | None = None) -> None:
        # dtype: float32 by default (float64 bills 2x and the exact path then exceeds 2^41 at n=1024).
        # max_sources: None = exact reference algorithm. An int keeps only the newest sources of the third
        # cumulant (an approximation used for the cost study in research/k3_port_notes.md).
        self._dtype = dtype
        self._max_sources = max_sources

    def setup(self, ctx: SetupContext) -> None:
        return None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        dt = fnp.float32 if self._dtype is None else self._dtype
        rate = 2 if dt == fnp.float64 else 1
        n, depth = mlp.width, mlp.depth
        keep = self._max_sources
        unit = n * n * (2 * n - 1) * rate
        # Budget guard: never triggers at the graded shape in float32 (about 0.67 of 2^41). If the exact path
        # cannot fit, keep the largest number of newest sources that does instead of being zeroed.
        while _matmul_units(n, depth, keep) * unit > 0.97 * budget and (keep is None or keep > 0):
            keep = (depth - 1 if keep is None else keep) - 1
        return self.propagate(mlp.weights, dt=dt, max_sources=keep)

    def propagate(self, weights, dt=None, max_sources: int | None = None, record: dict | None = None):
        """Per-layer means (depth, n). Phases are attributed to flopscope namespaces for the FLOP breakdown:
        cov (mean/covariance linear step), k3_push (sources through W^T, then diag(w11)), k3_slices (t3, T),
        wick_pk (Wick coefficients, power cumulants, pK -> K, c4, new source)."""
        dt = fnp.float32 if dt is None else dt
        depth = len(weights)
        n = weights[0].shape[0]
        pool = _Pool(n, dt)
        mul, add, sub, mm = fnp.multiply, fnp.add, fnp.subtract, fnp.matmul

        def keep(a):  # record mode only: snapshot a value that lives in a reused buffer
            return None if a is None else fnp.array(a)

        mu = None  # input mean 0
        C = None  # input covariance I
        C_buf = None  # pool buffer behind C, returned once the next layer has read C
        c4 = None  # no 4th cumulant at the input
        sources: list[tuple] = []  # (G or None for identity, H, E, cv), activation side
        rows = []
        for t in range(depth):
            W = weights[t]
            if W.dtype != dt:
                W = W.astype(dt)
            Wt = W.T
            last = t == depth - 1

            # ---- linear step: preactivation mean, covariance (only its diagonal at the last layer) ----
            with _ns("cov"):
                S = None
                if mu is None:
                    m = fnp.zeros(n, dtype=dt)
                    S = mm(Wt, W, out=pool.get())
                else:
                    m = Wt @ mu
                    if last:
                        CW = mm(C, W, out=pool.get())
                        mul(CW, W, out=CW)
                        var = CW.sum(axis=0)
                        pool.put(CW)
                    else:
                        S = fnp.einsum("ij,ia,jb->ab", C, W, W, out=pool.get())
                    pool.put(C_buf)
                    C = C_buf = None
                if S is not None:
                    var = fnp.diagonal(S)  # a view; fnp.maximum below copies it before the diagonal is zeroed

            # ---- third cumulant of the preactivation: push every source through W^T ----
            pre = []
            with _ns("k3_push"):
                while sources:
                    G, H, E, cv = sources.pop(0)
                    if G is None:
                        Gp = Wt  # a view of W: never written to, never pooled
                    else:
                        Gp = mm(Wt, G, out=pool.get())
                        pool.put(G)
                    Hp = mm(Wt, H, out=pool.get())
                    pool.put(H)
                    Ep = mm(Wt, E, out=pool.get())
                    pool.put(E)
                    pre.append((Gp, Hp, Ep, cv, G is None))
                    del G, H, E

            # ---- its (3,) slice t3_i = K3_iii and (2,1) slice T_ij = K3_iij (i != j) ----
            t3 = None
            T = None
            with _ns("k3_slices"):
                if pre:
                    s1, s2, s3 = pool.get(), pool.get(), pool.get()
                    P = pool.get() if not last else None
                    M = pool.get() if not last else None
                for Gp, Hp, Ep, cv, _ in pre:
                    mul(Hp, Gp, out=s1)  # HG
                    mul(s1, Hp, out=s2)
                    part = s2 @ cv
                    mul(Ep, Gp, out=s3)  # EG
                    mul(s3, Gp, out=s2)
                    part = part + s2.sum(axis=1)
                    t3 = part if t3 is None else t3 + part
                    if not last:
                        acc = pool.get() if T is None else P
                        mul(s1, (cv * (2.0 / 3.0))[None, :], out=s1)  # A1
                        mm(s1, Hp.T, out=acc)
                        mul(Hp, Hp, out=s2)
                        mul(s2, (cv * (1.0 / 3.0))[None, :], out=s2)
                        mul(s3, 2.0 / 3.0, out=s3)
                        add(s2, s3, out=s2)  # A2
                        mm(s2, Gp.T, out=M)
                        add(acc, M, out=acc)
                        mul(Gp, Gp, out=s1)
                        mul(s1, 1.0 / 3.0, out=s1)  # A3
                        mm(s1, Ep.T, out=M)
                        add(acc, M, out=acc)
                        if T is None:
                            T = acc
                        else:
                            add(T, acc, out=T)
                if T is not None:
                    fnp.fill_diagonal(T, 0.0)
                if pre:
                    pool.put(s1, s2, s3, P, M)

            with _ns("wick_pk"):
                # ---- Wick coefficients w_kp = E[d^k ReLU(Z)^p], Z ~ N(m, var) ----
                if record is not None:
                    record.setdefault("pre_var", []).append(keep(var))
                var = fnp.maximum(var, 1e-10)  # the reference's relu_wick_coef clamp
                sig = fnp.sqrt(var)
                a = m / sig
                phi = flops.stats.norm.pdf(a)
                Phi = flops.stats.norm.cdf(a)
                if dt != fnp.float64:
                    phi = phi.astype(dt)
                    Phi = Phi.astype(dt)
                aa = a * a
                sphi = sig * phi
                w01 = sphi + m * Phi
                w11 = Phi
                w21 = phi / sig
                w02 = var * (a * phi + (aa + 1.0) * Phi)
                if t3 is not None or c4 is not None:
                    isig = 1.0 / sig
                    w31 = -(a * phi) * (isig * isig)
                    w41 = (aa - 1.0) * phi * (isig * isig * isig)
                    w32 = 2.0 * w21
                    w42 = 2.0 * w31

                # ---- power-cumulant slices ----
                pK1 = w01
                pK2 = w02
                if t3 is not None:
                    pK1 = pK1 + t3 * w31 * (1.0 / 6.0)
                    pK2 = pK2 + t3 * w32 * (1.0 / 6.0)
                if c4 is not None:
                    k4d = c4 * _G2
                    pK1 = pK1 + k4d * w41 * (1.0 / 24.0)
                    pK2 = pK2 + k4d * w42 * (1.0 / 24.0)
                rows.append(pK1)
                if record is not None:
                    record.setdefault("pre_mean", []).append(m)
                    record.setdefault("t3", []).append(t3)
                    record.setdefault("T21", []).append(keep(T))
                    record.setdefault("c4_pre", []).append(c4)
                if last:
                    break

                w12 = 2.0 * (sphi + m * Phi)  # sigma (2 phi + 2 alpha Phi)
                w22 = 2.0 * Phi
                w03 = var * sig * ((aa + 2.0) * phi + (aa + 3.0) * a * Phi)
                w04 = var * var * ((aa + 5.0) * a * phi + (aa * aa + 6.0 * aa + 3.0) * Phi)
                pK3 = w03
                pK4 = w04
                if t3 is not None:
                    pK3 = pK3 + t3 * (6.0 * w11) * (1.0 / 6.0)  # w33 = 6 w11
                    pK4 = pK4 + t3 * (24.0 * w01) * (1.0 / 6.0)  # w34 = 24 w01
                if c4 is not None:
                    pK3 = pK3 + k4d * (6.0 * w21) * (1.0 / 24.0)  # w43 = 6 w21
                    pK4 = pK4 + k4d * (24.0 * w11) * (1.0 / 24.0)  # w44 = 24 w11

                S0 = S  # S's buffer becomes S0 (its diagonal was copied into var above)
                fnp.fill_diagonal(S0, 0.0)
                Q, R, O, X = pool.get(), pool.get(), pool.get(), pool.get()
                p11, p21, p22 = pool.get(), pool.get(), pool.get()
                mul(S0, S0, out=Q)
                mul(Q, 0.5, out=R)
                if c4 is not None:
                    add(R, c4 * (_G2 / 3.0) * 0.25, out=R)  # + k4o / 4
                for p, u1, u2, v1, v2 in ((p11, w11, w11, w21, w21), (p21, w12, w11, w22, w21),
                                          (p22, w12, w12, w22, w22)):
                    fnp.outer(u1, u2, out=O)
                    mul(O, S0, out=p)
                    fnp.outer(v1, v2, out=O)
                    mul(O, R, out=O)
                    add(p, O, out=p)
                if T is not None:
                    fnp.outer(w11, w21, out=O)
                    mul(T.T, O, out=X)  # X
                    add(X, X.T, out=O)
                    mul(O, 0.5, out=O)
                    add(p11, O, out=p11)
                    fnp.outer(w12, w21, out=O)
                    mul(T.T, O, out=X)
                    mul(X, 0.5, out=X)
                    add(p21, X, out=p21)
                    fnp.outer(w22, w11, out=O)
                    mul(T, O, out=X)
                    mul(X, 0.5, out=X)
                    add(p21, X, out=p21)
                    fnp.outer(w12, w22, out=O)
                    mul(T.T, O, out=X)  # Z
                    add(X, X.T, out=O)
                    mul(O, 0.5, out=O)
                    add(p22, O, out=p22)
                fnp.fill_diagonal(p11, 0.0)
                fnp.fill_diagonal(p21, 0.0)
                fnp.fill_diagonal(p22, 0.0)

                # ---- power cumulants -> cumulants ----
                mu = pK1
                p1sq = pK1 * pK1
                var_out = pK2 - p1sq
                K3_3 = pK3 - 3.0 * pK2 * pK1 + 2.0 * p1sq * pK1
                K4_4 = pK4 - 4.0 * pK1 * pK3 - 3.0 * pK2 * pK2 + 12.0 * p1sq * pK2 - 6.0 * p1sq * p1sq
                mul(p11, (2.0 * pK1)[:, None], out=X)
                sub(p21, X, out=X)  # X = K3_21
                mul(p21, pK1[None, :], out=O)  # U
                usum = O.sum()
                mul(p11, p11, out=O)
                K4_22_sum = p22.sum() - 4.0 * usum + 4.0 * (pK1 @ (p11 @ pK1)) - 2.0 * O.sum()
                c4 = (K4_4.sum() + K4_22_sum) * (3.0 / (n * (n + 2)))
                if record is not None:
                    record.setdefault("K3_21", []).append(keep(X))

                # ---- this layer's new source of the activation third cumulant ----
                xv = K3_3
                fnp.outer(w11 * w11, w21, out=O)
                mul(Q, O, out=O)
                sub(X, O, out=X)  # Y
                if t3 is not None:
                    xv = xv - (w11 * w11 * w11) * t3
                if T is not None:
                    fnp.outer(w11 * w11, w11, out=O)
                    mul(T, O, out=O)
                    sub(X, O, out=X)
                F = pool.get()
                mul(X.T, 3.0, out=F)
                fnp.fill_diagonal(F, xv)
                dcol = w11[:, None]
                mul(S0, dcol, out=S0)  # S0's buffer becomes the new source's H
                new = (None, S0, F, 3.0 * w21)

                fnp.fill_diagonal(p11, var_out)  # p11's buffer becomes the covariance
                C = flops.as_symmetric(p11, symmetry=(0, 1))
                C_buf = p11
                pool.put(Q, R, O, X, p21, p22, T)

            # ---- old sources to the activation side (rows scaled by w11); optional truncation ----
            with _ns("k3_push"):
                if max_sources is not None:
                    drop = len(pre) + 1 - max_sources
                    for Gp, Hp, Ep, cv, gview in pre[:max(drop, 0)]:
                        pool.put(None if gview else Gp, Hp, Ep)
                    del pre[:max(drop, 0)]
                while pre:
                    Gp, Hp, Ep, cv, gview = pre.pop(0)
                    Ga = mul(Gp, dcol, out=pool.get() if gview else Gp)
                    mul(Hp, dcol, out=Hp)
                    mul(Ep, dcol, out=Ep)
                    sources.append((Ga, Hp, Ep, cv))
                    del Gp, Hp, Ep, Ga
                if max_sources is None or max_sources > 0:
                    sources.append(new)
                else:
                    pool.put(new[1], new[2])
                del new
            if record is not None:  # research/k3_port_check.py and k3_structure.py only; predict() never records
                record.setdefault("act_cov", []).append(keep(C))
                record.setdefault("c4", []).append(c4)
                record.setdefault("K3_3", []).append(K3_3)
                if record.get("want_sources"):
                    record.setdefault("sources", []).append([(keep(g), keep(h), keep(e), cv) for g, h, e, cv in sources])
                if "hook" in record:
                    record["hook"](t, sources)
        return fnp.stack(rows, axis=0)
