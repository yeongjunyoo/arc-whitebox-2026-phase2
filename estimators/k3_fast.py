# Attribution: this estimator re-implements, in flopscope, the factored k_max=3 "SIMPLE" cumulant
# propagation of ARC's reference implementation
#   alignment-research-center/mlp_cumulant_propagation, commit 93d091a
#   (src/mlp_kprop/kprop_harmonic.py, factor_k3.py, wick.py, harmonic.py, diagslice.py, cumulants.py),
#   MIT License, Copyright (c) 2026 Alignment Research Center,
# and the companion paper arXiv:2605.05179. The per-layer equations are those of estimators/k3_simple.py (our
# port of that code, checked against it numerically by research/k3_port_check.py); this file regroups the same
# arithmetic into few, large flopscope operations. No other participant's code was read or used.
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
"""Factored K=3 SIMPLE cumulant propagation (the estimate of estimators/k3_simple.py) in few remote op calls.

On the grader every flopscope call (an op, a slice, a .T, a scalar fetch) is a round trip to the backend. The
client books the round trip as unbilled overhead (residual = wall - dispatch, flopscope-client 0.12.0 _budget.py),
but a server-side session view counts it, so this file keeps the call count low: k3_simple needs about 6,400
op calls, this file about 500 for depth 16:

* Sources of the third cumulant live in one slab stored transposed, (3 S n, n): rows (3s + x) n + j for legs
  x = [H, E, G] of source s, columns i over the current layer. A layer's push is one matmul slab @ W (no
  transpose op), the newest G^T is a copy of W, and the (2,1) slice T is one matmul 3 T^T = L^T A over the
  coefficient slab A (whose diagonal is 3 t3, so t3 needs no separate sum). The Hadamard work runs on persistent
  full-slab views (empty slots are zero), two einsums per layer.
* All per-neuron vector algebra (Wick coefficients, power cumulants, K3_3, K4_4, var_out and the products the
  matrix step needs) is three einsums with fixed coefficient tables over the bases Z0 = [1, tA, tB]
  (t3 = (tA + tB) / 3, read straight off the diagonals of [3 T^T, I]), Zs = [1, lin, sp11]
  (c4 = 3 (lin - 2 sp11) / (n (n + 2))), F = [phi, Phi, 1], a^0..a^4 and sigma^-3..sigma^4. Every layer's
  result lands in one (depth, rows, n) array, whose row v also feeds the next layers' Hadamard einsums.
* Every matrix the layer produces is one einsum over the stack M = [S0, 3T^T, 3T, Q, J, I]
  (Q = S0*S0, J = 1 - I): it writes the next source's H = D S0 and E = 3 Y^T + diag(x) and the next covariance
  C' = offdiag(pK11) + diag(var_out) straight into the new source's three slab slots (C' waits in the G slot
  until the next layer has used it). A second einsum over M gives the sums the radial 4th cumulant c4 needs.
"""

from __future__ import annotations

import contextlib

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


class Estimator(BaseEstimator):
    """ARC factored K=3 SIMPLE cumulant propagation, regrouped into few large flopscope operations."""

    def __init__(self, dtype=None) -> None:
        self._dtype = dtype  # float32 by default (float64 bills 2x and the exact path then exceeds 2^41)

    def setup(self, ctx: SetupContext) -> None:
        return None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return self.propagate(mlp.weights)

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
