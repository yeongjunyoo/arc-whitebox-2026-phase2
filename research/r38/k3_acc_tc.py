# Attribution: re-implements, in flopscope, the factored k_max=3 AUGMENT cumulant propagation of ARC's reference
#   alignment-research-center/mlp_cumulant_propagation, commit 93d091a (factor_k3.py, kprop_harmonic.py, harmonic.py),
#   MIT License, Copyright (c) 2026 Alignment Research Center, and the companion paper arXiv:2605.05179,
# with one approximation of ours: the off-diagonal of the activation fourth-cumulant core is not computed (that needs
# per-source contractions of the third cumulant) but set to lam'_l * C_off each layer (memoryless law, one scalar per
# layer, a data table). The one-scalar-per-layer law idea follows the text of AIcrowd forum post 18218 (no code from it
# or its repository was read); the activation-side form, exact diagonal and table are ours.
# Research estimator (accuracy study); see research/acc_report_2026-10-04.md.
# R38 (coordinator 2026-10-21): a carried fourth-cumulant core (transport law: G_off := lam'_t C_off + tau offd(D Gp D),
# Gp = W^T G W) and the scale-mode extraction described in the public write-up of il.miscusi (AIcrowd forum 18248 and its
# PDF, prose only; no participant code read): eps from the (i,i,j) slice, C <- (C - eps m m^T)/(1+eps), K3 -= 2 eps
# Sym(C (x) m) on the newborn source (a U pair and an E column term), K4 slices and core corrected. The U slot holds a list.
# MIT License text of the reference applies to the derived portions (see estimators/k3_simple.py for the full text).
"""Factored K=3 AUGMENT cumulant propagation with a memoryless law for the K4 core's off-diagonal.

State after layer t (activation side): mean mu, covariance C, third cumulant K3 held as sources, and the fourth
cumulant K4 = Sym(Ga (x) I) with core Ga = diag(gam) + lam'_t offdiag(C) (gam exact, lam'_t from a table).

Per layer (preactivation z = W^T h, metric g = 2):
  m = W^T mu, S = W^T C W, Gp = W^T Ga W = W^T diag(gam - lam' c) W + lam' S   (c = diag C)
  K4 slices: K4_iiii = g Gp_ii, K4_iijj = g (Gp_ii + Gp_jj)/6, K4_iiij = g Gp_ij / 2
  power-cumulant slices: SIMPLE's terms with the per-neuron K4, plus every AUG-only diagram on the (1,1), (2,1),
  (2,2) slices (listed by research/acc_terms_dump.py), plus the (3,1)-slice terms ((1,3),) and ((3,1),)
  new K3 source (G = I): path cv = 3 w21 on (H, G, H), H = D S0; heavy two-leg paths 3 Sym(P (x) G (x) H);
  K4 -> K3 feed Sym(A (x) G (x) u), A = D offdiag(Gp) D, u = g w21 / 4; repeated-slice fix E = diag(x) + 3 Y'
  K4 core diagonal: gam_a = 12 c11 (K4_4 + rowsum K4_22)_a + 24 (c12 + c22) (sum K4_4 + sum K4_22)
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

try:
    _ns = flops.namespace
except AttributeError:
    import contextlib

    def _ns(name: str):
        return contextlib.nullcontext()

_G = 2.0  # metric scalar g = init_scale (E[W^T W] = 2 I for He-init)
# oracle activation-side law coefficients measured with the reference at n = 256 (research/acc_ref_ablate.py, actlaw)
LAM256 = [0.05702, 0.04877, 0.04264, 0.04129, 0.03733, 0.03407, 0.03294, 0.02945, 0.02798, 0.02585, 0.02504, 0.02367,
          0.02242, 0.01828, 0.01877, 0.01743]


# lam' table refitted for the recommended form (fold_heavy, feed_mode="lam2") on training mini MLPs 50-52
# (five layer-group scalars around 0.25 * LAM256; research/acc_aug_fit.py, acc_aug_fit_aug_foldlam2_jacgrp_r0.json)
LAM_REFIT = [0.01798947, 0.01538665, 0.01345267, 0.01320129, 0.01193519, 0.0108929, 0.00879968, 0.00786735, 0.00747465, 0.00754678, 0.00731031, 0.00691034, 0.00820556, 0.00669035, 0.00686969, 0.0043575]


def _k3dot(s, v):
    """M_jk = sum_i v_i K3_ijk for one source (Sym_avg of its leg products; G None = identity), used for the K211 hub of
    the carried core (ARC factor_k3.py, 'pK_111 -> K_211' and 'pK_211 -> K_211'). Repeated-index parts are ignored."""
    G, H, E, P, A, cv, u, U = s["G"], s["H"], s["E"], s["P"], s["A"], s["cv"], s["u"], s["U"]
    o = fnp.outer

    def gT(X):  # X G^T
        return X if G is None else X @ G.T

    gv = v if G is None else G.T @ v
    hv = H.T @ v
    B1 = gT(H * (cv * hv)[None, :])
    M = (B1 + B1.T + (H * (cv * gv)[None, :]) @ H.T) * (1.0 / 3.0)
    if E is not None:
        ev = E.T @ v
        GdG = fnp.diag(ev) if G is None else (G * ev[None, :]) @ G.T
        B2 = gT(E * gv[None, :])
        M = M + (GdG + B2 + B2.T) * (1.0 / 3.0)
    if P is not None:
        pv = P.T @ v
        B3 = gT(H * pv[None, :]).T + (P * gv[None, :]) @ H.T + gT(P * hv[None, :])
        M = M + (B3 + B3.T) * 0.5
    terms = []
    if A is not None:
        terms.append((A, u))
    for up, cf in (U or []):
        terms.append((H * cf[None, :], up))
    for X, w in terms:  # Sym_avg(Mat (x) w), Mat = X G^T
        Mat = gT(X)
        r = Mat @ v + Mat.T @ v
        M = M + (o(r, w) + o(w, r) + (Mat + Mat.T) * float(v @ w)) * (1.0 / 6.0)
    return M


class Estimator(BaseEstimator):
    def __init__(self, dtype=None, lam=None, lam_scale: float = 0.25, heavy: bool = True, feed: bool = True,
                 aug2: bool = True, k4: bool = True, fold_heavy: bool = False, feed_mode: str = "full",
                 preset: str | None = None, carry: bool = False, tau: float = 1.0, ext: float = 0.0,
                 lam_mult: float = 1.0, cfeed: float = 0.0, k211: float = 0.0, lamrec: bool = False,
                 lam_base: float = 1.0, xc: float = 1.0, xk3: float = 1.0, xk4: float = 1.0, eps_tab: str = "") -> None:
        if preset == "recommended":  # the form evaluated on mini 0-39 in research/acc_report_2026-10-04.md
            fold_heavy, feed_mode, lam = True, "lam2", (LAM_REFIT if lam is None else lam)
        # fold_heavy: carry the heavy legs inside the path leg, H' = H + P diag(1/(2 w21)) (drops one leg per source;
        #   adds the second-order term 3 w21 Sym(P' (x) G (x) P')).
        # feed_mode: "full" = A = D offdiag(Gp) D as its own leg; "lam" = only its lam' D S0 D part, carried on the H leg
        #   with a pushed vector (no extra leg); "lam2" = as "lam" and Gp_off := lam' S_off also on the use side, so the
        #   W^T diag(.) W product is not needed (exact diagonal via a W*W matvec); "none" = no feed.
        self._fold = fold_heavy
        self._feed_mode = feed_mode if feed else "none"
        self._dtype = dtype
        self._lam = lam  # per-layer lam' table (length depth); default LAM256 * lam_scale
        self._lam_scale = lam_scale
        self._heavy, self._feed, self._aug2, self._k4 = heavy, feed, aug2, k4
        self._carry, self._tau, self._ext, self._lam_mult = carry, tau, ext, lam_mult
        # cfeed = 1: the carried core is fed to K3 through its projection on D S0 D (online lam_eff on the U pair, the
        # cheap form a production port would use); 0: as its own A leg
        self._cfeed, self._k211 = cfeed, k211
        # lamrec: the carried core held as lam_e C_off only (one scalar per layer and network, the production lam2 form
        # with lam' replaced by lam_e): lam_e <- lam_mult lam' + tau rho lam_e - 24 eps tr(C) / (n + 4),
        # rho = <D S0 D, C_off> / <C_off, C_off>; no n^3 product beyond the base engine
        self._lamrec = lamrec
        self._lam_base = lam_base  # multiplier on the lam' table in every mode
        # extraction parts (covariance, K3 newborn terms, K4 slices) and an optional fixed eps table ("a/b/c..." per layer)
        self._xc, self._xk3, self._xk4 = xc, xk3, xk4
        self._eps_tab = [float(x) for x in eps_tab.split("/")] if eps_tab else None
        if carry and self._feed_mode in ("lam", "lam2"):
            self._feed_mode = "lam" if cfeed else "full"

    def setup(self, ctx: SetupContext) -> None:
        return None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return self.propagate(mlp.weights)

    def _lam_table(self, depth):
        if self._lam is not None:
            lam = [float(v) for v in self._lam]
            if len(lam) == depth:
                return lam
        base = LAM256 if depth == len(LAM256) else [LAM256[min(t, len(LAM256) - 1)] for t in range(depth)]
        return [v * self._lam_scale for v in base]

    def propagate(self, weights, dt=None, record: dict | None = None):
        dt = (fnp.float32 if self._dtype is None else self._dtype) if dt is None else dt
        depth = len(weights)
        n = weights[0].shape[0]
        lam_tab = [x * self._lam_base for x in self._lam_table(depth)]
        g = _G
        cc = n / 2.0 + 2.0
        c1 = n / 2.0 + 1.0
        c11 = 1.0 / (2.0 * n + 8.0)
        c12 = -1.0 / (16.0 * cc * (cc - 2.0))
        c22 = 1.0 / (32.0 * c1 * (c1 - 1.0))
        mul, add, mm = fnp.multiply, fnp.add, fnp.matmul

        def offd(M):
            M = fnp.array(M)
            fnp.fill_diagonal(M, 0.0)
            return M

        mu = None
        C = None
        gam = None  # activation K4 core diagonal
        Goff = None  # carried activation K4 core off-diagonal (carry)
        eps_log = []
        lam_a = 0.0
        sources: list[dict] = []
        rows = []
        for t in range(depth):
            W = weights[t]
            if W.dtype != dt:
                W = W.astype(dt)
            Wt = W.T
            last = t == depth - 1
            # ---------------- linear step ----------------
            if mu is None:
                m = fnp.zeros(n, dtype=dt)
                S = Wt @ W
            else:
                m = Wt @ mu
                S = Wt @ (C @ W)
            var = fnp.diagonal(S).copy()
            Gp = None
            if gam is not None and self._k4:
                cdiag = fnp.diagonal(C)
                delta = gam - lam_a * cdiag
                if self._carry:  # carried core: Gp = W^T (diag(gam) + Goff) W
                    Ga = fnp.zeros((n, n), dtype=dt) if Goff is None else fnp.array(Goff)
                    fnp.fill_diagonal(Ga, gam)
                    Gp = Wt @ (Ga @ W)
                elif self._feed_mode == "lam2":  # Gp_off := lam' S_off everywhere; exact diagonal by a W*W matvec
                    Gp = lam_a * S
                    fnp.fill_diagonal(Gp, (W * W).T @ delta + lam_a * var)
                else:
                    Gp = Wt @ (W * delta[:, None]) + lam_a * S
            # push sources
            pre = []
            for s in sources:
                q = {"cv": s["cv"]}
                q["G"] = Wt if s["G"] is None else Wt @ s["G"]
                for k in ("H", "E", "P", "A"):
                    q[k] = None if s[k] is None else Wt @ s[k]
                q["u"] = None if s["u"] is None else Wt @ s["u"]
                q["U"] = None if s.get("U") is None else [(Wt @ u_, cf_) for u_, cf_ in s["U"]]
                pre.append(q)
            # ---------------- slices t3, T of the preactivation K3 ----------------
            t3 = None
            T = None
            for q in pre:
                G_, H_, E_, P_, A_, cv, u = q["G"], q["H"], q["E"], q["P"], q["A"], q["cv"], q["u"]
                HG = H_ * G_
                part = (HG * H_) @ cv + ((E_ * G_) * G_).sum(axis=1)
                if P_ is not None:
                    part = part + 3.0 * ((P_ * G_) * H_).sum(axis=1)
                if A_ is not None:
                    rA = (A_ * G_).sum(axis=1)
                    part = part + u * rA
                U_ = q["U"]
                rUs = []
                if U_ is not None:
                    for up, cf in U_:
                        rU = HG @ cf
                        part = part + up * rU
                        rUs.append(rU)
                t3 = part if t3 is None else t3 + part
                if not last:
                    LH = HG * (cv * (2.0 / 3.0))[None, :]
                    LG = (H_ * H_) * (cv * (1.0 / 3.0))[None, :] + (E_ * G_) * (2.0 / 3.0)
                    if P_ is not None:
                        LH = LH + P_ * G_
                        LG = LG + P_ * H_
                    if A_ is not None:
                        LG = LG + A_ * (u * (1.0 / 3.0))[:, None]
                    if U_ is not None:
                        for up, cf in U_:
                            LH = LH + (G_ * (cf * (1.0 / 3.0))[None, :]) * up[:, None]
                            LG = LG + (H_ * (cf * (1.0 / 3.0))[None, :]) * up[:, None]
                    acc = mm(LH, H_.T) + mm(LG, G_.T) + mm((G_ * G_) * (1.0 / 3.0), E_.T)
                    if P_ is not None:
                        acc = acc + mm(G_ * H_, P_.T)
                    if A_ is not None:
                        acc = acc + mm(G_ * (u * (1.0 / 3.0))[:, None], A_.T) + fnp.outer(rA * (1.0 / 3.0), u)
                    if U_ is not None:
                        for (up, cf), rU in zip(U_, rUs):
                            acc = acc + fnp.outer(rU * (1.0 / 3.0), up)
                    T = acc if T is None else T + acc
            if T is not None:
                fnp.fill_diagonal(T, 0.0)
            # ---------------- Wick coefficients ----------------
            var = fnp.maximum(var, 1e-10)
            sig = fnp.sqrt(var)
            a = m / sig
            phi = flops.stats.norm.pdf(a)
            Phi = flops.stats.norm.cdf(a)
            if dt != fnp.float64:
                phi = phi.astype(dt)
                Phi = Phi.astype(dt)
            aa = a * a
            isig = 1.0 / sig
            if record is not None:
                record.setdefault("alive2", []).append(int((a > -2.0).sum()))
                record.setdefault("alive15", []).append(int((a > -1.5).sum()))
            w01 = sig * phi + m * Phi
            w11 = Phi
            w21 = phi * isig
            w31 = -(a * phi) * (isig * isig)
            w41 = (aa - 1.0) * phi * (isig * isig * isig)
            w51 = -(a * (aa - 3.0)) * phi * (isig * isig * isig * isig)
            w02 = var * (a * phi + (aa + 1.0) * Phi)
            w12, w22, w32, w42, w52 = 2.0 * w01, 2.0 * w11, 2.0 * w21, 2.0 * w31, 2.0 * w41
            d4 = None if Gp is None else g * fnp.diagonal(Gp)  # K4_iiii per neuron
            pK1 = w01
            pK2 = w02
            if t3 is not None:
                pK1 = pK1 + t3 * w31 * (1.0 / 6.0)
                pK2 = pK2 + t3 * w32 * (1.0 / 6.0)
            if d4 is not None:
                pK1 = pK1 + d4 * w41 * (1.0 / 24.0)
                pK2 = pK2 + d4 * w42 * (1.0 / 24.0)
            rows.append(pK1)
            if last:
                break
            w03 = var * sig * ((aa + 2.0) * phi + (aa + 3.0) * a * Phi)
            w04 = var * var * ((aa + 5.0) * a * phi + (aa * aa + 6.0 * aa + 3.0) * Phi)
            pK3 = w03
            pK4 = w04
            if t3 is not None:
                pK3 = pK3 + t3 * w11  # w33 = 6 w11, /6
                pK4 = pK4 + t3 * (4.0 * w01)  # w34 = 24 w01, /6
            if d4 is not None:
                pK3 = pK3 + d4 * (w21 * 0.25)  # w43 = 6 w21, /24
                pK4 = pK4 + d4 * w11  # w44 = 24 w11, /24
            S0 = offd(S)
            Q = S0 * S0
            S3 = Q * S0
            if Gp is not None:
                gd = fnp.diagonal(Gp)
                k22 = (gd[:, None] + gd[None, :]) * (g / 6.0)  # K4_iijj (diagonal unused)
                fnp.fill_diagonal(k22, 0.0)
                k31 = offd(Gp) * (g / 2.0)  # K4_iiij = K4_jjji
            else:
                k22 = None
                k31 = None

            def o(x, y):
                return fnp.outer(x, y)

            R = Q * 0.5 if k22 is None else Q * 0.5 + k22 * 0.25
            p11 = o(w11, w11) * S0 + o(w21, w21) * R
            p21 = o(w12, w11) * S0 + o(w22, w21) * R
            p22 = o(w12, w12) * S0 + o(w22, w22) * R
            X11 = None
            X22 = None
            if T is not None:
                X11 = T.T * o(w11, w21)  # ((1,2),) count 2 coef 1/2, symmetrized below
                p21 = p21 + (T.T * o(w12, w21)) * 0.5 + (T * o(w22, w11)) * 0.5
                X22 = T.T * o(w12, w22)
            if k31 is not None:  # ((1,3),) and ((3,1),): K4 (3,1) slice
                X11 = (0.0 if X11 is None else X11) + k31 * o(w11, w31) * (1.0 / 3.0)
                p21 = p21 + k31 * o(w12, w31) * (1.0 / 6.0) + k31 * o(w32, w11) * (1.0 / 6.0)
                X22 = (0.0 if X22 is None else X22) + k31 * o(w12, w32) * (1.0 / 3.0)
            if self._aug2:
                t3v = t3 if t3 is not None else fnp.zeros(n, dtype=dt)
                d4v = d4 if d4 is not None else fnp.zeros(n, dtype=dt)
                k22m = k22 if k22 is not None else fnp.zeros((n, n), dtype=dt)
                # (1,1): a, b, c, d, e (count already folded into the coefficients; symmetrized with X11)
                Y = o(w11, w41 * t3v * (1.0 / 3.0) + w51 * d4v * (1.0 / 12.0)) * S0 + o(w31, w31) * S3 * (1.0 / 6.0) \
                    + o(w31, w31) * S0 * k22m * 0.25
                if T is not None:
                    Y = Y + o(w21, w31) * S0 * T.T
                X11 = Y if X11 is None else X11 + Y
                # (2,1): a .. h
                Y = o(w12, w41 * t3v * (1.0 / 6.0) + w51 * d4v * (1.0 / 24.0)) * S0 \
                    + o(w42 * t3v * (1.0 / 6.0) + w52 * d4v * (1.0 / 24.0), w11) * S0 \
                    + o(w32, w31) * S3 * (1.0 / 6.0) + o(w32, w31) * S0 * k22m * 0.25
                if T is not None:
                    Y = Y + o(w22, w31) * S0 * T.T * 0.5 + o(w32, w21) * S0 * T * 0.5
                p21 = p21 + Y
                # (2,2): a, b, c, d, e
                Y = o(w12, w42 * t3v * (1.0 / 3.0) + w52 * d4v * (1.0 / 12.0)) * S0 + o(w32, w32) * S3 * (1.0 / 6.0) \
                    + o(w32, w32) * S0 * k22m * 0.25
                if T is not None:
                    Y = Y + o(w22, w32) * S0 * T.T
                X22 = Y if X22 is None else X22 + Y
            if X11 is not None:
                p11 = p11 + (X11 + X11.T) * 0.5
            if X22 is not None:
                p22 = p22 + (X22 + X22.T) * 0.5
            fnp.fill_diagonal(p11, 0.0)
            fnp.fill_diagonal(p21, 0.0)
            fnp.fill_diagonal(p22, 0.0)
            # ---------------- power cumulants -> cumulants ----------------
            p1sq = pK1 * pK1
            var_out = pK2 - p1sq
            K3_3 = pK3 - 3.0 * pK2 * pK1 + 2.0 * p1sq * pK1
            K4_4 = pK4 - 4.0 * pK1 * pK3 - 3.0 * pK2 * pK2 + 12.0 * p1sq * pK2 - 6.0 * p1sq * p1sq
            K3_21 = p21 - p11 * (2.0 * pK1)[:, None]
            K4_22 = p22 - 2.0 * (p21 * pK1[None, :] + p21.T * pK1[:, None]) + 4.0 * o(pK1, pK1) * p11 - 2.0 * p11 * p11
            fnp.fill_diagonal(K4_22, 0.0)
            eps = 0.0
            if self._ext > 0.0:  # scale-mode extraction (il.miscusi, forum 18248): fit eps on the (i,i,j) slice
                Sx = o(var_out, pK1) + (pK1 * 2.0)[:, None] * p11  # S_ij = v_i m_j + 2 m_i C_ij (p11 has a zero diagonal)
                fnp.fill_diagonal(Sx, 0.0)
                D21 = fnp.array(K3_21)
                fnp.fill_diagonal(D21, 0.0)
                eps = self._ext * 0.5 * float((D21 * Sx).sum()) / max(float((Sx * Sx).sum()), 1e-30)
                if self._eps_tab is not None:
                    eps = self._eps_tab[min(t, len(self._eps_tab) - 1)]
                if self._xc:
                    p11 = (p11 - o(pK1, pK1) * eps) * (1.0 / (1.0 + eps))
                    fnp.fill_diagonal(p11, 0.0)
                    var_out = (var_out - pK1 * pK1 * eps) * (1.0 / (1.0 + eps))
                if self._xk4:
                    K4_4 = K4_4 - (var_out * var_out) * (12.0 * eps)
                    K4_22 = K4_22 - (o(var_out, var_out) + p11 * p11 * 2.0) * (4.0 * eps)
                    fnp.fill_diagonal(K4_22, 0.0)
            eps_log.append(eps)
            # K4 core diagonal of the activation (r >= 1 projection, diagonal only)
            row = K4_4 + K4_22.sum(axis=1)
            gam_new = row * (12.0 * c11) + (row.sum() * (24.0 * (c12 + c22)))
            # ---------------- new third-cumulant source ----------------
            H = S0 * w11[:, None]  # D S0
            Yc = K3_21 - o(w11 * w11, w11) * (T if T is not None else 0.0)
            xv = K3_3 if t3 is None else K3_3 - (w11 * w11 * w11) * t3
            new = {"G": None, "H": H, "cv": 3.0 * w21, "P": None, "A": None, "u": None, "U": None}
            if self._heavy and self._fold:
                # P' = P diag(1/(2 w21)) = (w21_i/2) T_im + rho_m [w21_i Q_im + w11_i T_mi + (w21_i/2) K4_iimm],
                # rho = w31 / (2 w21) = -a / (2 sigma); folded into the path leg: H' = H + P'
                rho = -(a * isig) * 0.5
                Pp = Q * w21[:, None]
                if T is not None:
                    Pp = Pp + T.T * w11[:, None]
                if k22 is not None:
                    Pp = Pp + k22 * (w21 * 0.5)[:, None]
                Pp = Pp * rho[None, :]
                if T is not None:
                    Pp = Pp + T * (w21 * 0.5)[:, None]
                H = H + Pp
                new["H"] = H
            Yc = Yc - (H * H) * w21[None, :]  # path piece (2,1) slice: w21_j H_ij^2 (with H' when folded)
            if self._heavy and not self._fold:
                # heavy two-leg paths, 3 Sym(sum_m P[:,m] (x) e_m (x) H[:,m]), P = Z + V (see module docstring)
                Z = (S0 * 0.0)
                Xh = Q * w21[:, None] * 0.5  # left S^2/2 (a=2): w21_i Q_ij / 2
                Vt = o(w31, w21) * Q * 0.5  # right S^2/2 for left S: w31_j w21_k Q_jk / 2 (indexed [j, k])
                if T is not None:
                    Z = (T * w21[:, None] * 0.5) * w21[None, :]  # left WK21: w21_i T_ij / 2, times w(b_l + 1) = w21_j
                    Xh = Xh + T.T * w11[:, None] * 0.5  # left WK21^T: w11_i T_ji / 2
                    Vt = Vt + o(w31, w11) * T * 0.5 + o(w21, w21) * T.T * 0.5
                if k22 is not None:
                    Xh = Xh + k22 * w21[:, None] * 0.25  # left WK22: w21_i K4_iijj / 4
                    Vt = Vt + o(w31, w21) * k22 * 0.25
                P = Z + Xh * w31[None, :] + Vt.T  # V = Y_extra' (V[:, m] = Y_extra[m, :])
                new["P"] = P
                Yc = Yc - P * H
            if self._feed_mode == "full" and Gp is not None:
                A = offd(Gp) * w11[:, None] * w11[None, :]
                u = w21 * (g / 4.0)
                new["A"], new["u"] = A, u
                Yc = Yc - A * (u * (2.0 / 3.0))[:, None]
            elif self._feed_mode in ("lam", "lam2") and Gp is not None and (lam_a != 0.0 or self._carry):
                u = w21 * (g / 4.0)
                lam_f = lam_a
                if self._carry:  # online lam_eff: least-squares projection of offd(Gp) on S0
                    lam_f = float((offd(Gp) * S0).sum()) / max(float((S0 * S0).sum()), 1e-30)
                cf = w11 * lam_f  # A ~ lam' D S0 D = H diag(c) (H' when folded)
                X = H * cf[None, :]
                new["U"] = [(u, cf)]
                Yc = Yc - (X + X.T) * (u * (1.0 / 3.0))[:, None]
            fnp.fill_diagonal(Yc, 0.0)
            F = Yc.T * 3.0
            fnp.fill_diagonal(F, xv)
            if eps != 0.0 and self._xk3:  # K3 -= 2 eps Sym(C (x) m): the diagonal of C on the E leg, C_off ~ H diag(w11) on a U pair
                F = F - o(pK1, var_out) * (6.0 * eps)
                new["U"] = (new["U"] or []) + [(pK1 * (-6.0 * eps), w11)]
            new["E"] = F
            # old sources to the activation side
            sources = []
            for q in pre:
                s = {"cv": q["cv"], "G": q["G"] * w11[:, None]}
                for k in ("H", "E", "P", "A"):
                    s[k] = None if q[k] is None else q[k] * w11[:, None]
                s["u"] = None if q["u"] is None else q["u"] * w11
                s["U"] = None if q["U"] is None else [(u_ * w11, cf_) for u_, cf_ in q["U"]]
                sources.append(s)
            sources.append(new)
            mu = pK1
            C = fnp.array(p11)
            fnp.fill_diagonal(C, var_out)
            gam = gam_new
            lam_prev = lam_a
            lam_a = lam_tab[t]
            if self._lamrec:
                Cof = fnp.array(p11)
                fnp.fill_diagonal(Cof, 0.0)
                cc2 = max(float((Cof * Cof).sum()), 1e-30)
                tau = self._tau if self._tau > 0.0 else g * float(w11.sum()) / (n + 4.0)
                rho = float((((S0 * w11[:, None]) * w11[None, :]) * Cof).sum()) / cc2 if t > 0 else 0.0
                lam_a = lam_tab[t] * self._lam_mult + tau * rho * lam_prev - 24.0 * eps * float(var_out.sum()) / (n + 4.0)
                if record is not None:
                    record.setdefault("lam_e", []).append(lam_a)
            if self._carry:
                Cof = fnp.array(p11)
                fnp.fill_diagonal(Cof, 0.0)
                Gn = Cof * (lam_tab[t] * self._lam_mult)
                if Gp is not None and self._tau != 0.0:
                    Gt_ = (Gp * w11[:, None]) * w11[None, :]
                    fnp.fill_diagonal(Gt_, 0.0)
                    # tau < 0: the transport through the ReLU read off the (2,1,1) hub, g sum_k w11_k / (n + 4)
                    tau = self._tau if self._tau > 0.0 else g * float(w11.sum()) / (n + 4.0)
                    Gn = Gn + Gt_ * tau
                if eps != 0.0:
                    Gn = Gn - Cof * (24.0 * eps * float(var_out.sum()) / (n + 4.0))
                if self._k211 != 0.0:  # K211 hub from K3 (ARC factor_k3.py): 12/(n+4) [D (K3pre x1 (w01 - w11 m)) D - K3new x1 m]
                    vo = w01 - w11 * pK1
                    Mo = None
                    for q in pre:
                        Mq = _k3dot(q, vo)
                        Mo = Mq if Mo is None else Mo + Mq
                    Kh = -_k3dot(new, pK1)
                    if Mo is not None:
                        Kh = Kh + (Mo * w11[:, None]) * w11[None, :]
                    Kh = offd(Kh) * (12.0 / (n + 4.0))
                    Gn = Gn + Kh * self._k211
                    if record is not None:
                        record.setdefault("k211_rel", []).append(
                            float(fnp.sqrt((Kh * Kh).sum() / max(float((Cof * Cof).sum()), 1e-30))))
                        record.setdefault("k211_lam", []).append(
                            float((Kh * Cof).sum()) / max(float((Cof * Cof).sum()), 1e-30))
                Goff = Gn
            if record is not None:
                record.setdefault("gam", []).append(gam_new)
                record["eps"] = eps_log
        return fnp.stack(rows, axis=0)
