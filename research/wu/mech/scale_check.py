"""Small check (1 core, under 1 GB): is the engine's next-layer variance the variance of the preactivations, or of
preactivations with the global scale taken out? 10^5 Gaussian inputs, three versions of z_t: as is; inputs put on the
sphere of radius sqrt(n) (z * sqrt(n) / |x|); each sample's z_t divided by its |h_{t-1}| / rms |h_{t-1}|. The engine's
variance at layer L + 1 is diag(W^T C'_L W) from the clean-band recording (rec/loc/Lbase)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ablate as AB
SNAP = Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a"
N, B = 100_000, 10_000
for mid in (0, 1):
    (_, _, _, W, gt), = list(AB.load_rows(SNAP, "mini", [mid]))
    L_, n = W.shape[0], W.shape[1]
    rng = np.random.default_rng([7, mid])
    s1 = np.zeros((3, L_, n)); s2 = np.zeros((3, L_, n)); hn2 = np.zeros(L_)
    A1 = np.zeros((L_, n)); A2 = np.zeros((L_, n))
    for b in range(N // B):
        x = rng.standard_normal((B, n), dtype=np.float32)
        xs = np.sqrt(n) / np.linalg.norm(x.astype(np.float64), axis=1)
        h = x
        for t in range(L_):
            hn = np.linalg.norm(h.astype(np.float64), axis=1)
            z = (h @ W[t]).astype(np.float64)
            hn2[t] += (hn ** 2).sum()
            for k, zz in enumerate((z, z * xs[:, None])):
                s1[k, t] += zz.sum(0); s2[k, t] += (zz ** 2).sum(0)
            zl = z / hn[:, None]  # rescaled below by rms |h_{t-1}|
            A1[t] += zl.sum(0); A2[t] += (zl ** 2).sum(0)
            h = np.maximum(z, 0.0).astype(np.float32)
    var = s2 / N - (s1 / N) ** 2
    r2 = hn2 / N
    var_ln = (A2 / N - (A1 / N) ** 2) * r2[:, None]
    base = np.load(HERE / "rec" / "loc" / f"Lbase_mini{mid}.npz")
    print("mini", mid, "mean rel err of the engine's variance against: full | sphere inputs | per-layer norm")
    for L in range(2, 15):
        if f"Cp_{L}" not in base.files:
            continue
        Wn = W[L + 1].astype(np.float64)
        v = np.einsum("ka,kl,la->a", Wn, base[f"Cp_{L}"].astype(np.float64), Wn, optimize=True)
        r = [((v - vv[L + 1]) / vv[L + 1]).mean() for vv in (var[0], var[1], var_ln)]
        q = [np.sqrt((((v - vv[L + 1]) / vv[L + 1]) ** 2).mean()) for vv in (var[0], var[1], var_ln)]
        print("  layer %2d  mean %+.4f %+.4f %+.4f   rms %.4f %.4f %.4f" % (L + 1, *r, *q))
