import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("submissions/shots.csv", encoding="utf-8")))  # run from the repository root
seen, shots = set(), []
for r in rows:
    shots.append((int(r["shot"]), r["submission_id"], float(r["adjusted_score"]), float(r["final_layer_mse"])))
shots = [s for s in shots if s[0] <= 39]
x = [s[0] for s in shots]; adj = [s[2] for s in shots]; mse = [s[3] for s in shots]; cb = [a / m for a, m in zip(adj, mse)]
LEFT_OUT = {37}  # Karstadt-Schwartz basis: graded, then left out for the patent (write-up section 8)
best, b = [], 1.0
for k, a in zip(x, adj):
    if k not in LEFT_OUT:
        b = min(b, a)
    best.append(b)
notes = [(1, "port of the reference factored K = 3"), (4, "old-source compression, Strassen-Winograd"),
         (6, "augmented terms, memoryless law, two legs"), (8, "lc (learned mean correction)"), (11, "chained corrector"),
         (13, "saturation-aware restriction"), (17, "dead births gathered out"), (19, "block covsym, approximate level"),
         (21, "weighted compression, star fold"), (27, "G fold"), (28, "corrector reads structure taps"),
         (30, "3T^T rows a > -1.5"), (34, "distilled corrector, kept-subspace SVD"), (36, "write-up file, 336955"),
         (38, "compression rank 256, one more restriction"), (39, "scale mode replaces the law, 337488")]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.55, 1]})
ax1.semilogy(x, adj, "o", color="#9aa9c7", ms=4, label="each submission")
ax1.semilogy(x, best, "-", color="#1f4e9c", lw=2, label="best so far")
for k in LEFT_OUT:
    ax1.plot([k], [adj[x.index(k)]], "x", color="#777777", ms=7, mew=1.5, label="graded, left out (patent)")
for k, _ in notes:
    i = x.index(k)
    ax1.plot([x[i]], [adj[i]], "o", ms=6, mfc="white", mec="#1f4e9c", mew=1.5)
    ax1.annotate(str(k), (x[i], adj[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#1f4e9c")
key = "\n".join(f"{k:>2}  {t}" for k, t in notes)
ax1.text(0.6, 0.97, key, transform=ax1.transAxes, ha="left", va="top", fontsize=7.6, family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax1.set_ylabel("adjusted score, MSE x max(0.1, C/B)")
ax1.set_title("Phase 2 submissions (grader values)", fontsize=11)
ax1.legend(loc="lower left", fontsize=8, frameon=False)
ax1.grid(True, which="both", alpha=0.25)
ax2.semilogy(x, mse, "s-", color="#b04a2f", ms=3.5, lw=1.2)
ax2.set_ylabel("final-layer MSE", color="#b04a2f")
ax3 = ax2.twinx()
ax3.plot(x, cb, "^-", color="#2f7d4a", ms=3.5, lw=1.2)
ax3.set_ylabel("FLOPs used, C/B", color="#2f7d4a")
ax2.set_xlabel("submission (shot) number")
ax2.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = "figures/trajectory.png"
fig.savefig(out, dpi=150)
print("saved", out)
