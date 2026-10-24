"""The write-up's ablation table from the ablate.py outputs of one network set (kp or pub), one paired.py reading per
row against its base: the engine-alone rows against `engine`, the where-introduced pairs against their predecessor,
everything else against the submitted file.

    python report.py --dir research/wu/kp --floor research/wu/kp_floor.json --out research/wu/kp_table.json
    python report.py --dir research/wu/pub --out research/wu/pub_table.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = [  # (variant tag, base tag)
    ("engine", "sub"), ("engine_nok4f", "engine"), ("engine_nogf", "engine"), ("engine_nostar", "engine"),
    ("engine_nolaw", "engine"), ("engine_nosat", "engine"), ("engine_nodc", "engine"), ("engine_nokc", "engine"), ("engine_noshape", "engine"), ("nostar", "sub"), ("noapa", "sub"), ("ref_simple", "sub"), ("ref_simple", "engine"),
    ("old_2leg", "sub"), ("old_2leg", "engine"), ("ref_simple", "old_2leg"),
    ("old_2leg", "old_3leg"), ("old_sat", "old_ex35"),
    ("notaps", "sub"), ("tapsnofold", "sub"), ("nostar", "tapsnofold"), ("nostar", "notaps"),
    ("engine_notaps", "engine"),
    ("engine_nopush", "engine"), ("engine_nott", "engine"), ("engine_nolast", "engine"),
]


def files(d: Path, tag: str) -> list[str]:  # kp: <tag>.jsonl; pub: <tag>_mini.jsonl and <tag>_full.jsonl
    return [str(p) for p in (d / f"{tag}.jsonl", d / f"{tag}_mini.jsonl", d / f"{tag}_full.jsonl") if p.exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--floor", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    table = {}
    for v, b in ROWS:
        fv, fb = files(d, v), files(d, b)
        if not fv or not fb:
            table[f"{v}/{b}"] = {"missing": True}
            continue
        cmd = [sys.executable, str(HERE / "paired.py"), *fv, "--base", *fb] + (["--floor", a.floor] if a.floor else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            table[f"{v}/{b}"] = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            table[f"{v}/{b}"] = {"error": (r.stderr or r.stdout)[-400:]}
        x = table[f"{v}/{b}"]
        if "adjusted_ratio" in x:
            print(f"{v:>14} vs {b:<8} n={x['n']:2d}  adj {x['adjusted_ratio'][0]:.4f} [{x['adjusted_ratio'][1][0]:.4f}, "
                  f"{x['adjusted_ratio'][1][1]:.4f}]  mse {x['mse_ratio'][0]:.4f} [{x['mse_ratio'][1][0]:.4f}, {x['mse_ratio'][1][1]:.4f}]"
                  f"  flops {x['flops_ratio'][0]:.4f}  better {x['better']}/{x['n']}  identical {x['bit_identical']}")
    Path(a.out).write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
