"""Screen 150 random parameter sets and keep the best 30 (Section 2.2).

    python 01_screen_initial_parameters.py [--arch cpu]

Each candidate is simulated on all thirteen series together (0-500 min) and
scored by the equal-contribution verification score
    L_rank = w_c L_curve + w_r L_response,   w = 1 / median over the pool.
Writes outputs/initial_parameter_sets.yaml (same format as
results/01_initial_parameter_sets.yaml).
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT, SERIES

from emp_pathway import init_taichi
from emp_pathway.fitting import build, curve_terms, push
from emp_pathway.model import PARAMETERS, Coordinates, load_ranges
from emp_pathway.series import (load_series, panel_data, response_pairs, series_data,
                                series_response_weights, tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_candidates", type=int, default=150)
    ap.add_argument("--n_keep", type=int, default=30)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=10, help="candidates per forward pass")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default=ROOT / "outputs/initial_parameter_sets.yaml")
    a = ap.parse_args()
    init_taichi(a.arch)

    coords = Coordinates(load_ranges())
    rng = np.random.default_rng(a.random_seed)
    cands = [coords.random_physical(rng) for _ in range(a.n_candidates)]

    allser = load_series()
    panel = panel_data([series_data(allser, s) for s in SERIES])
    pairs = response_pairs(panel, series_response_weights(allser, SERIES))
    K = panel["n_wells"]
    scores, cache = [], {}
    for c0 in range(0, len(cands), a.chunk):
        grp = cands[c0:c0 + a.chunk]
        C = len(grp)
        if C not in cache:
            cache[C] = build(panel, C, pairs, response_weight=1.0)[0]
        sim = cache[C]
        push(sim, coords, [coords.to_theta(p) for p in grp], K)
        tiled = tile(panel, C)
        sim.set_assay(tiled["inputs"], tiled["enzymes"])
        sim.run()
        terms = curve_terms(sim)
        x = sim.trace("NADH")
        pa, pb = pairs["pair_a"].astype(int), pairs["pair_b"].astype(int)
        w, nf = pairs["weights"].astype(float), pairs["norm_factors"].astype(float)
        td = pairs["target_diffs"].astype(float)
        for i in range(C):
            xs = x[i * K:(i + 1) * K]
            total = sum(float(np.sum(w * nf * ((xs[pb, j] - xs[pa, j]) - td[:, j]) ** 2))
                        for j in range(xs.shape[1]))
            resp = float(np.float32(total / (len(pa) * xs.shape[1])))
            scores.append({"curve": float(np.sqrt(np.mean(terms[i * K:(i + 1) * K]))),
                           "response": resp})
        print(f"  {c0 + C}/{len(cands)} candidates", flush=True)

    ok = [s for s in scores if np.isfinite(s["curve"]) and np.isfinite(s["response"])]
    med = {k: float(np.median([s[k] for s in ok])) for k in ("curve", "response")}
    wts = {k: 1.0 / med[k] for k in med}
    rank = sorted(range(len(scores)), key=lambda i: wts["curve"] * scores[i]["curve"]
                  + wts["response"] * scores[i]["response"])[:a.n_keep]
    out = {"ranking_weights": wts, "pool_medians": med,
           "sets": [{"set": j, "candidate": int(i), "curve_loss": scores[i]["curve"],
                     "response_loss": scores[i]["response"],
                     "ranking_score": wts["curve"] * scores[i]["curve"]
                     + wts["response"] * scores[i]["response"],
                     "parameters": {p: float(cands[i][p]) for p in PARAMETERS}}
                    for j, i in enumerate(rank)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
