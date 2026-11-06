"""Parameter-to-NADH sensitivity of the single-series fits (Sections 2.3, 3.1-3.3).

    python 04_parameter_to_nadh_sensitivity.py \
        --fits ../outputs/single_series_fits/curve.npz --evaluation ../outputs/evaluation/curve.csv

For each series, the five best parameter sets (lowest L_rank on that series
after fitting) are taken at the first update (0) and at the last stored
point before the 20th update (19).  Writes outputs/sensitivity/curve.npz with
  G[series, set, update, 52, 52]   information matrix J^T J
  G_dose[series, dose, 52, 52]     per-dose J_m^T J_m at update 19, mean over the
                                   five sets (doses padded with NaN to 8)
(same format as results/04_sensitivity/).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from _common import ROOT, load_initial_sets

from emp_pathway import init_taichi
from emp_pathway.model import PARAMETERS, Coordinates, load_ranges
from emp_pathway.sensitivity import information, jacobian
from emp_pathway.series import load_series, panel_data, series_data


def top_sets(evaluation, weights, n=5):
    """The n sets with the lowest L_rank on the series they were fitted to."""
    score = {}
    with open(evaluation) as fh:
        for r in csv.DictReader(fh):
            if r["role"] == "fitted":
                resp = float(r["response_loss"])
                score.setdefault(r["fit"], {})[int(r["set"])] = (
                    weights["curve"] * float(r["curve_loss"])
                    + weights["response"] * (resp if np.isfinite(resp) else 0.0))
    return {f: sorted(s, key=s.get)[:n] for f, s in score.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=ROOT / "results/02_single_series_fits/curve.npz")
    ap.add_argument("--evaluation", default=ROOT / "results/03_evaluation/curve.csv")
    ap.add_argument("--initial_sets", default=ROOT / "results/01_initial_parameter_sets.yaml")
    ap.add_argument("--updates", default="0,19")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default=ROOT / "outputs/sensitivity/curve.npz")
    a = ap.parse_args()
    init_taichi(a.arch)

    coords = Coordinates(load_ranges())
    _, weights = load_initial_sets(a.initial_sets, 1)
    z = np.load(a.fits)
    fits, stored = list(z["fits"]), list(z["stored_updates"])
    best = top_sets(a.evaluation, weights)
    updates = [int(u) for u in a.updates.split(",")]
    allser = load_series()
    G = np.zeros((len(fits), 5, len(updates), 52, 52))
    G_dose = np.full((len(fits), 8, 52, 52), np.nan)
    for r, name in enumerate(fits):
        panel = panel_data([series_data(allser, name)])
        sim, last = None, []
        for i, s in enumerate(best[name]):
            for u_i, u in enumerate(updates):
                phys = dict(zip(PARAMETERS, z["parameter_values"][r, s, stored.index(u)].astype(float)))
                J, sim = jacobian(panel, coords.to_theta(phys), coords, sim=sim)
                G[r, i, u_i], Gm = information(J["NADH"])
                if u == updates[-1]:
                    last.append(Gm)
        G_dose[r, :len(last[0])] = np.mean(last, axis=0)
        print(f"  {name}: sets {best[name]}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, series=np.array(fits), parameters=np.array(PARAMETERS),
                        sets=np.array([best[f] for f in fits]), updates=np.array(updates),
                        G=G.astype(np.float32), G_dose=G_dose.astype(np.float32))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
