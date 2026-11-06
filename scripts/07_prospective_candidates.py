"""Prospective substrate-dilution experiments (Sections 2.4, 3.4, S5).

    python 07_prospective_candidates.py

Parameter states: sets 1, 3, 2, 22 and 16 of the two-series S-E3 + S-E8 fit
(lowest mean L_rank over the two series after fitting, scored over the full
0-1225 min record) before each of their 20 updates, 100 states in all
(results/07_prospective/parameter_states.csv).
At each state the NADH and ATP Jacobians of every candidate in
data/candidate_series.yaml and of S-E3 and S-E8 are computed, and each
candidate's new-information fraction relative to S-E3 + S-E8 on the same
readout is recorded, with its leading sensitivity direction and per-parameter
RMS sensitivity.  Writes outputs/prospective.npz and outputs/new_information.csv
(same format as results/07_prospective/).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT

from emp_pathway import init_taichi
from emp_pathway.model import ASSAY_INPUTS, DATA, PARAMETERS, Coordinates, load_ranges
from emp_pathway.sensitivity import information, jacobian, leading_direction, new_information_fraction
from emp_pathway.series import load_series, panel_data, series_data

READOUTS = ("NADH", "ATP")


def candidate_panel(c, enzymes, steps):
    rows = []
    for dose in c["doses_uM"]:
        x = list(c["background_uM"])
        x[ASSAY_INPUTS.index(c["species"])] = float(dose)
        rows.append(x)
    return {"inputs": rows, "enzymes": [list(enzymes) for _ in rows], "steps": steps,
            "nadh": [[0.0] * len(steps) for _ in rows], "blocks": [], "n_wells": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=ROOT / "results/07_prospective/parameter_states.csv")
    ap.add_argument("--candidates", default=DATA / "candidate_series.yaml")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default=ROOT / "outputs")
    a = ap.parse_args()
    init_taichi(a.arch)

    coords = Coordinates(load_ranges())
    spec = yaml.safe_load(Path(a.candidates).read_text())
    allser = load_series()
    ref = {s: panel_data([series_data(allser, s)]) for s in spec["reference"]}
    steps = ref["S-E3"]["steps"]
    cands = {n: candidate_panel(c, spec["enzymes_uM"], steps)
             for n, c in spec["candidates"].items()}
    with open(a.states) as fh:
        states = [(int(r["set"]), int(r["update"]), {p: float(r[p]) for p in PARAMETERS})
                  for r in csv.DictReader(fh)]

    names = list(cands)
    frac = np.zeros((len(names), len(READOUTS), len(states)))
    lead = np.zeros_like(frac, shape=frac.shape + (52,))
    rms = np.zeros_like(lead)
    sims = {}
    for k, (s, u, phys) in enumerate(states):
        theta = coords.to_theta(phys)
        G = {}
        for n, panel in {**ref, **cands}.items():
            J, sims[n] = jacobian(panel, theta, coords, READOUTS, sim=sims.get(n))
            for ro in READOUTS:
                G[n, ro] = information(J[ro])[0]
                if n in cands:
                    rms[names.index(n), READOUTS.index(ro), k] = np.sqrt(np.mean(J[ro] ** 2, axis=(1, 2)))
        for i, n in enumerate(names):
            for j, ro in enumerate(READOUTS):
                G_ref = sum(G[r, ro] for r in spec["reference"])
                frac[i, j, k] = new_information_fraction(G_ref, G[n, ro])
                lead[i, j, k] = leading_direction(G[n, ro])
        print(f"  state {k + 1}/{len(states)} (set {s}, update {u})", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prospective.npz", candidates=np.array(names),
                        readouts=np.array(READOUTS), parameters=np.array(PARAMETERS),
                        states=np.array([(s, u) for s, u, _ in states]),
                        new_information_fraction=frac, leading_direction=lead.astype(np.float32),
                        rms_sensitivity=rms.astype(np.float32))
    with (out / "new_information.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate", "readout", "set", "update", "new_information_fraction"])
        for i, n in enumerate(names):
            for j, ro in enumerate(READOUTS):
                for k, (s, u, _) in enumerate(states):
                    w.writerow([n, ro, s, u, f"{frac[i, j, k]:.6g}"])
    print(f"wrote {out / 'prospective.npz'} and {out / 'new_information.csv'}")


if __name__ == "__main__":
    main()
