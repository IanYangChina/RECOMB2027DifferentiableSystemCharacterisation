"""Choose multi-series panels by the two selection rules (Sections 2.3, 3.3, S4).

    python 05_select_series.py --sensitivity ../outputs/sensitivity/curve.npz

Information-first: greedy D-optimal walk on each series' trace-normalised
information matrix (update 19, mean over its five best sets).
Transfer-first: series ranked by the mean of their in-distribution and
transfer ranks (complementary single-series fits at each series' best
lambda, from results/05_selection/transfer_first_performance.csv), admitted
only if they add at least 20 % of the first accepted information gain.
Writes outputs/selection.json (same format as results/05_selection/selection.json).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import ROOT

from emp_pathway.sensitivity import information_first, transfer_first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", default=ROOT / "results/04_sensitivity/curve.npz")
    ap.add_argument("--performance",
                    default=ROOT / "results/05_selection/transfer_first_performance.csv")
    ap.add_argument("--out", default=ROOT / "outputs/selection.json")
    a = ap.parse_args()
    z = np.load(a.sensitivity)
    G = {s: z["G"][r, :, -1].astype(np.float64).mean(axis=0)
         for r, s in enumerate(z["series"])}
    with open(a.performance) as fh:
        perf = {r["series"]: {"lambda": float(r["lambda"]),
                              "in_distribution": float(r["in_distribution"]),
                              "transfer": float(r["transfer"])} for r in csv.DictReader(fh)}
    out = {}
    for rule, res in (("information_first", information_first(G)),
                      ("transfer_first", transfer_first(G, perf))):
        res["lambda"] = {s: perf[s]["lambda"] for s in res["panel"]}
        out[rule] = res
        print(f"{rule}: {', '.join(res['panel'])}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
