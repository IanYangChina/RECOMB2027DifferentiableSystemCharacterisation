"""Fit selected series jointly against one shared parameter set (Sections 2.3, 3.3).

    python 06_fit_multi_series.py --rule information_first
    python 06_fit_multi_series.py --rule transfer_first

Fits the panels formed by the first 2, 3, ... series of the rule's selection
order, each from the 30 screened sets for 20 Adam updates, with every series
contributing one unit-length complementary direction (its own lambda).
--sizes restricts the panel sizes (e.g. "2").
Writes outputs/multi_series_fits/<rule>.npz with each set's parameters after
the last update (same format as results/06_multi_series_fits/).
"""
import argparse
import json
import time

from _common import ROOT, SERIES, load_initial_sets, save_fits

from emp_pathway import init_taichi
from emp_pathway.fitting import fit_multi
from emp_pathway.model import Coordinates, load_ranges
from emp_pathway.series import load_series, series_data, series_response_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["information_first", "transfer_first"],
                    default="information_first")
    ap.add_argument("--selection", default=ROOT / "results/05_selection/selection.json")
    ap.add_argument("--sizes", default="", help="panel sizes, default 2..7")
    ap.add_argument("--n_sets", type=int, default=30)
    ap.add_argument("--n_updates", type=int, default=20)
    ap.add_argument("--initial_sets", default=ROOT / "results/01_initial_parameter_sets.yaml")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    init_taichi(a.arch)

    sel = json.loads(open(a.selection).read())[a.rule]
    order, lam = sel["panel"], sel["lambda"]
    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else range(2, len(order) + 1)
    coords = Coordinates(load_ranges())
    start, _ = load_initial_sets(a.initial_sets, a.n_sets)
    allser = load_series()
    resp_w = series_response_weights(allser, SERIES)
    names, fits = [], []
    for n in sizes:
        t0 = time.time()
        panel = order[:n]
        fits.append(fit_multi([series_data(allser, s) for s in panel], lam, start, coords,
                              a.n_updates, resp_w))
        names.append("+".join(panel))
        print(f"  {names[-1]}: {time.time() - t0:.0f} s", flush=True)
    out = a.out or ROOT / f"outputs/multi_series_fits/{a.rule}.npz"
    save_fits(out, names, fits, [a.n_updates])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
