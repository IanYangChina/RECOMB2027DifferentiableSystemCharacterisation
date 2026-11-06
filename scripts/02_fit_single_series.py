"""Fit the 52 parameters to each dilution series on its own (Sections 2.2, 3.1).

    python 02_fit_single_series.py --objective curve
    python 02_fit_single_series.py --objective response
    python 02_fit_single_series.py --objective complementary --lam 0.5

Each of the 30 screened parameter sets is optimised independently for 20 Adam
updates; all sets of one series share a forward pass.  --series restricts the
run to a comma-separated subset (e.g. "S-E3,S-Gluc-E1 mid").  The curve and
response fits keep the parameters before updates 0 and 19 and after update 20;
the complementary fits keep those after update 20.
Writes outputs/single_series_fits/<objective>.npz (same format as
results/02_single_series_fits/).
"""
import argparse
import time

import numpy as np

from _common import ROOT, load_initial_sets, save_fits, series_list, SERIES

from emp_pathway import init_taichi
from emp_pathway.fitting import fit_single
from emp_pathway.model import Coordinates, load_ranges
from emp_pathway.series import load_series, series_data, series_response_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["curve", "response", "complementary"],
                    default="curve")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="complementary weight lambda (0.25, 0.5, 1 or 2 in the paper)")
    ap.add_argument("--series", default="", help="default: all thirteen series")
    ap.add_argument("--n_sets", type=int, default=30)
    ap.add_argument("--n_updates", type=int, default=20)
    ap.add_argument("--initial_sets", default=ROOT / "results/01_initial_parameter_sets.yaml")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    init_taichi(a.arch)

    coords = Coordinates(load_ranges())
    start, rank_w = load_initial_sets(a.initial_sets, a.n_sets)
    allser = load_series()
    resp_w = series_response_weights(allser, SERIES)
    names = series_list(a.series)
    fits = []
    for name in names:
        t0 = time.time()
        fits.append(fit_single(series_data(allser, name), start, coords, a.objective,
                               a.lam, a.n_updates, rank_w, resp_w))
        print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    tag = a.objective if a.objective != "complementary" else f"complementary_lambda{a.lam:g}"
    out = a.out or ROOT / f"outputs/single_series_fits/{tag}.npz"
    n = a.n_updates
    if a.objective == "complementary":
        save_fits(out, names, fits, [n])
    else:
        # The first-update gradient (optimisation coordinates, NADH floor term
        # included) and the states before updates 0 and 19 and after the last.
        save_fits(out, names, fits, sorted({0, n - 1, n}), {"first_update_gradient": np.array(
            [[f["first_gradient"] for f in fit] for fit in fits])})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
