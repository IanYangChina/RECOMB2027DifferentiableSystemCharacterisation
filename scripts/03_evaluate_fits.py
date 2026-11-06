"""Score every fitted parameter set on all thirteen series (Sections 3.1, 3.3).

    python 03_evaluate_fits.py --fits ../outputs/single_series_fits/curve.npz

Uses each set's parameters after the last update, over the 0-500 min fitting
window.  A row is `fitted` when the series was part of that fit and `unseen`
otherwise.  curve_loss_uM = (median measured NADH + 50 uM)(exp(L_curve) - 1).
Writes outputs/evaluation/<fits>.csv (same format as results/03_evaluation/).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from _common import ROOT, SERIES

from emp_pathway import init_taichi
from emp_pathway.fitting import evaluate
from emp_pathway.series import curve_loss_to_uM, load_series, series_response_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help=".npz written by script 02 or 06")
    ap.add_argument("--scale_window", type=float, default=500.0,
                    help="minutes over which each series' dilution-response scale is "
                         "taken (results/03_evaluation multi-series files: 1225)")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    init_taichi(a.arch)

    z = np.load(a.fits)
    fits, P = list(z["fits"]), list(z["parameters"])
    V = z["parameter_values"][:, :, -1, :]                      # after the last update
    params = [{p: float(V[r, s, j]) for j, p in enumerate(P)}
              for r in range(V.shape[0]) for s in range(V.shape[1])]
    allser = load_series()
    rows = evaluate(allser, SERIES, params, series_response_weights(allser, SERIES),
                    scale_window_min=a.scale_window)
    out = Path(a.out or ROOT / "outputs/evaluation" / (Path(a.fits).stem + ".csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fit", "set", "series", "role", "curve_loss", "response_loss",
                    "curve_loss_uM"])
        for r in rows:
            fit = fits[r["set"] // V.shape[1]]
            w.writerow([fit, r["set"] % V.shape[1], r["series"],
                        "fitted" if r["series"] in fit.split("+") else "unseen",
                        f"{r['curve_loss']:.6g}", f"{r['response_loss']:.6g}",
                        f"{float(curve_loss_to_uM(r['curve_loss'], allser, r['series'])):.6g}"])
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
