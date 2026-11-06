"""Shared command-line plumbing for the scripts in this folder."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emp_pathway.model import PARAMETERS  # noqa: E402
from emp_pathway.series import SERIES  # noqa: E402


def load_initial_sets(path, n):
    """The screened initial parameter sets and the ranking weights."""
    d = yaml.safe_load(Path(path).read_text())
    return [s["parameters"] for s in d["sets"][:n]], d["ranking_weights"]


def series_list(arg):
    return [s for s in arg.split(",") if s] if arg else list(SERIES)


def save_fits(path, names, fits, keep_updates, extra=None):
    """Losses and parameter values of single- or multi-series fits.

    parameter_values[r, s, i, p]: parameter p of set s in fit r before update
    stored_updates[i] (0 = the initial set; 20 = after the last update).
    """
    from emp_pathway.model import Coordinates, load_ranges
    co = Coordinates(load_ranges())
    keep = list(keep_updates)
    arr = {
        "fits": np.array(names), "parameters": np.array(PARAMETERS),
        "stored_updates": np.array(keep),
        "parameter_values": np.array([[[[co.to_physical(f["theta"][u])[p] for p in PARAMETERS]
                                        for u in keep] for f in fit] for fit in fits]),
    }
    for key in ("curve_loss", "response_loss", "ranking_score"):
        if key in fits[0][0] and not isinstance(fits[0][0][key], dict):
            arr[key] = np.array([[f[key] for f in fit] for fit in fits], np.float64)
    arr.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arr)
