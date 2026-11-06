"""The thirteen NADH dilution series (Table 1, Table S1) and the quantities
built from them: fitting-window targets, adjacent-dose pairs for the
dilution-response loss, and the micromolar reading of a curve loss.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .model import DATA

OD340_PER_NADH = 0.7989        # OD340 (mM) per mM NADH, from the calibration curve
CURVE_OFFSET_UM = 50.0         # c in L_curve
STEP_S = 6.0                   # one stored simulator step (60 x 0.1 s)
SERIES = [f"S-E{k}" for k in range(1, 11)] + ["S-Gluc", "S-Gluc-E1 mid", "S-Gluc-E1 high"]


def load_series(path: Path = DATA / "nadh_dilution_series.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def series_data(all_series: dict, name: str, window_min: float = 500.0) -> dict:
    """One series restricted to the fitting window, doses in ascending order."""
    s = all_series[name]
    conds = sorted(s["conditions"], key=lambda c: c["dose_uM"])
    keep = [j for j, t in enumerate(s["time_min"]) if t <= window_min]
    times = [s["time_min"][j] for j in keep]
    return {
        "name": name,
        "doses": [c["dose_uM"] for c in conds],
        "inputs": [c["inputs_uM"] for c in conds],
        "enzymes": [c["enzymes_uM"] for c in conds],
        "time_min": times,
        "steps": [round(t * 60.0 / STEP_S) for t in times],
        "nadh": [[c["OD340_uM"][j] / OD340_PER_NADH for j in keep] for c in conds],
    }


def panel_data(series: list[dict]) -> dict:
    """Several series as one block of wells on a common time grid.

    The grid of the series with the most time points is used (the glucose
    series' within 500 min); the others are interpolated onto it.
    """
    if len(series) == 1:
        d = series[0]
        return {"inputs": d["inputs"], "enzymes": d["enzymes"], "steps": d["steps"],
                "nadh": d["nadh"], "blocks": [(d["name"], 0, len(d["doses"]))],
                "n_wells": len(d["doses"])}
    grid = series[max(range(len(series)), key=lambda i: len(series[i]["steps"]))]["steps"]
    inputs, enzymes, nadh, blocks = [], [], [], []
    for d in series:
        lo = len(inputs)
        inputs += d["inputs"]
        enzymes += d["enzymes"]
        blocks.append((d["name"], lo, len(inputs)))
        if d["steps"] == grid:
            nadh += d["nadh"]
        else:
            for y in d["nadh"]:
                nadh.append(np.interp(np.array(grid, float), np.array(d["steps"], float),
                                      np.array(y, float)).tolist())
    return {"inputs": inputs, "enzymes": enzymes, "steps": grid, "nadh": nadh,
            "blocks": blocks, "n_wells": len(inputs)}


def tile(panel: dict, n: int) -> dict:
    """n copies of a panel, one per parameter set sharing a forward pass."""
    out = dict(panel)
    for k in ("inputs", "enzymes", "nadh"):
        out[k] = [row for _ in range(n) for row in panel[k]]
    out["n_wells"] = panel["n_wells"] * n
    return out


def response_scale(diffs, floor: float = 5.0) -> float:
    """Robust scale of a series' adjacent-dose differences (uM)."""
    a = np.abs(np.asarray(diffs, np.float64))
    return max(float(np.median(a)), 0.25 * float(np.sqrt(np.mean(a * a))), float(floor))


def response_pairs(panel: dict, series_weight: dict[str, float]) -> dict:
    """Adjacent-dose pairs of every series in the panel.

    Each pair is normalised by 1/scale^2 of its own series, and the series
    contribute in proportion to `series_weight` whatever their number of doses.
    """
    nadh = panel["nadh"]
    T = len(panel["steps"])
    pa, pb, diffs, raw_w, nf, owner, used_w = [], [], [], [], [], [], []
    for name, lo, hi in panel["blocks"]:
        n_pairs = hi - lo - 1
        if n_pairs <= 0:
            continue
        d_all = [float(nadh[lo + i + 1][t]) - float(nadh[lo + i][t])
                 for i in range(n_pairs) for t in range(T)]
        s = response_scale(d_all)
        norm = min(1.0 / (s * s), 1e3)
        w = max(float(series_weight.get(name, 1.0)), 0.0)
        used_w.append(w)
        for i in range(n_pairs):
            pa.append(lo + i)
            pb.append(lo + i + 1)
            diffs.append([float(nadh[lo + i + 1][t]) - float(nadh[lo + i][t]) for t in range(T)])
            raw_w.append(w / float(n_pairs))
            nf.append(norm)
            owner.append(name)
    weights = np.asarray(raw_w, np.float32)
    if pa and sum(used_w) > 0:
        weights = weights * (float(len(pa)) / float(sum(used_w)))
    return {"pair_a": np.asarray(pa, np.int32), "pair_b": np.asarray(pb, np.int32),
            "target_diffs": np.asarray(diffs, np.float32), "weights": weights,
            "norm_factors": np.asarray(nf, np.float32), "series": owner}


def tile_pairs(pairs: dict, n: int, wells_per_copy: int) -> dict:
    return {
        "pair_a": np.concatenate([pairs["pair_a"] + s * wells_per_copy for s in range(n)]),
        "pair_b": np.concatenate([pairs["pair_b"] + s * wells_per_copy for s in range(n)]),
        "target_diffs": np.concatenate([pairs["target_diffs"]] * n, axis=0),
        "weights": np.concatenate([pairs["weights"]] * n),
        "norm_factors": np.concatenate([pairs["norm_factors"]] * n),
    }


def series_response_weights(all_series: dict, names: list[str], window_min: float = 500.0,
                            n_points: int = 10, min_weight: float = 0.1) -> dict:
    """How strongly each series' NADH slope changes between adjacent doses.

    Used to weight series against each other in a multi-series dilution-response
    loss: the late-time slope of each dose (second half of the window, 10 points)
    is compared between neighbouring doses and the mean relative change is
    rescaled over the panel to [0.1, 1].
    """
    raw = {}
    for name in names:
        s = all_series[name]
        conds = sorted(s["conditions"], key=lambda c: c["dose_uM"])
        t_all = s["time_min"]
        t_max = min(window_min, t_all[-1])
        valid = [i for i, t in enumerate(t_all) if t_max * 0.5 <= t <= t_max]
        pick = [valid[j] for j in np.linspace(0, len(valid) - 1, min(n_points, len(valid)),
                                              dtype=int).tolist()]
        t = np.array([t_all[i] for i in pick], np.float64)
        slopes = []
        for c in conds:
            y = np.array([c["OD340_uM"][i] / OD340_PER_NADH for i in pick], np.float64)
            slopes.append(np.sum((t - t.mean()) * (y - y.mean()))
                          / max(np.sum((t - t.mean()) ** 2), 1e-8))
        rel = [abs(abs(slopes[i]) - abs(slopes[i + 1]))
               / ((abs(slopes[i]) + abs(slopes[i + 1])) / 2.0 + 1e-8)
               for i in range(len(slopes) - 1)]
        raw[name] = float(np.mean(rel))
    v = np.array(list(raw.values()))
    p5, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    out = {}
    for name, x in raw.items():
        z = (x - p5) / (p95 - p5) if p95 - p5 > 1e-8 else 0.5
        out[name] = float(min_weight + (1.0 - min_weight) * max(0.0, min(1.0, z)))
    return out


def curve_loss_to_uM(loss, all_series: dict, name: str, window_min: float = 500.0):
    """Absolute NADH discrepancy (uM) matching a curve loss on one series:
    (median measured NADH + c) * (exp(L_curve) - 1)."""
    med = float(np.median(np.asarray(series_data(all_series, name, window_min)["nadh"])))
    return (med + CURVE_OFFSET_UM) * (np.exp(np.asarray(loss, float)) - 1.0)
