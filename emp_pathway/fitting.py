"""Fitting the 52 kinetic parameters to NADH dilution series (Section 2.2).

All parameter sets ("seeds") of one fit are simulated together: the panel of
wells is copied once per seed and every well carries its own parameters, so a
single forward and reverse pass serves every seed.

Objectives
  curve       L_curve, the RMS log-ratio of simulated to measured NADH, c = 50 uM
  response    L_response, the weighted error of the NADH change between
              adjacent doses
  complementary
              g = ghat_c + lambda [ghat_r - (ghat_r . ghat_c) ghat_c], both
              gradients normalised in optimisation coordinates
Multi-series fits form one such direction h_r per series and average the
unit vectors h_r / ||h_r|| with equal weight.

A small penalty (weight 10) on simulated NADH below 1 uM at the end of the
window accompanies the curve gradient, and the response-only objective,
to keep fits away from the zero-flux state.

Each fit takes 20 Adam updates (beta1 0.6, beta2 0.999, step 1/50 of each
coordinate's range).  A seed's parameters are ranked by the equal-contribution
score w_c L_curve + w_r L_response, with w = 1 / median over the screened pool.
"""
from __future__ import annotations

import numpy as np

from .model import PARAMETERS, Coordinates
from .series import CURVE_OFFSET_UM, panel_data, response_pairs, tile, tile_pairs
from .simulator import Pathway

FLOOR_WEIGHT = 10.0
ADAM = (0.6, 0.999, 1e-8)


def build(panel: dict, n_copies: int, pairs: dict | None = None,
          response_weight: float = 0.0) -> tuple[Pathway, dict]:
    """A simulator holding `n_copies` of the panel."""
    tiled = tile(panel, n_copies)
    sim = Pathway(n_wells=tiled["n_wells"], n_targets=len(panel["steps"]),
                  n_steps=int(panel["steps"][-1]) + 2)
    sim.set_targets(panel["steps"], tiled["nadh"])
    if pairs is not None and len(pairs["pair_a"]):
        sim.set_response_pairs(**{k: v for k, v in tile_pairs(
            pairs, n_copies, panel["n_wells"]).items()})
    sim.response_weight = response_weight
    sim.floor_weight = FLOOR_WEIGHT
    sim.set_assay(tiled["inputs"], tiled["enzymes"])
    return sim, tiled


def push(sim: Pathway, coords: Coordinates, thetas: list[dict], wells_per_copy: int):
    """Load one parameter set per copy (optimisation coordinates)."""
    phys = [coords.to_physical(t) for t in thetas]
    sim.set_parameters({p: np.repeat(np.array([x[p] for x in phys], np.float32),
                                     wells_per_copy) for p in PARAMETERS})
    return phys


def curve_terms(sim: Pathway) -> np.ndarray:
    """Per-well, per-time squared log-ratios, as stored in single precision."""
    x = sim.trace("NADH")
    y = sim.target_nadh.to_numpy().astype(np.float64)
    r = np.log(np.maximum(x + CURVE_OFFSET_UM, 1e-6) / np.maximum(y + CURVE_OFFSET_UM, 1e-6))
    return (r * r).astype(np.float32).astype(np.float64)


def losses(x, y, pa, pb, w, nf) -> tuple[float, float]:
    """(L_curve, L_response) of one series block: x simulated, y measured NADH."""
    r = np.log(np.maximum(x + CURVE_OFFSET_UM, 1e-6) / np.maximum(y + CURVE_OFFSET_UM, 1e-6))
    curve = float(np.sqrt(np.mean(r * r)))
    if len(pa):
        d = (x[pb] - x[pa]) - (y[pb] - y[pa])
        resp = float(np.sum(w[:, None] * nf[:, None] * d * d) / max(len(pa) * x.shape[1], 1))
    else:
        resp = float("nan")
    return curve, resp


def _adam(theta, g, m1, m2, ep, lr, coords):
    b1, b2, eps = ADAM
    g = np.where(np.isfinite(g), g, 0.0)
    m1[:] = b1 * m1 + (1 - b1) * g
    m2[:] = b2 * m2 + (1 - b2) * g * g
    upd = lr * (m1 / (1 - b1 ** (ep + 1))) / (np.sqrt(m2 / (1 - b2 ** (ep + 1))) + eps)
    new = {p: float(theta[p] - upd[j]) for j, p in enumerate(PARAMETERS)}
    return coords.clamp(new)


def _to_vec(coords, theta, phys_grad_vec):
    g = coords.gradient(theta, {p: phys_grad_vec[j] for j, p in enumerate(PARAMETERS)})
    return np.array([g[p] for p in PARAMETERS])


def fit_single(series: dict, start: list[dict], coords: Coordinates, objective: str,
               lam: float, n_updates: int, rank_w: dict, response_w: dict) -> list[dict]:
    """Fit one series from each starting parameter set (physical values).

    objective: 'curve', 'response' or 'complementary' (with weight `lam`).
    Returns, per seed, the parameters before every update and after the last,
    the per-update losses, the best-ranked update and the update direction of
    the first update (optimisation coordinates).
    """
    panel = panel_data([series])
    k, S = panel["n_wells"], len(start)
    pairs = response_pairs(panel, response_w)
    sim, _ = build(panel, S, pairs, response_weight=0.0 if objective == "curve" else 1.0)
    pl = (pairs["pair_a"].astype(int), pairs["pair_b"].astype(int),
          pairs["weights"].astype(float), pairs["norm_factors"].astype(float))
    y = np.asarray(panel["nadh"], float)
    theta = [coords.to_theta(p) for p in start]
    lr = coords.learning_rate()
    m1 = np.zeros((S, len(PARAMETERS)))
    m2 = np.zeros((S, len(PARAMETERS)))
    out = [{"theta": [], "curve_loss": [], "response_loss": [], "ranking_score": [],
            "best": (float("inf"), None, -1)} for _ in range(S)]

    def per_seed(raw, scale):
        return np.array([[raw[p][s * k:(s + 1) * k].sum() * scale[s] for p in PARAMETERS]
                         for s in range(S)])

    for ep in range(n_updates):
        push(sim, coords, theta, k)
        sim.run()
        terms = curve_terms(sim)
        x = sim.trace("NADH")
        M_s = np.array([np.mean(terms[s * k:(s + 1) * k]) for s in range(S)])
        M_all = float(np.mean(terms))
        # Per-seed gradient of each seed's own L_curve from the pooled RMS.
        c_scale = S * np.sqrt(M_all) / np.maximum(np.sqrt(M_s), 1e-12)
        if objective == "complementary":
            gc_raw = sim.gradients("curve", floor=True)
            push(sim, coords, theta, k)
            sim.run()
            gr_raw = sim.gradients("response", floor=False)
            Gc = per_seed(gc_raw, c_scale)
            Gr = per_seed(gr_raw, np.full(S, float(S)))
        else:
            raw = sim.gradients(objective, floor=True)
            G = per_seed(raw, c_scale if objective == "curve" else np.full(S, float(S)))
        for s in range(S):
            if objective == "complementary":
                # Projection in optimisation coordinates, where the 52
                # coordinates are commensurable and where Adam acts.
                hc = _to_vec(coords, theta[s], Gc[s])
                hr = _to_vec(coords, theta[s], Gr[s])
                hc = hc / max(np.linalg.norm(hc), 1e-30)
                hr = hr / max(np.linalg.norm(hr), 1e-30)
                g = hc + lam * (hr - float(hc @ hr) * hc)
            else:
                g = _to_vec(coords, theta[s], G[s])
            c, r = losses(x[s * k:(s + 1) * k], y, *pl)
            score = rank_w["curve"] * c + rank_w["response"] * (r if np.isfinite(r) else 0.0)
            o = out[s]
            o["theta"].append(dict(theta[s]))
            if ep == 0:
                o["first_gradient"] = g.copy()
            o["curve_loss"].append(c)
            o["response_loss"].append(r)
            o["ranking_score"].append(score)
            if score < o["best"][0]:
                o["best"] = (score, dict(theta[s]), ep)
            theta[s] = _adam(theta[s], g, m1[s], m2[s], ep, lr, coords)
    for s in range(S):
        out[s]["theta"].append(dict(theta[s]))
    return out


def fit_multi(series: list[dict], lam: dict[str, float], start: list[dict],
              coords: Coordinates, n_updates: int, response_w: dict) -> list[dict]:
    """Fit one shared parameter set per seed to several series jointly.

    Each series r gives h_r = ghat_c,r + lambda_r [ghat_r,r - (ghat_r,r . ghat_c,r) ghat_c,r];
    the update direction is the mean of h_r / ||h_r||.
    """
    panel = panel_data(series)
    k, S = panel["n_wells"], len(start)
    pairs = response_pairs(panel, response_w)
    sim, _ = build(panel, S, pairs, response_weight=1.0)
    y = np.asarray(panel["nadh"], float)
    own = {}
    for name, lo, hi in panel["blocks"]:
        m = (pairs["pair_a"] >= lo) & (pairs["pair_a"] < hi)
        own[name] = (pairs["pair_a"][m].astype(int) - lo, pairs["pair_b"][m].astype(int) - lo,
                     pairs["weights"][m].astype(float), pairs["norm_factors"][m].astype(float))
    theta = [coords.to_theta(p) for p in start]
    lr = coords.learning_rate()
    m1 = np.zeros((S, len(PARAMETERS)))
    m2 = np.zeros((S, len(PARAMETERS)))
    out = [{"theta": [], "curve_loss": {n: [] for n, _, _ in panel["blocks"]},
            "response_loss": {n: [] for n, _, _ in panel["blocks"]}} for _ in range(S)]
    for ep in range(n_updates):
        push(sim, coords, theta, k)
        sim.run()
        x = sim.trace("NADH")
        gc_raw = sim.gradients("curve", floor=True)
        push(sim, coords, theta, k)
        sim.run()
        gr_raw = sim.gradients("response", floor=False)
        for s in range(S):
            hs = []
            for name, lo, hi in panel["blocks"]:
                a, b = s * k + lo, s * k + hi
                hc = _to_vec(coords, theta[s], [gc_raw[p][a:b].sum() for p in PARAMETERS])
                hr = _to_vec(coords, theta[s], [gr_raw[p][a:b].sum() for p in PARAMETERS])
                hc = hc / max(np.linalg.norm(hc), 1e-30)
                hr = hr / max(np.linalg.norm(hr), 1e-30)
                h = hc + lam[name] * (hr - float(hc @ hr) * hc)
                hs.append(h / max(np.linalg.norm(h), 1e-30))
                c, r = losses(x[a:b], y[lo:hi], *own[name])
                out[s]["curve_loss"][name].append(c)
                out[s]["response_loss"][name].append(r)
            out[s]["theta"].append(dict(theta[s]))
            theta[s] = _adam(theta[s], np.mean(hs, axis=0), m1[s], m2[s], ep, lr, coords)
    for s in range(S):
        out[s]["theta"].append(dict(theta[s]))
    return out


def evaluate(all_series: dict, names: list[str], params: list[dict], response_w: dict,
             window_min: float = 500.0, scale_window_min: float | None = None,
             chunk: int = 10) -> list[dict]:
    """L_curve and L_response of each parameter set (physical values) on every series.

    The series are simulated together on the densest time grid (the glucose
    series'), the enzyme series' measurements interpolated onto it.  The
    dilution-response scale of each series is taken over `scale_window_min`
    minutes (default: the scoring window).
    """
    from .series import series_data
    data = [series_data(all_series, n, window_min) for n in names]
    panel = panel_data(data)
    K = panel["n_wells"]
    y = np.asarray(panel["nadh"], float)
    pairs = {}
    for d, (name, lo, hi) in zip(data, panel["blocks"]):
        if scale_window_min is not None:
            d = series_data(all_series, name, scale_window_min)
        p = response_pairs(panel_data([d]), response_w)
        pairs[name] = (p["pair_a"].astype(int) + lo, p["pair_b"].astype(int) + lo,
                       p["weights"].astype(float), p["norm_factors"].astype(float))
    rows, cache = [], {}
    for s0 in range(0, len(params), chunk):
        grp = params[s0:s0 + chunk]
        C = len(grp)
        if C not in cache:
            cache[C] = build(panel, C)[0]
        sim = cache[C]
        sim.set_parameters({p: np.repeat(np.array([g[p] for g in grp], np.float32), K)
                            for p in PARAMETERS})
        tiled = tile(panel, C)
        sim.set_assay(tiled["inputs"], tiled["enzymes"])
        sim.run()
        x = sim.trace("NADH")
        for i in range(C):
            off = i * K
            for name, lo, hi in panel["blocks"]:
                pa, pb, w, nf = pairs[name]
                xs, ys = x[off + lo:off + hi], y[lo:hi]
                r = np.log(np.maximum(xs + CURVE_OFFSET_UM, 1e-6)
                           / np.maximum(ys + CURVE_OFFSET_UM, 1e-6))
                d = (x[off + pb] - x[off + pa]) - (y[pb] - y[pa])
                rows.append({"set": s0 + i, "series": name,
                             "curve_loss": float(np.sqrt(np.mean(r * r))),
                             "response_loss": float(np.sum(w[:, None] * nf[:, None] * d * d)
                                                    / max(len(pa) * xs.shape[1], 1))})
    return rows
