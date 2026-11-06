"""Parameter-to-readout sensitivity (Sections 2.3-2.4, Supplementary S4-S5).

J = dy/dtheta is estimated by central differences in optimisation
coordinates, h_p = 0.01 x (range of coordinate p).  All 52 x 2 perturbed
parameter sets are simulated in one pass.  The information matrix of a series
is G = J^T J over its flattened dose x time readout; each dose m also has its
own G_m = J_m^T J_m, and G = sum_m G_m.
"""
from __future__ import annotations

import numpy as np

from .fitting import build, push
from .model import PARAMETERS, Coordinates
from .series import tile

EPS_LOGDET = 1e-6


def jacobian(panel: dict, theta: dict, coords: Coordinates, readouts=("NADH",),
             frac: float = 0.01, sim=None):
    """{readout: J} with J of shape (52, doses, times); also returns the simulator."""
    V = 2 * len(PARAMETERS)
    k = panel["n_wells"]
    if sim is None:
        sim = build(panel, V)[0]
    base = coords.to_physical(theta)
    values, h = {}, {}
    for j, p in enumerate(PARAMETERS):
        h[p] = frac * coords.span(p)
        col = np.full(V, float(base[p]), dtype=np.float32)
        col[2 * j] = coords.to_physical({p: float(theta[p]) + h[p]}, [p])[p]
        col[2 * j + 1] = coords.to_physical({p: float(theta[p]) - h[p]}, [p])[p]
        values[p] = np.repeat(col, k)
    sim.set_parameters(values)
    tiled = tile(panel, V)
    sim.set_assay(tiled["inputs"], tiled["enzymes"])
    sim.run()
    out = {}
    for ro in readouts:
        y = sim.trace(ro).reshape(V, k, len(panel["steps"]))
        out[ro] = np.stack([(y[2 * j] - y[2 * j + 1]) / (2.0 * h[p])
                            for j, p in enumerate(PARAMETERS)])
    return out, sim


def information(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G, per-dose G_m) from J of shape (52, doses, times)."""
    flat = J.reshape(J.shape[0], -1)
    return flat @ flat.T, np.stack([J[:, m, :] @ J[:, m, :].T for m in range(J.shape[1])])


def correlation_form(G: np.ndarray) -> np.ndarray:
    """G_ij / sqrt(G_ii G_jj): parameter-specific magnitude removed."""
    d = np.sqrt(np.maximum(np.diag(G), 0.0))
    live = d > 0
    R = np.zeros_like(G)
    R[np.ix_(live, live)] = G[np.ix_(live, live)] / np.outer(d[live], d[live])
    return R


def effective_rank(G: np.ndarray) -> float:
    """exp of the entropy of the normalised eigenvalues."""
    w = np.clip(np.linalg.eigvalsh(G), 0, None)
    w = w / max(w.sum(), 1e-300)
    return float(np.exp(-np.sum(w * np.log(np.clip(w, 1e-300, None)))))


def leading_direction(G: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((G + G.T) / 2.0)
    return V[:, np.argmax(w)]


def abs_cosine(G1: np.ndarray, G2: np.ndarray) -> float:
    """|cos| between the leading sensitivity directions of two information matrices."""
    return float(abs(leading_direction(G1) @ leading_direction(G2)))


def step_shares(G: np.ndarray) -> np.ndarray:
    """Share of root-mean-square sensitivity carried by each reaction step E1-E10."""
    from .model import STEP_OF
    rms = np.sqrt(np.maximum(np.diag(G), 0.0))
    by_step = np.array([rms[[STEP_OF[p] == s for p in PARAMETERS]].sum() for s in range(1, 11)])
    return by_step / by_step.sum()


def new_information_fraction(G_ref: np.ndarray, G_new: np.ndarray, tol: float = 1e-2) -> float:
    """Fraction of a candidate's information lying outside a reference panel's.

    Along each eigen-direction v of G_new (eigenvalue l > tol x the largest),
    l / (v^T G_ref v + l) is the share the candidate supplies; the fraction is
    the eigenvalue-weighted mean of that share.
    """
    w, V = np.linalg.eigh((G_new + G_new.T) / 2.0)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > tol * max(w[0], 1e-300)
    lam, Vk = w[keep], V[:, keep]
    if not lam.size or lam.sum() <= 0:
        return 0.0
    along = np.einsum("ij,jk,ki->i", Vk.T, G_ref, Vk)
    return float(np.sum(lam * lam / np.maximum(along + lam, 1e-300)) / np.sum(lam))


def logdet(M: np.ndarray) -> float:
    w = np.linalg.eigvalsh(M + EPS_LOGDET * np.eye(M.shape[0]))
    return float(np.sum(np.log(np.clip(w, 1e-300, None))))


def information_first(G: dict[str, np.ndarray], threshold: float = 0.2) -> dict:
    """Greedy D-optimal panel on trace-normalised information matrices.

    Opens with the pair of largest log det, then adds the series of largest
    gain until a gain falls below `threshold` x the first addition.
    """
    names = sorted(G)
    Gn = {s: G[s] / np.trace(G[s]) for s in names}
    base = logdet(np.zeros_like(Gn[names[0]]))
    pairs = sorted(((logdet(Gn[a] + Gn[b]) - base, a, b)
                    for i, a in enumerate(names) for b in names[i + 1:]), reverse=True)
    order = [pairs[0][1], pairs[0][2]]
    cur = Gn[order[0]] + Gn[order[1]]
    gains, eranks = [], [effective_rank(cur)]
    while len(order) < len(names):
        g, r = max((logdet(cur + Gn[r]) - logdet(cur), r) for r in names if r not in order)
        cur = cur + Gn[r]
        order.append(r)
        gains.append(g)
        eranks.append(effective_rank(cur))
    stop = next((i for i, g in enumerate(gains) if g < threshold * gains[0]), len(gains))
    return {"opening_pair_logdet": pairs[0][0], "full_order": order, "gains": gains,
            "effective_rank": eranks, "panel": order[:stop + 2]}


def transfer_first(G: dict[str, np.ndarray], performance: dict[str, dict],
                   threshold: float = 0.2) -> dict:
    """Walk series ranked by fit and transfer quality, admitting each only if
    its information gain is at least `threshold` x the first accepted gain.

    performance[s] = {"in_distribution": ..., "transfer": ...}: median ranking
    score of the series' complementary fits at its best lambda (lowest
    in-distribution score), on the series itself and on the twelve others.
    """
    names = [s for s in performance if s in G]
    Gn = {s: G[s] / np.trace(G[s]) for s in names}
    ri = {s: i + 1 for i, s in enumerate(sorted(names, key=lambda x: performance[x]["in_distribution"]))}
    rt = {s: i + 1 for i, s in enumerate(sorted(names, key=lambda x: performance[x]["transfer"]))}
    ranked = sorted(names, key=lambda s: (ri[s] + rt[s]) / 2)   # ties keep table order
    panel, cur, ref, walk = [ranked[0]], Gn[ranked[0]].copy(), None, []
    for s in ranked[1:]:
        g = logdet(cur + Gn[s]) - logdet(cur)
        ok = ref is None or g >= threshold * ref
        if ref is None:
            ref = g
        if ok:
            panel.append(s)
            cur = cur + Gn[s]
        walk.append({"series": s, "gain": g, "accepted": ok, "effective_rank": effective_rank(cur)})
    return {"ranking": ranked, "walk": walk, "panel": panel}
