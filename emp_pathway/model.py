"""Species, enzymes and the 52 kinetic parameters of the EMP pathway model,
and the bounded optimisation coordinates the parameters are fitted in
(Supplementary Methods S2-S3, Table S2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

DATA = Path(__file__).resolve().parents[1] / "data"

# Simulated species, in the order of the simulator's state vector.
SPECIES = ["Glucose", "G6P", "F6P", "FBP", "DHAP", "G3P", "NAD", "NADH",
           "13BPG", "Pi", "ADP", "3PG", "ATP", "2PG", "PEP", "Pyruvate"]
# Species that can be supplied at the start of an assay, in input order.
ASSAY_INPUTS = ["Glucose", "FBP", "DHAP", "G3P", "NAD", "ADP", "ATP", "Pi", "3PG"]
# E1-E10 (Fig. 1a).
ENZYMES = ["Glk", "Pgi", "PfkA", "FbaA", "TpiA", "GapA", "Pgk", "GpmA", "Eno", "PykF"]

# The 52 fitted parameters, grouped by reaction step s1-s10 (Table S2).
PARAMETERS_BY_STEP = {
    1: ["K_cat_Glucose_s1", "K_m_Glucose_s1", "K_m_ATP_s1"],
    2: ["K_m_G6P_s2", "K_m_F6P_s2", "K_cat_G6P_s2", "K_cat_F6P_s2"],
    3: ["alpha_ADP_s3", "alpha_PEP_s3", "alpha_ATP_s3", "K_cat_f_s3", "K_m_F6P_s3",
        "K_m_ATP_s3", "K_a_ADP_s3", "K_i_PEP_s3", "K_i_ATP_s3"],
    4: ["K_m_FBP_s4", "K_m_DHAP_s4", "K_m_G3P_s4", "K_cat_FBP_s4", "K_cat_DHAP_s4"],
    5: ["K_m_DHAP_s5", "K_m_G3P_s5", "K_cat_DHAP_s5", "K_cat_G3P_s5"],
    6: ["K_m_G3P_s6", "K_m_NAD_s6", "K_m_13BPG_s6", "K_m_NADH_s6", "K_cat_G3P_s6",
        "K_cat_NADH_s6"],
    7: ["K_m_ADP_s7", "K_m_13BPG_s7", "K_m_3PG_s7", "K_m_ATP_s7", "K_cat_ADP_s7",
        "K_cat_ATP_s7"],
    8: ["K_m_3PG_s8", "K_m_2PG_s8", "K_cat_3PG_s8", "K_cat_2PG_s8"],
    9: ["K_m_2PG_s9", "K_m_PEP_s9", "K_cat_2PG_s9", "K_cat_PEP_s9"],
    10: ["K_cat_f_s10", "K_m_PEP_s10", "K_m_ADP_s10", "alpha_FBP_s10", "K_a_FBP_s10",
         "alpha_ATP_s10", "K_i_ATP_s10"],
}
PARAMETERS = [p for s in range(1, 11) for p in PARAMETERS_BY_STEP[s]]
STEP_OF = {p: s for s, ps in PARAMETERS_BY_STEP.items() for p in ps}

# Empirical enzyme-activity decay (Supplementary Methods S3), in seconds.
HALF_LIFE_S = 14400.0
LIFE_TIME_S = 86400.0

# Optimisation coordinates.  k_cat, K_m, K_a and K_i are fitted in log
# coordinates unless their range spans less than a factor of 10 (then linear);
# the coupling factors alpha are fitted in a logit coordinate.
_LOG_PREFIXES = ("K_cat_", "K_m_", "K_i_", "K_a_")
_EPS = 1e-9
_FLOOR_K = 0.1      # K_m, K_i, K_a >= 0.1 uM
_FLOOR_KCAT = 0.01  # k_cat >= 0.01 1/s


def load_ranges(path: Path = DATA / "parameter_ranges.yaml") -> dict[str, dict[str, float]]:
    """Permitted range of each parameter (Table S2)."""
    return yaml.safe_load(Path(path).read_text())


def linear_parameters(ranges: dict) -> set[str]:
    return {p for p in PARAMETERS if p.startswith(_LOG_PREFIXES)
            and float(ranges[p]["max"]) / max(float(ranges[p]["min"]), _EPS) < 10.0}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


def _logit(y: float) -> float:
    y = float(np.clip(y, _EPS, 1.0 - _EPS))
    return float(np.log(y / (1.0 - y)))


class Coordinates:
    """Maps between physical parameter values and optimisation coordinates."""

    def __init__(self, ranges: dict):
        self.ranges = ranges
        self.linear = linear_parameters(ranges)

    def kind(self, p: str) -> str:
        if p.startswith("alpha_"):
            return "logit"
        if p.startswith(_LOG_PREFIXES) and p not in self.linear:
            return "log"
        return "lin"

    def _lo_hi(self, p):
        return float(self.ranges[p]["min"]), float(self.ranges[p]["max"])

    def to_theta(self, phys: dict) -> dict:
        out = {}
        for p in PARAMETERS:
            x = float(phys[p])
            lo, hi = self._lo_hi(p)
            k = self.kind(p)
            if k == "log":
                out[p] = float(np.log(max(x, _EPS)))
            elif k == "logit":
                out[p] = _logit((x - lo) / max(hi - lo, _EPS))
            else:
                out[p] = x
        return out

    def to_physical(self, theta: dict, names=None) -> dict:
        out = {}
        for p in names or PARAMETERS:
            t = float(theta[p])
            lo, hi = self._lo_hi(p)
            k = self.kind(p)
            if k == "log":
                out[p] = float(np.exp(t))
            elif k == "logit":
                out[p] = float(lo + (hi - lo) * _sigmoid(t))
            else:
                out[p] = t
        return out

    def gradient(self, theta: dict, phys_grad: dict) -> dict:
        """Chain rule from dL/d(parameter) to dL/d(coordinate)."""
        out = {}
        for p in PARAMETERS:
            g = float(phys_grad.get(p, 0.0))
            t = float(theta[p])
            lo, hi = self._lo_hi(p)
            k = self.kind(p)
            if k == "log":
                out[p] = float(g * np.exp(t))
            elif k == "logit":
                s = _sigmoid(t)
                out[p] = float(g * (hi - lo) * s * (1.0 - s))
            else:
                out[p] = g
        return out

    def clamp(self, theta: dict) -> dict:
        """Keep every coordinate inside its permitted box and absolute floors."""
        out = {}
        for p in PARAMETERS:
            t = float(theta[p])
            lo, hi = self._lo_hi(p)
            k = self.kind(p)
            if k == "log":
                t = min(max(t, np.log(max(lo, _EPS))), np.log(max(hi, _EPS)))
            elif k == "logit":
                z = _logit(1.0 - _EPS)
                t = max(min(t, z), -z)
            else:
                t = max(min(t, hi), lo)
            out[p] = t
        for p in PARAMETERS:
            if not p.startswith(_LOG_PREFIXES):
                continue
            k = self.kind(p)
            x = float(np.exp(out[p])) if k == "log" else float(out[p])
            x = max(x, _FLOOR_KCAT if p.startswith("K_cat_") else _FLOOR_K)
            lo, hi = self._lo_hi(p)
            if k == "log":
                t = float(np.log(max(x, _EPS)))
                out[p] = min(max(t, np.log(max(lo, _EPS))), np.log(max(hi, _EPS)))
            else:
                out[p] = max(min(x, hi), lo)
        return out

    def span(self, p: str) -> float:
        """Width of the permitted range in optimisation coordinates."""
        lo, hi = self._lo_hi(p)
        a = self.to_theta_one(p, lo)
        b = self.to_theta_one(p, hi)
        return abs(b - a)

    def to_theta_one(self, p: str, x: float) -> float:
        lo, hi = self._lo_hi(p)
        k = self.kind(p)
        if k == "log":
            return float(np.log(max(x, _EPS)))
        if k == "logit":
            return _logit((x - lo) / max(hi - lo, _EPS))
        return float(x)

    def learning_rate(self, divisor: float = 50.0) -> np.ndarray:
        """Per-coordinate Adam step: 1/50 of the coordinate's range."""
        return np.array([self.span(p) / max(divisor, 1.0) for p in PARAMETERS])

    def random_physical(self, rng: np.random.Generator) -> dict:
        """One uniformly drawn parameter set (uniform in its coordinate)."""
        out = {}
        for p in PARAMETERS:
            lo, hi = self._lo_hi(p)
            u = float(rng.uniform())
            if p.startswith("alpha_") or p in self.linear:
                out[p] = float(lo + u * (hi - lo))
            else:
                out[p] = float(np.exp(np.log(max(lo, 1e-12))
                                      + u * (np.log(max(hi, 1e-12)) - np.log(max(lo, 1e-12)))))
        return out
