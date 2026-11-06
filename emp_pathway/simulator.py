"""Differentiable simulator of the reconstituted ten-enzyme EMP pathway.

One `Pathway` object simulates many assay wells at once.  Every well carries
its own copy of the 52 kinetic parameters, so several dose ladders, parameter
sets or finite-difference perturbations can share one forward pass.

Numerics (Supplementary Methods S3):
  * explicit Euler with a 0.1 s sub-step, 60 sub-steps per stored 6 s step;
  * every flux is capped smoothly at the substrate available in one sub-step,
    a normalised p-norm (p = 8) approximation of min(v, S/dt);
  * enzyme-activity decay: a(t) halves over 4 h, then falls to 1 % of that by 24 h;
  * reverse-mode gradients by Taichi automatic differentiation, replaying each
    stored 6 s step from a checkpoint of its starting state.

The rate laws are written in single-assignment form (no variable is assigned
twice inside a @ti.func), as Taichi's reverse mode requires.
"""
import numpy as np
import taichi as ti

from .model import HALF_LIFE_S, LIFE_TIME_S, PARAMETERS, SPECIES

NADH = SPECIES.index("NADH")
ATP = SPECIES.index("ATP")


@ti.data_oriented
class Pathway:
    def __init__(self, n_wells: int, n_targets: int, n_steps: int,
                 dt_sub: float = 0.1, n_substeps: int = 60,
                 curve_offset: float = 50.0):
        self.n_wells = n_wells
        self.n_targets = n_targets
        self.n_steps = n_steps
        self.dt_sub = dt_sub
        self.n_substeps = n_substeps
        self.eps = 1e-8
        self.limiter_power = 8.0
        self.limiter_eps = 1e-8
        self.curve_offset = float(curve_offset)
        self.h_s3 = 2
        self.h_s10 = 2
        self.decay_tail = 0.01

        state = ti.types.struct(
            Glucose=ti.f32, G6P=ti.f32, F6P=ti.f32, FBP=ti.f32, DHAP=ti.f32,
            G3P=ti.f32, NAD=ti.f32, NADH=ti.f32, One3BPG=ti.f32, Pi=ti.f32,
            ADP=ti.f32, ThreePG=ti.f32, ATP=ti.f32, TwoPG=ti.f32, PEP=ti.f32,
            Pyruvate=ti.f32)
        self.c = state.field(shape=(n_substeps + 1, n_wells), layout=ti.Layout.SOA,
                             needs_grad=True)
        enzymes = ti.types.struct(E1=ti.f32, E2=ti.f32, E3=ti.f32, E4=ti.f32,
                                  E5=ti.f32, E6=ti.f32, E7=ti.f32, E8=ti.f32,
                                  E9=ti.f32, E10=ti.f32)
        self.e = enzymes.field(shape=(n_wells,), layout=ti.Layout.SOA, needs_grad=True)
        self.params = {p: ti.field(shape=(n_wells,), dtype=ti.f32, needs_grad=True)
                       for p in PARAMETERS}

        # Checkpoints: the state at the start and at the end of every 6 s step.
        self.n_species = len(SPECIES)
        self.start_cache = np.zeros((n_steps, n_wells, self.n_species), np.float32)
        self.end_cache = np.zeros((n_steps, n_wells, self.n_species), np.float32)
        self._row = np.zeros((n_wells, self.n_species), np.float32)

        # Objectives: curve loss, dilution-response loss and the NADH floor.
        self.curve_terms = ti.field(shape=(n_wells, n_targets), dtype=ti.f32,
                                    needs_grad=True)
        self.target_nadh = ti.field(shape=(n_wells, n_targets), dtype=ti.f32)
        self.response_loss = ti.field(shape=(), dtype=ti.f32, needs_grad=True)
        self.floor_penalty = ti.field(shape=(), dtype=ti.f32, needs_grad=True)
        self.response_weight = 0.0
        self.floor_weight = 0.0
        self._pairs = None
        self.target_steps: list[int] = []
        self._target_index: dict[int, int] = {}

    # ── enzyme activity ────────────────────────────────────────────────────
    def activity(self, t_global: int, t_sub: int) -> float:
        t = (t_global * self.n_substeps + t_sub) * self.dt_sub
        if t <= HALF_LIFE_S:
            return float(2.0 ** (-t / HALF_LIFE_S))
        if t <= LIFE_TIME_S:
            frac = (t - HALF_LIFE_S) / (LIFE_TIME_S - HALF_LIFE_S)
            return float(0.5 * self.decay_tail ** frac)
        return 0.0

    # ── smooth substrate cap ───────────────────────────────────────────────
    @ti.func
    def cap(self, rate, capacity):
        """Smooth min(rate, capacity) for non-negative inputs."""
        p = ti.static(self.limiter_power)
        inv_p = ti.static(1.0 / self.limiter_power)
        eps = ti.static(self.limiter_eps)
        r = ti.max(rate, 0.0)
        c = ti.max(capacity, 0.0)
        scale = r + c + eps
        rate_n = r / scale
        cap_n = c / scale
        denom = ti.pow(ti.pow(rate_n, p) + ti.pow(cap_n, p) + eps, inv_p)
        return scale * rate_n * cap_n / denom

    # ── rate laws (Supplementary Methods S2) ───────────────────────────────
    @ti.func
    def v1(self, e, a, glc, atp, kcat, km_glc, km_atp):
        V = kcat * e * a
        num = V * glc * atp
        den = km_glc * atp + km_atp * glc + glc * atp + self.eps
        return num / den

    @ti.func
    def uni_uni(self, e, a, s, p, kcat_f, kcat_r, km_s, km_p):
        """Reversible Michaelis-Menten with independent forward and reverse constants."""
        Vf = kcat_f * e * a
        Vr = kcat_r * e * a
        den = 1 + s / km_s + p / km_p
        return ti.Vector([(Vf * s / km_s) / den, (Vr * p / km_p) / den])

    @ti.func
    def v3(self, e, a, f6p, atp, adp, pep, al_adp, al_pep, al_atp,
           kcat, km_f6p, km_atp, ka_adp, ki_pep, ki_atp):
        V = kcat * e * a
        f6p_h = (f6p / km_f6p) ** self.h_s3
        atp_hm = (atp / km_atp) ** self.h_s3
        adp_h = (adp / ka_adp) ** self.h_s3
        pep_h = (pep / ki_pep) ** self.h_s3
        atp_hi = (atp / ki_atp) ** self.h_s3
        d1 = ((1 + adp_h) * (1 + pep_h) * (1 + atp_hi)
              / ((1 + (al_adp ** (4 * self.h_s3)) * adp_h)
                 * (1 + (al_pep ** (4 * self.h_s3)) * pep_h)
                 * (1 + (al_atp ** (4 * self.h_s3)) * atp_hi)))
        d2 = (((1 + (al_adp ** (2 * self.h_s3)) * adp_h)
               / (1 + (al_adp ** (4 * self.h_s3)) * adp_h))
              * ((1 + (al_pep ** (2 * self.h_s3)) * pep_h)
                 / (1 + (al_pep ** (4 * self.h_s3)) * pep_h))
              * ((1 + (al_atp ** (2 * self.h_s3)) * atp_hi)
                 / (1 + (al_atp ** (4 * self.h_s3)) * atp_hi))
              * (f6p_h + atp_hm))
        d3 = f6p_h * atp_hm
        return V * (f6p_h * atp_hm) / (d1 + d2 + d3)

    @ti.func
    def v4(self, e, a, fbp, dhap, g3p, kcat_fbp, kcat_dhap, km_fbp, km_dhap, km_g3p):
        Vf = kcat_fbp * e * a
        Vr = kcat_dhap * e * a
        den = (1 + fbp / km_fbp + (dhap / km_dhap) * (g3p / km_g3p) + dhap / km_dhap)
        return ti.Vector([(Vf * fbp / km_fbp) / den,
                          (Vr * dhap * g3p / (km_dhap * km_g3p)) / den])

    @ti.func
    def bi_bi(self, e, a, s1, s2, p1, p2, kcat_f, kcat_r, km_s1, km_s2, km_p1, km_p2):
        """Reversible random bi-bi (GAPDH)."""
        Vf = kcat_f * e * a
        Vr = kcat_r * e * a
        x1 = s1 / km_s1
        x2 = s2 / km_s2
        y1 = p1 / km_p1
        y2 = p2 / km_p2
        den = 1 + x1 + x2 + y1 + y2 + x1 * x2 + y1 * y2
        return ti.Vector([Vf * x1 * x2 / den, Vr * y1 * y2 / den])

    @ti.func
    def v7(self, e, a, bpg, adp, pg3, atp, pi, kcat_f, kcat_r, km_bpg, km_adp, km_3pg, km_atp):
        """Random bi-bi (PGK) times the empirical phosphate factor."""
        Vf = kcat_f * e * a
        Vr = kcat_r * e * a
        x1 = bpg / km_bpg
        x2 = adp / km_adp
        y1 = pg3 / km_3pg
        y2 = atp / km_atp
        den = 1 + x1 + x2 + y1 + y2 + x1 * x2 + y1 * y2
        f = 0.0000000008 * ti.pow(pi, 2.226) * ti.pow(1.144, -0.0005 * pi)
        return ti.Vector([f * Vf * x1 * x2 / den, f * Vr * y1 * y2 / den])

    @ti.func
    def v10(self, e, a, pep, adp, atp, fbp, al_fbp, al_atp,
            kcat, km_pep, km_adp, ka_fbp, ki_atp):
        V = kcat * e * a
        pep_h = (pep / km_pep) ** self.h_s10
        adp_h = (adp / km_adp) ** self.h_s10
        fbp_h = (fbp / ka_fbp) ** self.h_s10
        atp_h = (atp / ki_atp) ** self.h_s10
        d1 = ((1 + fbp_h) * (1 + atp_h)
              / ((1 + (al_fbp ** (4 * self.h_s10)) * fbp_h)
                 * (1 + (al_atp ** (4 * self.h_s10)) * atp_h)))
        d2 = (((1 + (al_fbp ** (2 * self.h_s10)) * fbp_h)
               / (1 + (al_fbp ** (4 * self.h_s10)) * fbp_h))
              * ((1 + (al_atp ** (2 * self.h_s10)) * atp_h)
                 / (1 + (al_atp ** (4 * self.h_s10)) * atp_h))
              * (pep_h + adp_h))
        d3 = pep_h * adp_h
        return V * (pep_h * adp_h) / (d1 + d2 + d3)

    # ── one Euler sub-step, all ten reactions in pathway order ─────────────
    @ti.kernel
    def substep(self, t: ti.i32, a: ti.f32):
        for i in range(self.n_wells):
            dt = self.dt_sub
            Glc = self.c[t, i].Glucose
            G6P = self.c[t, i].G6P
            F6P = self.c[t, i].F6P
            FBP = self.c[t, i].FBP
            DHAP = self.c[t, i].DHAP
            G3P = self.c[t, i].G3P
            NAD = self.c[t, i].NAD
            NADH = self.c[t, i].NADH
            BPG = self.c[t, i].One3BPG
            Pi = self.c[t, i].Pi
            ADP = self.c[t, i].ADP
            PG3 = self.c[t, i].ThreePG
            ATP = self.c[t, i].ATP
            PG2 = self.c[t, i].TwoPG
            PEP = self.c[t, i].PEP
            Pyr = self.c[t, i].Pyruvate

            # E1 glucokinase: Glucose + ATP -> G6P + ADP
            v_raw = self.v1(self.e[i].E1, a, Glc, ATP, self.params["K_cat_Glucose_s1"][i],
                            self.params["K_m_Glucose_s1"][i], self.params["K_m_ATP_s1"][i])
            v = self.cap(v_raw, self.cap(Glc / dt, ATP / dt))
            Glc -= v * dt; ATP -= v * dt; G6P += v * dt; ADP += v * dt

            # E2 phosphoglucose isomerase: G6P <-> F6P
            r = self.uni_uni(self.e[i].E2, a, G6P, F6P, self.params["K_cat_G6P_s2"][i],
                             self.params["K_cat_F6P_s2"][i], self.params["K_m_G6P_s2"][i], self.params["K_m_F6P_s2"][i])
            v = self.cap(r[0], G6P / dt) - self.cap(r[1], F6P / dt)
            G6P -= v * dt; F6P += v * dt

            # E3 phosphofructokinase: F6P + ATP -> FBP + ADP
            v_raw = self.v3(self.e[i].E3, a, F6P, ATP, ADP, PEP,
                            self.params["alpha_ADP_s3"][i], self.params["alpha_PEP_s3"][i], self.params["alpha_ATP_s3"][i],
                            self.params["K_cat_f_s3"][i], self.params["K_m_F6P_s3"][i], self.params["K_m_ATP_s3"][i],
                            self.params["K_a_ADP_s3"][i], self.params["K_i_PEP_s3"][i], self.params["K_i_ATP_s3"][i])
            v = self.cap(v_raw, self.cap(F6P / dt, ATP / dt))
            F6P -= v * dt; ATP -= v * dt; FBP += v * dt; ADP += v * dt

            # E4 aldolase: FBP <-> DHAP + G3P
            r = self.v4(self.e[i].E4, a, FBP, DHAP, G3P, self.params["K_cat_FBP_s4"][i],
                        self.params["K_cat_DHAP_s4"][i], self.params["K_m_FBP_s4"][i], self.params["K_m_DHAP_s4"][i],
                        self.params["K_m_G3P_s4"][i])
            v = self.cap(r[0], FBP / dt) - self.cap(r[1], self.cap(DHAP / dt, G3P / dt))
            FBP -= v * dt; DHAP += v * dt; G3P += v * dt

            # E5 triosephosphate isomerase: DHAP <-> G3P
            r = self.uni_uni(self.e[i].E5, a, DHAP, G3P, self.params["K_cat_DHAP_s5"][i],
                             self.params["K_cat_G3P_s5"][i], self.params["K_m_DHAP_s5"][i], self.params["K_m_G3P_s5"][i])
            v = self.cap(r[0], DHAP / dt) - self.cap(r[1], G3P / dt)
            DHAP -= v * dt; G3P += v * dt

            # E6 GAPDH: G3P + NAD+ <-> 1,3-BPG + NADH  (the measured NADH)
            r = self.bi_bi(self.e[i].E6, a, G3P, NAD, BPG, NADH,
                           self.params["K_cat_G3P_s6"][i], self.params["K_cat_NADH_s6"][i], self.params["K_m_G3P_s6"][i],
                           self.params["K_m_NAD_s6"][i], self.params["K_m_13BPG_s6"][i], self.params["K_m_NADH_s6"][i])
            v = (self.cap(r[0], self.cap(G3P / dt, NAD / dt))
                 - self.cap(r[1], self.cap(BPG / dt, NADH / dt)))
            G3P -= v * dt; NAD -= v * dt; NADH += v * dt; BPG += v * dt

            # E7 phosphoglycerate kinase: 1,3-BPG + ADP <-> 3PG + ATP,
            # multiplied by the empirical phosphate factor
            r = self.v7(self.e[i].E7, a, BPG, ADP, PG3, ATP, Pi,
                           self.params["K_cat_ADP_s7"][i], self.params["K_cat_ATP_s7"][i], self.params["K_m_13BPG_s7"][i],
                           self.params["K_m_ADP_s7"][i], self.params["K_m_3PG_s7"][i], self.params["K_m_ATP_s7"][i])
            v = (self.cap(r[0], self.cap(BPG / dt, ADP / dt))
                 - self.cap(r[1], self.cap(PG3 / dt, ATP / dt)))
            BPG -= v * dt; ADP -= v * dt; PG3 += v * dt; ATP += v * dt

            # E8 phosphoglycerate mutase: 3PG <-> 2PG
            r = self.uni_uni(self.e[i].E8, a, PG3, PG2, self.params["K_cat_3PG_s8"][i],
                             self.params["K_cat_2PG_s8"][i], self.params["K_m_3PG_s8"][i], self.params["K_m_2PG_s8"][i])
            v = self.cap(r[0], PG3 / dt) - self.cap(r[1], PG2 / dt)
            PG3 -= v * dt; PG2 += v * dt

            # E9 enolase: 2PG <-> PEP
            r = self.uni_uni(self.e[i].E9, a, PG2, PEP, self.params["K_cat_2PG_s9"][i],
                             self.params["K_cat_PEP_s9"][i], self.params["K_m_2PG_s9"][i], self.params["K_m_PEP_s9"][i])
            v = self.cap(r[0], PG2 / dt) - self.cap(r[1], PEP / dt)
            PG2 -= v * dt; PEP += v * dt

            # E10 pyruvate kinase: PEP + ADP -> pyruvate + ATP
            v_raw = self.v10(self.e[i].E10, a, PEP, ADP, ATP, FBP,
                             self.params["alpha_FBP_s10"][i], self.params["alpha_ATP_s10"][i], self.params["K_cat_f_s10"][i],
                             self.params["K_m_PEP_s10"][i], self.params["K_m_ADP_s10"][i], self.params["K_a_FBP_s10"][i],
                             self.params["K_i_ATP_s10"][i])
            v = self.cap(v_raw, self.cap(PEP / dt, ADP / dt))
            PEP -= v * dt; ADP -= v * dt; Pyr += v * dt; ATP += v * dt

            self.c[t + 1, i].Glucose = Glc
            self.c[t + 1, i].G6P = G6P
            self.c[t + 1, i].F6P = F6P
            self.c[t + 1, i].FBP = FBP
            self.c[t + 1, i].DHAP = DHAP
            self.c[t + 1, i].G3P = G3P
            self.c[t + 1, i].NAD = NAD
            self.c[t + 1, i].NADH = NADH
            self.c[t + 1, i].One3BPG = BPG
            self.c[t + 1, i].Pi = Pi
            self.c[t + 1, i].ADP = ADP
            self.c[t + 1, i].ThreePG = PG3
            self.c[t + 1, i].ATP = ATP
            self.c[t + 1, i].TwoPG = PG2
            self.c[t + 1, i].PEP = PEP
            self.c[t + 1, i].Pyruvate = Pyr

    # ── state <-> checkpoint copies ────────────────────────────────────────
    @ti.kernel
    def _read_state(self, slot: ti.i32, out: ti.types.ndarray()):
        for i in range(self.n_wells):
            out[i, 0] = self.c[slot, i].Glucose
            out[i, 1] = self.c[slot, i].G6P
            out[i, 2] = self.c[slot, i].F6P
            out[i, 3] = self.c[slot, i].FBP
            out[i, 4] = self.c[slot, i].DHAP
            out[i, 5] = self.c[slot, i].G3P
            out[i, 6] = self.c[slot, i].NAD
            out[i, 7] = self.c[slot, i].NADH
            out[i, 8] = self.c[slot, i].One3BPG
            out[i, 9] = self.c[slot, i].Pi
            out[i, 10] = self.c[slot, i].ADP
            out[i, 11] = self.c[slot, i].ThreePG
            out[i, 12] = self.c[slot, i].ATP
            out[i, 13] = self.c[slot, i].TwoPG
            out[i, 14] = self.c[slot, i].PEP
            out[i, 15] = self.c[slot, i].Pyruvate

    @ti.kernel
    def _write_state(self, slot: ti.i32, src: ti.types.ndarray()):
        for i in range(self.n_wells):
            self.c[slot, i].Glucose = src[i, 0]
            self.c[slot, i].G6P = src[i, 1]
            self.c[slot, i].F6P = src[i, 2]
            self.c[slot, i].FBP = src[i, 3]
            self.c[slot, i].DHAP = src[i, 4]
            self.c[slot, i].G3P = src[i, 5]
            self.c[slot, i].NAD = src[i, 6]
            self.c[slot, i].NADH = src[i, 7]
            self.c[slot, i].One3BPG = src[i, 8]
            self.c[slot, i].Pi = src[i, 9]
            self.c[slot, i].ADP = src[i, 10]
            self.c[slot, i].ThreePG = src[i, 11]
            self.c[slot, i].ATP = src[i, 12]
            self.c[slot, i].TwoPG = src[i, 13]
            self.c[slot, i].PEP = src[i, 14]
            self.c[slot, i].Pyruvate = src[i, 15]

    @ti.kernel
    def _write_enzymes(self, src: ti.types.ndarray()):
        for i in range(self.n_wells):
            self.e[i].E1 = src[i, 0]
            self.e[i].E2 = src[i, 1]
            self.e[i].E3 = src[i, 2]
            self.e[i].E4 = src[i, 3]
            self.e[i].E5 = src[i, 4]
            self.e[i].E6 = src[i, 5]
            self.e[i].E7 = src[i, 6]
            self.e[i].E8 = src[i, 7]
            self.e[i].E9 = src[i, 8]
            self.e[i].E10 = src[i, 9]

    @ti.kernel
    def _carry_over(self):
        """The end of one 6 s step becomes the start of the next."""
        for i in range(self.n_wells):
            self.c[0, i].Glucose = self.c[self.n_substeps, i].Glucose
            self.c[0, i].G6P = self.c[self.n_substeps, i].G6P
            self.c[0, i].F6P = self.c[self.n_substeps, i].F6P
            self.c[0, i].FBP = self.c[self.n_substeps, i].FBP
            self.c[0, i].DHAP = self.c[self.n_substeps, i].DHAP
            self.c[0, i].G3P = self.c[self.n_substeps, i].G3P
            self.c[0, i].NAD = self.c[self.n_substeps, i].NAD
            self.c[0, i].NADH = self.c[self.n_substeps, i].NADH
            self.c[0, i].One3BPG = self.c[self.n_substeps, i].One3BPG
            self.c[0, i].Pi = self.c[self.n_substeps, i].Pi
            self.c[0, i].ADP = self.c[self.n_substeps, i].ADP
            self.c[0, i].ThreePG = self.c[self.n_substeps, i].ThreePG
            self.c[0, i].ATP = self.c[self.n_substeps, i].ATP
            self.c[0, i].TwoPG = self.c[self.n_substeps, i].TwoPG
            self.c[0, i].PEP = self.c[self.n_substeps, i].PEP
            self.c[0, i].Pyruvate = self.c[self.n_substeps, i].Pyruvate

    # ── set-up ─────────────────────────────────────────────────────────────
    def set_assay(self, inputs, enzymes):
        """Initial concentrations (uM): `inputs` per ASSAY_INPUTS, `enzymes` E1-E10."""
        x = np.zeros((self.n_wells, self.n_species), np.float32)
        inputs = np.asarray(inputs, np.float32)
        for j, name in enumerate(("Glucose", "FBP", "DHAP", "G3P", "NAD", "ADP",
                                  "ATP", "Pi", "3PG")):
            x[:, SPECIES.index(name)] = inputs[:, j]
        self.c.fill(0.0)
        self._write_state(0, x)
        self._write_enzymes(np.ascontiguousarray(np.asarray(enzymes, np.float32)[:, :10]))
        self.start_cache.fill(0.0)
        self.end_cache.fill(0.0)
        self._initial = x

    def set_parameters(self, values: dict[str, np.ndarray]):
        """One value per well for each of the 52 parameters."""
        for p in PARAMETERS:
            self.params[p].from_numpy(np.asarray(values[p], np.float32))

    def set_targets(self, target_steps, nadh):
        """Measured NADH (uM) at the stored steps that carry a measurement."""
        self.target_steps = list(target_steps)
        self._target_index = {s: j for j, s in enumerate(self.target_steps)}
        self.target_nadh.from_numpy(np.asarray(nadh, np.float32))

    def set_response_pairs(self, pair_a, pair_b, target_diffs, weights, norm_factors):
        """Adjacent-dose pairs for the dilution-response loss."""
        self._pairs = (np.asarray(pair_a, np.int32), np.asarray(pair_b, np.int32),
                       np.asarray(target_diffs, np.float32),
                       np.asarray(weights, np.float32),
                       np.asarray(norm_factors, np.float32))

    # ── forward simulation ─────────────────────────────────────────────────
    def run(self):
        for t in range(self.n_steps):
            self._read_state(0, self._row)
            self.start_cache[t] = self._row
            for s in range(self.n_substeps):
                self.substep(s, self.activity(t, s))
            self._read_state(self.n_substeps, self._row)
            self.end_cache[t] = self._row
            self._carry_over()

    def trace(self, species: str = "NADH") -> np.ndarray:
        """(wells, targets) concentrations at the measured time points."""
        k = SPECIES.index(species)
        return self.end_cache[self.target_steps, :, k].astype(np.float64).T

    # ── objectives ─────────────────────────────────────────────────────────
    @ti.kernel
    def _curve(self, j: ti.i32):
        for i in range(self.n_wells):
            c = ti.static(self.curve_offset)
            sim = ti.max(self.c[self.n_substeps, i].NADH + c, 1e-6)
            tgt = ti.max(self.target_nadh[i, j] + c, 1e-6)
            e = ti.log(sim / tgt)
            self.curve_terms[i, j] += e * e

    @ti.kernel
    def _response(self, j: ti.i32, pa: ti.types.ndarray(dtype=ti.i32),
                  pb: ti.types.ndarray(dtype=ti.i32), td: ti.types.ndarray(dtype=ti.f32),
                  w: ti.types.ndarray(dtype=ti.f32), nf: ti.types.ndarray(dtype=ti.f32)):
        for p in range(pa.shape[0]):
            a = pa[p]
            b = pb[p]
            sim_a = self.c[self.n_substeps, a].NADH
            sim_b = self.c[self.n_substeps, b].NADH
            self.response_loss[None] += (w[p] * nf[p] * (sim_b - sim_a - td[p, j]) ** 2
                                         / float(pa.shape[0] * self.n_targets))

    @ti.kernel
    def _floor(self):
        for i in range(self.n_wells):
            deficit = ti.max(0.0, 1.0 - self.c[self.n_substeps, i].NADH)
            self.floor_penalty[None] += deficit * deficit / float(self.n_wells)

    def _curve_seed(self) -> float:
        """d sqrt(mean(curve terms)) / d(each term), from the stored trajectory."""
        sim = self.end_cache[self.target_steps, :, NADH].astype(np.float64).T
        tgt = self.target_nadh.to_numpy().astype(np.float64)
        c = self.curve_offset
        r = np.log(np.maximum(sim + c, 1e-6) / np.maximum(tgt + c, 1e-6))
        mse = float(np.mean(r * r))
        if not np.isfinite(mse) or mse <= self.eps:
            return 0.0
        return 1.0 / (2.0 * float(r.size) * float(np.sqrt(mse)))

    def gradients(self, objective: str, floor: bool) -> dict[str, np.ndarray]:
        """Reverse-mode dL/d(parameter) per well after `run()`.

        objective  'curve'    L_curve = sqrt(mean_ij log((x+c)/(y+c))^2), or
                   'response' the weighted dilution-response loss;
        floor      also include the NADH floor penalty (weight `floor_weight`).
        """
        seed = self._curve_seed() if objective == "curve" else 0.0
        self.curve_terms.fill(0.0)
        self.curve_terms.grad.fill(seed)
        self.response_loss.fill(0.0)
        self.floor_penalty.fill(0.0)
        self.c.grad.fill(0.0)
        self.e.grad.fill(0.0)
        for p in PARAMETERS:
            self.params[p].grad.fill(0.0)
        use_response = objective == "response" and self.response_weight > 0 \
            and self._pairs is not None
        use_floor = floor and self.floor_weight > 0

        for t in range(self.n_steps - 1, -1, -1):
            self._row[:] = self.start_cache[t]
            self._write_state(0, self._row)
            for s in range(self.n_substeps):
                self.substep(s, self.activity(t, s))
            if use_floor and t == self.n_steps - 1:
                self.floor_penalty.fill(0.0)
                self._floor()
                self.floor_penalty.grad.fill(self.floor_weight)
                self._floor.grad()
            if t in self._target_index:
                j = self._target_index[t]
                if objective == "curve":
                    self._curve(j)
                    self._curve.grad(j)
                if use_response:
                    self.response_loss.grad.fill(self.response_weight)
                    self._response(j, *self._pairs)
                    self._response.grad(j, *self._pairs)
            for s in range(self.n_substeps - 1, -1, -1):
                self.substep.grad(s, self.activity(t, s))
            self.e.grad.fill(0.0)
            if t > 0:
                self._pass_back()
                self._clear_inner()
        return {p: self.params[p].grad.to_numpy().astype(np.float64) for p in PARAMETERS}

    # Adjoint hand-over between stored steps.  The min/max bounds (1e12) never
    # bind; they keep the copy a recorded operation on Taichi's tape.
    @ti.kernel
    def _pass_back(self):
        for i in range(self.n_wells):
            self.c.grad[self.n_substeps, i].Glucose = ti.min(ti.max(self.c.grad[0, i].Glucose * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].G6P = ti.min(ti.max(self.c.grad[0, i].G6P * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].F6P = ti.min(ti.max(self.c.grad[0, i].F6P * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].FBP = ti.min(ti.max(self.c.grad[0, i].FBP * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].DHAP = ti.min(ti.max(self.c.grad[0, i].DHAP * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].G3P = ti.min(ti.max(self.c.grad[0, i].G3P * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].NAD = ti.min(ti.max(self.c.grad[0, i].NAD * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].NADH = ti.min(ti.max(self.c.grad[0, i].NADH * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].One3BPG = ti.min(ti.max(self.c.grad[0, i].One3BPG * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].Pi = ti.min(ti.max(self.c.grad[0, i].Pi * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].ADP = ti.min(ti.max(self.c.grad[0, i].ADP * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].ThreePG = ti.min(ti.max(self.c.grad[0, i].ThreePG * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].ATP = ti.min(ti.max(self.c.grad[0, i].ATP * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].TwoPG = ti.min(ti.max(self.c.grad[0, i].TwoPG * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].PEP = ti.min(ti.max(self.c.grad[0, i].PEP * 1.0, -1e12), 1e12)
            self.c.grad[self.n_substeps, i].Pyruvate = ti.min(ti.max(self.c.grad[0, i].Pyruvate * 1.0, -1e12), 1e12)

    @ti.kernel
    def _clear_inner(self):
        for j, i in ti.ndrange(self.n_substeps, self.n_wells):
            self.c.grad[j, i].Glucose = 0.0
            self.c.grad[j, i].G6P = 0.0
            self.c.grad[j, i].F6P = 0.0
            self.c.grad[j, i].FBP = 0.0
            self.c.grad[j, i].DHAP = 0.0
            self.c.grad[j, i].G3P = 0.0
            self.c.grad[j, i].NAD = 0.0
            self.c.grad[j, i].NADH = 0.0
            self.c.grad[j, i].One3BPG = 0.0
            self.c.grad[j, i].Pi = 0.0
            self.c.grad[j, i].ADP = 0.0
            self.c.grad[j, i].ThreePG = 0.0
            self.c.grad[j, i].ATP = 0.0
            self.c.grad[j, i].TwoPG = 0.0
            self.c.grad[j, i].PEP = 0.0
            self.c.grad[j, i].Pyruvate = 0.0
