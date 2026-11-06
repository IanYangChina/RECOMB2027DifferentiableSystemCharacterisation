# Code and data: information analysis and next-experiment design for multi-enzyme pathway characterisation via differentiable modelling

This archive holds the code, the wet-lab data and the recorded results behind
the RECOMB 2027 paper *Enabling information analysis and next experiment
design for multi-enzyme pathway characterisation via differentiable
modelling* and its Supplementary Material. Section, figure and table numbers
below refer to that paper. Every file is listed with the place in the paper
where it is used or reported.

> **Tested environment.** The code has been verified only on Ubuntu 22.04.5 LTS
> (kernel 6.8), Python 3.11.14, Taichi 1.7.3 and NumPy 2.4.6. It was run on
> one NVIDIA GeForce RTX 5090 (driver 570.195.03, CUDA 12.8) and on an Intel
> Xeon w7-3455 CPU. It may not run without problems on other machines, other
> GPUs or other package versions.

## Installation

```bash
conda env create -f environment.yml      # Python 3.11, NumPy, PyYAML, openpyxl, Taichi 1.7.3
conda activate emp-pathway
cd scripts
python 02_fit_single_series.py --objective curve --series S-E3 --n_updates 2   # quick check
```

The scripts that simulate (01-04, 06, 07) run on the GPU by default
(`--arch cuda`). Add `--arch cpu` to run on the CPU, which is roughly ten
times slower. Scripts write to `outputs/`. Each output has the same format as
the matching file in `results/`, which holds the runs reported in the paper.

## Naming

| In the paper and in this archive | Meaning |
|---|---|
| E1-E10 | Glk, Pgi, PfkA, FbaA, TpiA, GapA, Pgk, GpmA, Eno, PykF (Fig. 1a) |
| S-E1 ... S-E10 | enzyme-dilution series: the named enzyme varied, all others at default (Table 1, Table S1) |
| S-Gluc, S-Gluc-E1 mid, S-Gluc-E1 high | glucose-dilution series at E1 = 0.0065, 0.013 and 0.026 µM |
| S-Gluc, S-FBP, S-DHAP, S-G3P, S-3PG (candidates) | prospective substrate-dilution series (Section 2.4); the glucose candidate is the S-Gluc ladder and composition |
| `K_cat_*`, `K_m_*`, `K_a_*`, `K_i_*`, `alpha_*`, suffix `_s1`...`_s10` | the 52 kinetic parameters and their reaction step (Table S2) |
| curve loss, L_curve | RMS of log((x + c)/(y + c)) over doses and times, c = 50 µM (Section 2.2) |
| dilution-response loss, L_response | weighted error of the NADH change between adjacent doses (Section 2.2) |
| complementary weight λ | weight of the part of the response gradient orthogonal to the curve gradient (Section 2.2) |
| ranking score, L_rank | w_c L_curve + w_r L_response, w = 1 / median over the 150 screened sets (Section 2.2) |
| set | one of the 30 screened initial parameter sets, numbered 0-29 (a "seed" in Fig. 2 and S5) |

Concentrations are in µM and times in minutes, except inside the simulator,
which works in seconds.

## Code: `emp_pathway/`

| File | Contents | Paper |
|---|---|---|
| `model.py` | species, enzymes, the 52 parameters by step, enzyme-activity decay constants, optimisation coordinates (log / linear / logit), absolute floors, Adam step sizes, uniform draws | Table S2, S2, S3 |
| `simulator.py` | differentiable Taichi simulator: ten rate laws, smooth substrate cap (p = 8), explicit Euler (0.1 s, 60 sub-steps per 6 s), enzyme-activity decay, curve and dilution-response losses, reverse-mode gradients with per-step checkpoints | Fig. 1, Section 2.2, S2, S3 |
| `series.py` | loading the thirteen series, 0-500 min fitting window, NADH from OD340 (/ 0.7989), joint panels, adjacent-dose pairs and their normalisation, per-series response weights, curve loss in µM | Table 1, S1, Section 2.2, Fig. 2 |
| `fitting.py` | single-series fits (curve, response or complementary objective), multi-series fits (mean of unit per-series directions), scoring fitted sets on every series | Sections 2.2-2.3, 3.1, 3.3 |
| `sensitivity.py` | central-difference Jacobians (h = 1 % of each coordinate's range), information matrices G and per-dose G_m, correlation form, effective rank, leading directions and \|cos\|, step shares, new-information fraction, information-first and transfer-first selection | Sections 2.3-2.4, 3.1-3.4, S4, S5 |

A small penalty on simulated NADH below 1 µM at the end of the fitting window
(weight 10) accompanies the curve gradient and the response-only objective.
It keeps fits away from the zero-flux state (`fitting.py`, `FLOOR_WEIGHT`).

## Scripts: `scripts/`

Run them from the `scripts/` folder, in order. Scripts 02-07 read their
inputs (initial sets, fits, evaluations, information matrices, selection,
parameter states) from the recorded `results/` by default, so any step can be
run on its own; pass your own `outputs/` files through the file options to
chain your own runs.

| Script | What it does | Paper | Approximate GPU time |
|---|---|---|---|
| `00_prepare_nadh_targets.py` | rebuilds `data/nadh_dilution_series.yaml` from the two workbooks | Table 1, S1 | seconds |
| `01_screen_initial_parameters.py` | draws 150 parameter sets (random seed 42), scores them on all thirteen series and keeps the best 30 by L_rank | Section 2.2 | about 5 min |
| `02_fit_single_series.py` | fits each series alone from all 30 sets for 20 Adam updates (`--objective curve`, `response` or `complementary --lam 0.25/0.5/1/2`) | Sections 2.2, 3.1; Fig. 2, Fig. 3a-b | 12-15 min per series and objective |
| `03_evaluate_fits.py` | scores each fitted set (after 20 updates) on all thirteen series, 0-500 min, in loss units and µM; use `--scale_window 1225` for the multi-series fits | Sections 3.1, 3.3; Fig. 2, Fig. 4c-d | 5-12 min per file |
| `04_parameter_to_nadh_sensitivity.py` | Jacobians of NADH for each series' five best sets at updates 0 and 19, giving G and the per-dose G_m | Sections 3.1-3.3; Fig. 3c-d, Fig. 4a-b | about 45 min |
| `05_select_series.py` | information-first and transfer-first panels | Sections 2.3, 3.3; Fig. 4b; S4 | seconds |
| `06_fit_multi_series.py` | joint fits of the first 2-7 series of each panel, 30 sets, 20 updates | Sections 2.3, 3.3; Fig. 4c-d | about 7 h per rule |
| `07_prospective_candidates.py` | NADH and ATP Jacobians of the five candidate series and of S-E3 and S-E8 at the 100 parameter states; new-information fractions | Sections 2.4, 3.4; Fig. 5; S5 | about 4 h |

## Data: `data/`

| File | Contents | Paper |
|---|---|---|
| `wet_lab/enzyme_dilution_series_NADH_mM.xlsx` | S-E1-S-E10 plate reads: series id, enzyme, µM per g/L, dose (µM), protein loading, then NADH (mM) at 128 reads to 1200 min | Table 1, Table S1, S1 |
| `wet_lab/glucose_dilution_series_OD340_mM.xlsx` | glucose plates; rows `E1-10,E1(1)`, `E1-10,E1(0.5)` and `E1-10,E1(0.25)` are S-Gluc-E1 high, S-Gluc-E1 mid and S-Gluc, and the second column (`G5`...`G0`) gives the glucose dose in mM, OD340 (mM) at 133 reads to 1225 min (the first at t = 0) | Table 1, Table S1, S1 |
| `nadh_dilution_series.yaml` | the thirteen series as used: per condition the dose, assay inputs (Glucose, FBP, DHAP, G3P, NAD+, ADP, ATP, Pi, 3PG), E1-E10 concentrations and OD340 (µM) on the nominal reading schedule | Table 1, Table S1, S1 |
| `parameter_ranges.yaml` | permitted range of each of the 52 parameters | Table S2 |
| `candidate_series.yaml` | doses and assay compositions of the prospective series | Section 2.4, S5 |

The workbooks contain only the reads used in the paper. In the enzyme
workbook, the dose column is the protein loading multiplied by the µM per
g/L factor.

## Recorded results: `results/`

These are the runs reported in the paper. `.npz` files open with
`numpy.load`. Parameter values are physical (units as in Table S2). Arrays
follow the order of their `fits` / `series`, `parameters` and set axes.

| File | Contents | Paper |
|---|---|---|
| `01_initial_parameter_sets.yaml` | the 30 screened sets (parameters, L_curve, L_response, L_rank on all series) and the ranking weights w_c, w_r | Section 2.2 |
| `02_single_series_fits/curve.npz`, `response.npz` | per series and set: `curve_loss`, `response_loss`, `ranking_score` at each of the 20 updates (before the update); `parameter_values` before updates 0 and 19 and after update 20 (`stored_updates`); `first_update_gradient`, the gradient of the objective (with the NADH floor term) at the first update, in optimisation coordinates | Sections 3.1-3.2; Fig. 2 (curve losses converted with `series.curve_loss_to_uM`), Fig. 3b |
| `02_single_series_fits/complementary_lambda{0.25,0.5,1,2}.npz` | the same losses for the complementary objective at each λ, with parameters after the last update only | Section 2.2, S4 |
| `03_evaluation/curve.csv`, `response.csv` | every single-series fitted set scored on all thirteen series (`fitted` = its own series, `unseen` = the others): L_curve, L_response, L_curve in µM | Section 3.1 (final losses, top-5 sets), Section 3.3, Fig. 3a, Fig. 4d |
| `03_evaluation/information_first.csv`, `transfer_first.csv` | the same for every multi-series panel (fit = panel, series joined by `+`); L_response with the response scale taken over 0-1225 min | Section 3.3, Fig. 4c-d |
| `04_sensitivity/curve.npz` | for each series' five best curve-objective sets (`sets`, by L_rank on that series): `G` (series, set, update 0/19, 52, 52) and `G_dose` (series, dose, 52, 52), the per-dose information at update 19 averaged over the five sets (doses in ascending order, NaN-padded to 8) | Sections 3.1-3.3; Fig. 3c-d, Fig. 4a-b; S4 |
| `05_selection/transfer_first_performance.csv` | per series: best λ (lowest `in_distribution`), median L_rank of its complementary single-series fits after 20 updates on itself (`in_distribution`) and on the twelve others (`transfer`); rows in ranking order, ties in mean rank keep this order | S4 |
| `05_selection/selection.json` | both selection rules: order, information gains, effective ranks, panel and λ per series | Section 3.3, Fig. 4b, S4 |
| `06_multi_series_fits/information_first.npz`, `transfer_first.npz` | per panel (first 2-7 series of each rule) and set: `parameter_values` after update 20 | Section 3.3, Fig. 4c-d |
| `07_prospective/parameter_states.csv` | the 100 parameter states: sets 1, 3, 2, 22 and 16 of the S-E3 + S-E8 fit (lowest mean L_rank over the two series after fitting, scored over the full 0-1225 min record) before each of their 20 updates | Section 2.4, S5 |
| `07_prospective/new_information.csv`, `prospective.npz` | per candidate, readout (NADH, ATP) and state: fraction of the candidate's information new relative to S-E3 + S-E8; `leading_direction` (52) and `rms_sensitivity` (52, RMS of J over doses and times) | Section 3.4, Fig. 5, S5 |

Notes:

* The "update u" of a recorded loss is the loss at the parameters update u
  starts from. Update 0 is the screened initial set.
* Fitted sets are scored on all series together, on the glucose series'
  time grid, with the enzyme series' measurements interpolated onto it.
* Computations are in single precision, so reruns agree with the recorded
  results to rounding level rather than bit for bit (as between the GPU and
  CPU runs in S3).
