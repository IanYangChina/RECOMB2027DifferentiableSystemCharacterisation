"""Build data/nadh_dilution_series.yaml from the two wet-lab workbooks.

    python 00_prepare_nadh_targets.py

Enzyme series (S-E1..S-E10): NADH in mM, converted to OD340 equivalents
(x 0.7989) and to uM.  Glucose series: OD340 in mM, converted to uM.  Each
condition carries its assay composition (Table 1, Table S1).  Time stamps are
the plate reader's nominal reading schedule: every 3 min to 180 min (186 min
for the glucose plate), then every 15 min (from 195 min, or 190 min).
"""
import argparse
from pathlib import Path

import openpyxl
import yaml

from _common import ROOT

OD340_PER_NADH = 0.7989
DEFAULT_ENZYMES = [0.0065, 0.31, 0.52, 0.47, 0.66, 0.51, 0.45, 0.63, 0.41, 0.37]
# Glucose, FBP, DHAP, G3P, NAD+, ADP, ATP, Pi, 3PG  (uM)
DEFAULT_INPUTS = [1250.0, 625.0, 0.0, 0.0, 3500.0, 10000.0, 0.0, 10000.0, 0.0]
GLUCOSE_PLATES = {"S-Gluc": ("E1-10,E1(0.25)", 0.0065),
                  "S-Gluc-E1 mid": ("E1-10,E1(0.5)", 0.013),
                  "S-Gluc-E1 high": ("E1-10,E1(1)", 0.026)}


def enzyme_series(path):
    rows = list(openpyxl.load_workbook(path, data_only=True).active.iter_rows(values_only=True))
    times = list(range(3, 181, 3)) + list(range(195, 1201, 15))
    assert len(times) == len([x for x in rows[0][5:] if x is not None])
    out = {}
    for r in rows[2:]:
        k = int(r[1][1:])
        enz = list(DEFAULT_ENZYMES)
        enz[k - 1] = float(r[3])
        s = out.setdefault(f"S-E{k}", {"varied": f"E{k}", "time_min": times, "conditions": []})
        s["conditions"].append({
            "dose_uM": float(r[3]), "inputs_uM": list(DEFAULT_INPUTS), "enzymes_uM": enz,
            "OD340_uM": [round(float(v) * 1000.0 * OD340_PER_NADH, 1) for v in r[5:5 + len(times)]]})
    return out


def glucose_series(path):
    rows = list(openpyxl.load_workbook(path, data_only=True).active.iter_rows(values_only=True))
    times = list(range(3, 187, 3)) + list(range(190, 1226, 15))
    assert len(times) == len([x for x in rows[0][3:] if x is not None])
    out = {}
    for name, (plate, e1) in GLUCOSE_PLATES.items():
        s = {"varied": "Glucose", "time_min": times, "conditions": []}
        for r in rows:
            if r[0] != plate:
                continue
            dose = float(r[1][1:]) * 1000.0
            inputs = list(DEFAULT_INPUTS)
            inputs[0] = dose
            enz = list(DEFAULT_ENZYMES)
            enz[0] = e1
            s["conditions"].append({
                "dose_uM": dose, "inputs_uM": inputs, "enzymes_uM": enz,
                # the t = 0 reading (column 3) precedes the first stored time point
                "OD340_uM": [float(v) * 1000.0 for v in r[3:3 + len(times)]]})
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enzyme_workbook",
                    default=ROOT / "data/wet_lab/enzyme_dilution_series_NADH_mM.xlsx")
    ap.add_argument("--glucose_workbook",
                    default=ROOT / "data/wet_lab/glucose_dilution_series_OD340_mM.xlsx")
    ap.add_argument("--out", default=ROOT / "data/nadh_dilution_series.yaml")
    a = ap.parse_args()
    data = {**enzyme_series(a.enzyme_workbook), **glucose_series(a.glucose_workbook)}
    for s in data.values():
        s["conditions"].sort(key=lambda c: c["dose_uM"])
    Path(a.out).write_text(yaml.safe_dump(data, sort_keys=False, width=120))
    print(f"wrote {a.out}: {len(data)} series, "
          f"{sum(len(s['conditions']) for s in data.values())} conditions")


if __name__ == "__main__":
    main()
