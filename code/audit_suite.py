#!/usr/bin/env python3
"""Audit suite: six checks for cross-cohort pathology AI claims.

Runs on the existing six-cohort assets and writes a PASS/FAIL report:
  1. hyperparameter sensitivity (source-tuned vs fixed conservative)
  2. baseline completeness (tile-only vs standard MIL)
  3. calibration-set independence (reports the measured selection-bias gap)
  4. alignment stability (sign consistency across two target cohorts)
  5. label tie-rule sensitivity (multi-reader slides crossing ISUP>=2)
  6. cohort independence (shared slides/patients between source and target)

PASS means the claim survives the check; FAIL means a pitfall is detected.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitfall_matrix import (alignment_effect, hyperparameter_effect,
                            strawman_effect, tie_effect)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/audit_suite"
OUT.mkdir(parents=True, exist_ok=True)
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def cohort_independence():
    sic = np.load(ROOT / "features/sicap_c224/patch_features.npz",
                  allow_pickle=True)
    sic_slides = {NORM(s.split("/")[-1]) for s in sic["slide_ids"]}
    par = {NORM(p.stem) for p in
           (ROOT / "features/phikon").glob("*.npz")}
    tcga = {NORM(p.stem) for p in
            (ROOT / "features/tcga_phikon").glob("*.npz")}
    par_g = {NORM(p.stem) for p in
             (ROOT / "features/phikon_grundium").glob("*.npz")}
    return {"sicap_n": len(sic_slides), "par_n": len(par),
            "tcga_n": len(tcga),
            "sicap_par_overlap": len(sic_slides & par),
            "sicap_tcga_overlap": len(sic_slides & tcga),
            "par_grundium_same_slide_pairs": len(par & par_g)}


def calibration_bias():
    naive = pd.read_csv(ROOT / "runs/active_acq_phikon/simulation.csv")
    naive = naive[(naive.strategy == "uncertainty") & (naive.alpha == 0.10)
                  & (naive.K == 80)]
    audit = pd.read_csv(ROOT / "runs/audit_active_phikon/simulation.csv")
    audit = audit[(audit.arm == "uncertainty_fit") & (audit.K == 80)]
    return {"naive_certification": float(naive.feasible.mean()),
            "valid_certification": float(audit.feasible.mean())}


def main():
    checks = []
    hp = hyperparameter_effect()
    checks.append({"check": "Hyperparameter sensitivity",
                   "metric": f"ΔAUC TCGA {hp['delta_TCGA']:+.3f}",
                   "status": "FAIL" if abs(hp["delta_TCGA"]) > 0.02 else "PASS",
                   "detail": hp})

    sm = strawman_effect()
    checks.append({"check": "Baseline completeness",
                   "metric": f"gain vs tile-only {sm['tcga_weak_gain_phikon']:+.3f}"
                             f" / {sm['tcga_weak_gain_hoptimus0']:+.3f}",
                   "status": "FAIL",
                   "detail": sm})

    cb = calibration_bias()
    checks.append({"check": "Calibration-set independence",
                   "metric": f"certification {100*cb['naive_certification']:.1f}%"
                             f" → {100*cb['valid_certification']:.1f}%",
                   "status": "FAIL", "detail": cb})

    al = alignment_effect()
    signs = np.sign([al["tcga_delta"], al["par_delta"]])
    checks.append({"check": "Alignment stability",
                   "metric": f"TCGA {al['tcga_delta']:+.3f}, PAR {al['par_delta']:+.3f}",
                   "status": "FAIL" if len(set(signs)) > 1 else "PASS",
                   "detail": al})

    ti = tie_effect()
    checks.append({"check": "Tie-rule sensitivity",
                   "metric": f"{ti['ties_crossing_ge2']}/{ti['n_multireader']}"
                             f" labels flip",
                   "status": "FAIL" if ti["ties_crossing_ge2"] > 0 else "PASS",
                   "detail": ti})

    ci = cohort_independence()
    checks.append({"check": "Cohort independence",
                   "metric": f"shared slides SICAP-PAR {ci['sicap_par_overlap']},"
                             f" SICAP-TCGA {ci['sicap_tcga_overlap']}",
                   "status": "PASS" if ci["sicap_par_overlap"] == 0
                             and ci["sicap_tcga_overlap"] == 0 else "FAIL",
                   "detail": ci})

    lines = ["# Audit report", ""]
    for c in checks:
        lines.append(f"- **{c['status']}** — {c['check']}: {c['metric']}")
    report = "\n".join(lines)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(checks, indent=2,
                                                default=float))
    print(report)


if __name__ == "__main__":
    main()
