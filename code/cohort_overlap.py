#!/usr/bin/env python3
"""Patient-level cohort independence check (slide and patient identifiers)."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SICAP_ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")


def main():
    x = pd.read_excel(io.BytesIO(zipfile.ZipFile(SICAP_ZIP).read(
        "SICAPv2/wsi_labels.xlsx")))
    sicap_slides = {str(s).lower() for s in x.slide_id}
    sicap_patients = {str(p).lower() for p in x.patient_id}

    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    par_slides = {str(s).lower() for s in lab.slide_id}
    par_patients = {s.rstrip("abcdefghijklmnopqrstuvwxyz")
                    for s in par_slides}

    tc = pd.read_csv(ROOT / "data/refined_tcga_prad_matched.csv")
    tc_slides = {str(s).lower().replace(".svs", "") for s in tc.slide_name
                 if pd.notna(s)}
    tc_patients = {"-".join(s.split("-")[:3]) for s in tc_slides}

    out = {
        "sicap_slides": len(sicap_slides),
        "sicap_patients": len(sicap_patients),
        "par_slides": len(par_slides), "par_patients": len(par_patients),
        "tcga_slides": len(tc_slides), "tcga_patients": len(tc_patients),
        "slide_overlap_sicap_par": len(sicap_slides & par_slides),
        "slide_overlap_sicap_tcga": len(sicap_slides & tc_slides),
        "patient_overlap_sicap_par": len(sicap_patients & par_patients),
        "patient_overlap_sicap_tcga": len(sicap_patients & tc_patients),
        "patient_overlap_par_tcga": len(par_patients & tc_patients),
    }
    outdir = ROOT / "runs/cohort_overlap"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "overlap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
