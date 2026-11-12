# prostate-audit-suite

Code and analysis scripts for the manuscript:

> **Silent failures in cross-site prostate grading AI: a five-pitfall audit and a reproducible evaluation suite**

This repository contains the six-check audit suite and the scripts used to reproduce the main tables and figures. No patient images, raw whole-slide images, model weights, or derived feature files are redistributed. All cohorts are public; see the manuscript Data availability statement.

## Repository structure

```
code/                  analysis and figure scripts
docs/data_layout.md    expected data and feature layout
requirements.txt       pinned Python dependencies
LICENSE                MIT
```

## Installation

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required; the manuscript analyses were run with Python 3.12, PyTorch 2.5.1, scikit-learn 1.9.1 and timm 1.0.30.

## Data layout

Set these environment variables before running:

```
export PROSTATE_DATA_ROOT=/path/to/public/cohorts
export PROSTATE_MODEL_ROOT=/path/to/pretrained/weights
```

Expected subdirectories and file naming are documented in `docs/data_layout.md`. Derived features are read from `PROJECT_ROOT/features`; the repository does not include them.

## Reproducing the main results

| Script | Output |
|---|---|
| `code/audit_suite.py` | Six-check PASS/FAIL report (Table S4, Table S12) |
| `code/pitfall_matrix.py` | Five pitfall effect sizes (Fig. 2, Table S5) |
| `code/clinical_utility.py` | Site-calibration operating points (Table 1, Tables S6/S8) |
| `code/active_acquisition.py`, `code/audit_active.py` | Calibration-set selection bias (Fig. 3) |
| `code/cross_gp4_eval.py`, `code/cross_gp4_thresholds.py` | Cross-cohort %GP4 transfer (Table S7) |
| `code/scanner_shift.py` | Paired-scanner analysis (Fig. S3) |
| `code/mgod_pilot.py`, `code/mgod_loco.py` | Transfer and leave-one-cohort-out results (Table S5) |
| `code/make_figures_E.py`, `code/make_figures_visual.py` | Main figures |
| `code/make_supplement.py` | Supplementary tables and figures |

Scripts assume cached features and the public cohorts; feature extraction is compute-heavy and is not required for the audit suite itself. The suite runs in under two minutes on cached features, excluding feature extraction and probe refitting.

## What is not included

- Raw WSI or patient images
- Patient identifiers or clinical data
- Pretrained encoder weights
- Derived feature archives (available from the authors under the terms in the manuscript)

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the manuscript. A Zenodo DOI will be added after the first archived release:

```
Renlihua93. prostate-audit-suite (2026). GitHub repository.
https://github.com/Renlihua93/prostate-audit-suite
```
