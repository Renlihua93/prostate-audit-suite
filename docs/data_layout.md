# Data layout

The code expects the study cohorts under `PROSTATE_DATA_ROOT` and encoder weights under `PROSTATE_MODEL_ROOT`. Derived features are read from `PROJECT_ROOT/features`; they are not included in this repository.

## Cohorts

| Cohort | Expected location | Notes |
|---|---|---|
| SICAPv2 | `$PROSTATE_DATA_ROOT/sicapv2/SICAPv2.zip` | development anchor; patch labels |
| cohort A | `$PROSTATE_DATA_ROOT/cohort_A/` | biopsy WSI; 2–3 reader ISUP labels |
| TCGA-PRAD | `$PROSTATE_DATA_ROOT/tcga_prad/` | prostatectomy WSI; refined grade group |
| PANDA-PLUS-Bench | `$PROSTATE_DATA_ROOT/panda_plus_bench/` | perturbation stress test |
| cohort B | `$PROSTATE_DATA_ROOT/cohort_B/` | detection regions |
| Arvaniti TMA | `$PROSTATE_DATA_ROOT/arvaniti_images_extracted/` | TMA cores; pixel GP3/4/5 |

## Derived features

Feature archives are `.npz` files with two arrays:

- `tile_emb` — `[n_tiles, d]` float32 tile embeddings
- `coords` — `[n_tiles, 2]` tile coordinates in the source WSI

Directory names follow the encoder and cohort, for example:

```
features/phikon/                 # cohort A, Phikon-v1
features/tcga_phikon/            # TCGA-PRAD, Phikon-v1
features/hoptimus0_par/          # cohort A, H-optimus-0
features/hoptimus0_tcga/         # TCGA-PRAD, H-optimus-0
features/sicap_phikon_c224/      # SICAPv2 patch features, Phikon-v1
features/sicap_c224/             # SICAPv2 patch features, H-optimus-0 / ResNet50
```

## Running the audit suite

```
python code/audit_suite.py
```

The suite writes `runs/audit_suite/report.md` and `report.json`. It requires cached features and the cohort label tables described above; it does not extract features or refit the encoder.
