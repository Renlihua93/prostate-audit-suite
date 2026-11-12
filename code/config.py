"""Path configuration for the prostate-audit code release.

Set PROSTATE_DATA_ROOT and PROSTATE_MODEL_ROOT to the locations of the public
cohorts and model weights. Derived features are expected under
PROJECT_ROOT/features (see docs/data_layout.md).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PROSTATE_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("PROSTATE_MODEL_ROOT", PROJECT_ROOT / "models"))
