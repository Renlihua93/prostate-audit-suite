from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402
#!/usr/bin/env python3
"""Three-encoder comparison of the audit effects with bootstrap CIs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "figures"
C = ["#4C72B0", "#DD8452", "#55A868"]

df = pd.read_csv(ROOT / "runs/audit_ci/ci.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
for ax, analysis, title in ((axes[0], "hyperparameter",
                             "(a) Hyperparameter sensitivity"),
                            (axes[1], "alignment",
                             "(b) Alignment (centering)")):
    d = df[df.analysis == analysis]
    targets = ["PAR", "TCGA"]
    encoders = ["phikon", "hoptimus0", "resnet50"]
    x = np.arange(len(targets))
    width = 0.26
    for i, enc in enumerate(encoders):
        vals, lo, hi = [], [], []
        for t in targets:
            r = d[(d.encoder == enc) & (d.target == t)]
            if len(r):
                vals.append(float(r.delta.iloc[0]))
                lo.append(float(r.delta.iloc[0] - r.lo.iloc[0]))
                hi.append(float(r.hi.iloc[0] - r.delta.iloc[0]))
            else:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        ax.bar(x + (i - 1) * width, vals, width, color=C[i],
               label={"phikon": "Phikon-v1", "hoptimus0": "H-optimus-0",
                      "resnet50": "ResNet50"}[enc])
        ax.errorbar(x + (i - 1) * width, vals, yerr=[lo, hi], fmt="none",
                    ecolor="#333333", elinewidth=1, capsize=2)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)
axes[0].set_ylabel("Delta AUC (95% CI)")
fig.suptitle("Audit effects across three frozen encoders", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "S_encoder_effects.png", dpi=300, bbox_inches="tight")
print("encoder figure written")
