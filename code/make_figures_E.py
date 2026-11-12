#!/usr/bin/env python3
"""Publication figures for the evaluation-audit story (English labels)."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_GREY = "#8C8C8C"
C_PURPLE = "#8172B2"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def fig1_overview():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, title, lines, color, alpha=0.10):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=color, edgecolor=color, alpha=0.10, lw=1.2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color=color)
        for i, ln in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.62 - i * 0.30, ln, ha="left",
                    va="top", fontsize=7.6, color="#222222")

    box(0.2, 1.0, 3.3, 2.6, "Reported pipeline",
        ["Cross-site gains of +12–25% AUC (weak labels)",
         "53% of AS candidates auto-cleared",
         "4% false negatives among cleared",
         "Safe-onboarding story ready to submit"],
        C_BLUE)
    box(4.2, 0.55, 3.5, 3.1, "Five standard practices audited",
        ["1. Hyperparameters tuned on the source cohort",
         "2. Tile-only straw-man baseline",
         "3. Conformal calibration on selected labels",
         "4. Unsupervised cohort centering",
         "5. Median tie handling in multi-reader labels"],
        C_ORANGE)
    box(8.4, 1.0, 3.3, 2.6, "After audit",
        ["AUC 0.47 or 0.64 depending on hyperparameters",
         "Weak-label gain ≈0 vs standard MIL",
         "Certification 94.7% → 1.0% (valid audit)",
         "Centering +0.15 (TCGA) but −0.09 (PAR)",
         "13/338 multi-reader labels flip at ISUP≥2"],
        C_RED)

    arr1 = FancyArrowPatch((3.55, 2.3), (4.15, 2.3), arrowstyle="-|>",
                           mutation_scale=14, color="#555555", lw=1.4)
    arr2 = FancyArrowPatch((7.75, 2.3), (8.35, 2.3), arrowstyle="-|>",
                           mutation_scale=14, color="#555555", lw=1.4)
    ax.add_patch(arr1)
    ax.add_patch(arr2)
    ax.set_title("Silent evaluation failures in cross-cohort prostate grading AI",
                 fontsize=12, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(FIG / "E1_audit_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_effects():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    # (a) hyperparameter
    ax = axes[0, 0]
    x = np.arange(2)
    tuned = [0.795, 0.473]
    fixed = [0.847, 0.641]
    ax.bar(x - 0.18, tuned, 0.36, label="Source-tuned C", color=C_GREY)
    ax.bar(x + 0.18, fixed, 0.36, label="Fixed C=0.001 (raw)", color=C_BLUE)
    for xi, (a, b) in enumerate(zip(tuned, fixed)):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
        ax.annotate(f"+{b-a:.2f}", (xi, max(a, b) + 0.07), ha="center",
                    fontsize=8, color=C_RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["PAR", "TCGA"])
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("AUC (ISUP/GG ≥ 2)")
    ax.set_title("(a) Source-tuned hyperparameters", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (b) straw-man baseline
    ax = axes[0, 1]
    x = np.arange(2)
    tile = [0.733, 0.610]
    weak = [0.821, 0.761]
    ax.bar(x - 0.18, tile, 0.36, label="Tile-only baseline", color=C_GREY)
    ax.bar(x + 0.18, weak, 0.36, label="+ slide-level weak labels",
           color=C_GREEN)
    for xi, (a, b) in enumerate(zip(tile, weak)):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
        ax.annotate(f"+{b-a:.2f}", (xi, max(a, b) + 0.07), ha="center",
                    fontsize=8, color=C_RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Phikon-v1", "H-optimus-0"])
    ax.set_ylim(0, 1.30)
    ax.set_title("(b) Straw-man baseline inflates gains", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (c) selection bias
    ax = axes[0, 2]
    bars = ax.bar(["Naive\n(selected labels)", "Valid\n(random audit)"],
                  [100.0, 0.0], color=[C_ORANGE, C_BLUE], width=0.55)
    for b, v in zip(bars, [100.0, 0.0]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Apparent certification rate (%)")
    ax.set_title("(c) Calibration-set selection bias", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)

    # (d) alignment sign flip
    ax = axes[1, 0]
    x = np.arange(2)
    zero = [0.485, 0.792]
    cent = [0.636, 0.702]
    ax.bar(x - 0.18, zero, 0.36, label="Zero-shot", color=C_GREY)
    ax.bar(x + 0.18, cent, 0.36, label="Cohort centering", color=C_PURPLE)
    for xi, (a, b) in enumerate(zip(zero, cent)):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
        ax.annotate(f"{b-a:+.2f}", (xi, max(a, b) + 0.07), ha="center",
                    fontsize=8, color=C_RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["TCGA", "PAR"])
    ax.set_ylim(0, 1.30)
    ax.set_title("(d) Unsupervised alignment: sign flip", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (e) ties
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.86, "(e) Multi-reader label ties", ha="center",
            fontsize=9.5, fontweight="bold")
    ax.text(0.5, 0.60, "338 slides with ≥2 readers", ha="center",
            fontsize=8.5)
    ax.text(0.5, 0.40, "51 medians at x.5", ha="center", fontsize=8.5)
    ax.text(0.5, 0.20, "13 cross the ISUP≥2 boundary", ha="center",
            fontsize=8.5, color=C_RED, fontweight="bold")
    ax.text(0.5, 0.02, "3.8% of slides change binary label\nwith the tie rule",
            ha="center", fontsize=7.5, color="#555555")

    # (f) clinical translation
    ax = axes[1, 2]
    miss = [21, 58]
    bars = ax.bar(["4% FN\namong cleared", "11% FN\namong cleared"],
                  miss, color=[C_GREEN, C_RED], width=0.55)
    for b, v in zip(bars, miss):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.set_ylabel("High-risk cases missed per 1,000")
    ax.set_title("(f) Clinical translation of the audit", fontsize=9.5)
    ax.annotate("+37 missed\nper 1,000", xy=(0.5, 62), ha="center",
                fontsize=8.5, color=C_RED, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Reversal table: each pitfall changes the conclusion",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "E2_reversal_effects.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig3_selection_bias():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    d = pd.read_csv(ROOT / "runs/active_acq_phikon/simulation.csv")
    d = d[(d.strategy == "uncertainty") & (d.alpha == 0.10)]
    g = d.groupby("K").feasible.mean()
    ax.plot(g.index, 100 * g.values, "o-", color=C_ORANGE, lw=1.8,
            label="Naive (calibrate on selected labels)")
    a = pd.read_csv(ROOT / "runs/audit_active_phikon/simulation.csv")
    a = a[(a.arm == "uncertainty_fit") & (a.alpha == 0.10)]
    ga = a.groupby("K").feasible.mean()
    ax.plot(ga.index, 100 * ga.values, "s--", color=C_BLUE, lw=1.8,
            label="Valid (random audit calibration)")
    for x, y in zip(g.index, 100 * g.values):
        ax.text(x, y + 2.5, f"{y:.0f}%", ha="center", fontsize=7,
                color=C_ORANGE)
    for x, y in zip(ga.index, 100 * ga.values):
        ax.text(x, y + 2.5, f"{y:.0f}%", ha="center", fontsize=7,
                color=C_BLUE)
    ax.set_xlabel("Labelled site slides (K)")
    ax.set_ylabel("Apparent certification rate (%)")
    ax.set_ylim(-5, 110)
    ax.set_title("Decision-targeted labelling: manufactured certification",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "E3_selection_bias.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig4_prototypes():
    rows = [
        ("Weak-supervision bridge", "label budget unchanged (K=60)", "fail"),
        ("Cascade v1 (det+grade)", "α=0.05 never certifiable", "fail"),
        ("Cascade v2 (benign+grade)", "more clearing, more misses", "fail"),
        ("Decision-targeted labels", "selection bias; valid design 1%", "fail"),
        ("Reader-distribution MIL", "entropy ≈ majority model", "fail"),
        ("Auxiliary detection head", "ΔAUC ≤ 0.003", "fail"),
        ("QWK threshold calibration", "Phikon +0.006–0.023", "fail"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1.5)
    ax.text(0.02, len(rows) + 0.9, "Seven method prototypes audited",
            fontsize=11, fontweight="bold")
    ax.text(0.55, len(rows) + 0.9, "Outcome", fontsize=9, color="#555555")
    for i, (name, outcome, status) in enumerate(rows[::-1]):
        y = i + 0.4
        color = C_RED if status == "fail" else C_GREEN
        ax.plot(0.03, y + 0.15, marker="o", markersize=5, color=color)
        ax.text(0.06, y + 0.08, name, fontsize=8.5)
        ax.text(0.55, y + 0.08, outcome, fontsize=8, color="#333333")
    fig.tight_layout()
    fig.savefig(FIG / "E4_prototypes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_suite():
    rep = json.load(open(ROOT / "runs/audit_suite/report.json"))
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rep) + 2.0)
    ax.text(0.02, len(rep) + 0.75, "Audit suite: six checks on a cross-site claim",
            fontsize=11.5, fontweight="bold")
    for i, c in enumerate(rep[::-1]):
        y = i + 0.60
        good = c["status"] == "PASS"
        color = C_GREEN if good else C_RED
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.12), 0.10, 0.34,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color, edgecolor=color, alpha=0.14, lw=1.0))
        ax.text(0.07, y + 0.05, c["status"], ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold")
        ax.text(0.15, y + 0.05, c["check"], va="center", fontsize=8.5)
        ax.text(0.62, y + 0.05, c["metric"], va="center", fontsize=7.6,
                color="#333333")
    ax.text(0.02, 0.15, "Five of six checks change the conclusion; "
            "runtime < 2 min on cached features",
            fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "E5_audit_suite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_overview()
    fig2_effects()
    fig3_selection_bias()
    fig4_prototypes()
    fig5_suite()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
