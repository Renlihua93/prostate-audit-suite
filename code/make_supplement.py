#!/usr/bin/env python3
"""Supplementary tables and figures for the audit manuscript."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "figures"
OUT = PROJECT_ROOT / "SUPPLEMENT_v1.md"
FIG.mkdir(parents=True, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY, C_PURPLE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B2")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def tables():
    md = ["# Supplementary Information\n"]
    # S2 LOCO
    rows = []
    for enc in ("phikon", "hoptimus0"):
        d = json.load(open(ROOT / f"runs/mgod_loco_{enc}/results.json"))
        for mode, runs in d.items():
            row = {"encoder": enc, "mode": mode}
            for k in ("sicap", "par", "tcga"):
                vals = [r[k] for r in runs if k in r]
                if vals:
                    row[k] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            rows.append(row)
    md += ["## Table S2. Leave-one-cohort-out matrix\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    # S3 PPBench
    rows = []
    for enc in ("phikon", "hoptimus0"):
        d = pd.read_csv(ROOT / f"runs/mgod_ppbench_{enc}/results.csv")
        g = d.groupby(["condition", "mode"]).auc_g4plus.agg(["mean", "std"])
        for cond, r in g.groupby(level=0):
            rows.append({"encoder": enc, "condition": cond,
                         "tile_only": f"{r.loc[(cond,'tile_only'),'mean']:.3f}"
                         f" ± {r.loc[(cond,'tile_only'),'std']:.3f}",
                         "+slide": f"{r.loc[(cond,'tile_plus_slide'),'mean']:.3f}"
                         f" ± {r.loc[(cond,'tile_plus_slide'),'std']:.3f}"})
    md += ["## Table S3. PANDA-PLUS-Bench by condition (AUC G4+)\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    # S4 %GP4
    cg = json.load(open(ROOT / "runs/cross_gp4/results.json"))
    rows = []
    for k, v in cg.items():
        rows.append({"direction": k, "n": v.get("n"),
                     "MAE": f"{v.get('mae', np.nan):.1f}",
                     "Spearman": f"{v.get('spearman', np.nan):.2f}",
                     "AUC>=10%": f"{v.get('auc_ge10', np.nan):.3f}",
                     "band MAE": f"{v.get('mae_band20', np.nan):.1f}"})
    md += ["## Table S4. %GP4 transfer\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    # S5 per seed
    rows = []
    for enc in ("phikon", "hoptimus0"):
        d = json.load(open(ROOT / f"runs/mgod_pilot_{enc}/results.json"))
        for mode, runs in d["runs"].items():
            for i, r in enumerate(runs):
                rows.append({"encoder": enc, "mode": mode, "seed": i,
                             "PAR": f"{r.get('par_auc_ge2', np.nan):.3f}",
                             "TCGA": f"{r.get('tcga_auc_ge2', np.nan):.3f}",
                             "in-domain": f"{r.get('auc_ge2', np.nan):.3f}"})
    md += ["## Table S5. Per-seed transfer results\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    # S6 DOC grid + subgroups
    rows = []
    for enc in ("phikon", "hoptimus0"):
        d = pd.read_csv(ROOT / f"runs/clinical_utility_{enc}/simulation.csv")
        if enc == "hoptimus0":
            s012 = ROOT / "runs/clinical_utility_hoptimus0/simulation_s012.csv"
            if s012.exists():
                d = pd.concat([pd.read_csv(s012), d], ignore_index=True)
        ok = d[d.feasible]
        g = ok.groupby(["alpha", "K"]).agg(
            cleared=("cleared", "mean"), missed=("missed_per100", "mean"),
            sens=("sens_detected", "mean"), n=("rep", "count"))
        for (a, k), r in g.iterrows():
            rows.append({"encoder": enc, "alpha": a, "K": k,
                         "cleared": f"{r.cleared:.3f}",
                         "missed/100": f"{r.missed:.2f}",
                         "sensitivity": f"{r.sens:.3f}",
                         "n": int(r.n)})
    md += ["## Table S6. Deployment grid\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    sub = pd.read_csv(ROOT / "runs/sensitivity_phikon/subgroups_PAR.csv")
    g = sub.groupby("group").cleared.agg(["mean", "count"])
    rows = []
    for grp, r in g.iterrows():
        k = int(round(r["mean"] * r["count"]))
        lo, hi = wilson(k, int(r["count"]))
        rows.append({"group": grp, "cleared": f"{r['mean']:.3f}",
                     "95% CI": f"{lo:.3f}–{hi:.3f}", "n": int(r["count"])})
    md += ["### Table S6b. Clearing by subgroup (Phikon-v1, alpha = 0.10)\n",
           pd.DataFrame(rows).to_string(index=False), ""]
    return "\n".join(md)


def fig_task_dependence():
    rows = [("Detection\n(PESO region)", 0.988, 0.999),
            ("Grading\n(PAR slide)", 0.896, 0.931),
            ("Grading\n(TCGA slide)", 0.821, 0.761),
            ("%GP4 ≥10%\n(cross-cohort)", 0.627, 0.453)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color=C_BLUE,
           label="Phikon-v1")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color=C_ORANGE,
           label="H-optimus-0")
    for xi, (a, b) in enumerate([(r[1], r[2]) for r in rows]):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color=C_GREY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Task dependence of cross-cohort transfer", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "S_task_dependence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_scanner():
    df = pd.read_csv(ROOT / "runs/scanner_shift_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.scatter(df.leica, df.grundium, s=18, color=C_BLUE, alpha=0.8)
    lim = [min(df.leica.min(), df.grundium.min()),
           max(df.leica.max(), df.grundium.max())]
    ax.plot(lim, lim, ls="--", color=C_GREY, lw=0.8)
    ax.set_xlabel("Leica score")
    ax.set_ylabel("Grundium score")
    ax.set_title("Paired rescan scores (r = 0.91)", fontsize=8.5)
    ax.grid(alpha=0.25)
    # embedding cosine distributions
    feats_l, feats_g = {}, {}
    for f in (ROOT / "features/phikon").glob("*.npz"):
        feats_l[f.stem.lower().replace("_leica", "")] = \
            np.load(f)["tile_emb"].mean(0)
    for f in (ROOT / "features/phikon_grundium").glob("*.npz"):
        feats_g[f.stem.lower().replace("_grundium", "")] = \
            np.load(f)["tile_emb"].mean(0)
    common = sorted(set(feats_l) & set(feats_g))
    L = np.stack([feats_l[s] for s in common])
    G = np.stack([feats_g[s] for s in common])
    same = (L * G).sum(1) / (np.linalg.norm(L, axis=1)
                             * np.linalg.norm(G, axis=1))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(common))
    while np.any(perm == np.arange(len(common))):
        perm = rng.permutation(len(common))
    diff = (L * G[perm]).sum(1) / (np.linalg.norm(L, axis=1)
                                   * np.linalg.norm(G[perm], axis=1))
    ax = axes[1]
    ax.hist(same, bins=12, alpha=0.65, color=C_BLUE,
            label=f"same slide ({same.mean():.2f})")
    ax.hist(diff, bins=12, alpha=0.65, color=C_ORANGE,
            label=f"different slide ({diff.mean():.2f})")
    ax.set_xlabel("Mean-embedding cosine")
    ax.set_ylabel("Slides")
    ax.set_title("Scanner drift in embeddings", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "S_scanner_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_calibration():
    c = json.load(open(ROOT / "runs/sensitivity_phikon/calibration_PAR.json"))
    rel = pd.DataFrame(c["reliability"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=0.8)
    ax.plot(rel.pred, rel.obs, "o-", color=C_BLUE, lw=1.2, ms=4,
            label="Raw (slope 0.63)")
    ax.set_xlabel("Predicted P(ISUP ≥ 2)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration reliability (Phikon-v1, K = 60)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "S_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_dca():
    d = pd.read_csv(ROOT / "runs/sensitivity_phikon/dca_PAR.csv")
    g = d.groupby("pt")[["ai_only", "read_all", "treat_all"]].mean()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(g.index, g.ai_only, "o-", color=C_BLUE, lw=1.3,
            label="AI-only")
    ax.plot(g.index, g.read_all, ls="--", color=C_GREEN, lw=1.2,
            label="Read all")
    ax.plot(g.index, g.treat_all, ls=":", color=C_ORANGE, lw=1.2,
            label="Treat all")
    ax.axhline(0, color=C_GREY, lw=0.7)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve (PAR, Phikon-v1)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "S_dca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_site_size():
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for a, color in ((0.05, C_BLUE), (0.10, C_ORANGE)):
        f = ROOT / f"runs/sensitivity_phikon/site_size_PAR_a{a}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        g = d.groupby("n_site").feasible.agg(["mean", "count"])
        lo, hi = [], []
        for n, r in g.iterrows():
            k = int(round(r["mean"] * r["count"]))
            a1, a2 = wilson(k, int(r["count"]))
            lo.append(a1); hi.append(a2)
        err_lo = np.clip(np.array(g["mean"]) - np.array(lo), 0, None)
        err_hi = np.clip(np.array(hi) - np.array(g["mean"]), 0, None)
        ax.errorbar(g.index, g["mean"], yerr=[err_lo, err_hi],
                    fmt="o-", color=color, capsize=3, lw=1.2,
                    label=f"alpha = {a}")
    ax.set_xlabel("Site size (slides)")
    ax.set_ylabel("Certification probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title("Site-size feasibility (95% Wilson CI)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "S_site_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    md = tables()
    md += """

## Supplementary figure legends

**Supplementary Fig. 7.** Task dependence of cross-cohort transfer. Detection on PESO regions transfers (0.988 and 0.999 AUC), grading degrades (PAR 0.896/0.931; TCGA-PRAD 0.821/0.761) and %GP4 does not transfer (0.627/0.453 across directions).

**Supplementary Fig. 8.** Scanner shift in 30 paired PAR slides. Left: Leica versus Grundium scores (r = 0.91). Right: mean-embedding cosine similarity for same-slide cross-scanner pairs versus different-slide pairs.

**Supplementary Fig. 9.** Calibration reliability of the refit site probe before temperature scaling (slope 0.63; after scaling, 1.06).

**Supplementary Fig. 10.** Decision curve on PAR for the AI-only rule versus read-all and treat-all strategies.

**Supplementary Fig. 11.** Site-size feasibility with Wilson 95% confidence intervals for alpha = 0.05 and 0.10.
"""
    OUT.write_text(md, encoding="utf-8")
    fig_task_dependence()
    fig_scanner()
    fig_calibration()
    fig_dca()
    fig_site_size()
    print("supplement written", OUT)


if __name__ == "__main__":
    main()
