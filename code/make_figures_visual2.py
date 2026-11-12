#!/usr/bin/env python3
"""Revised image-rich figures: pipeline, attention, TMA maps."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
import tiffslide
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures_visual import (FIG, ROOT, C_BLUE, C_ORANGE, C_GREEN,
                                 C_RED, C_GREY, sicap_patch, sicap_mask,
                                 wsi_crop)
from mgod_pilot import load_sicap, train


def fig_pipeline():
    from clinical_utility import site_data, tau_one_sided
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fig = plt.figure(figsize=(12.2, 4.8))
    ann = pd.read_csv(f"{DATA_ROOT}/sicapv2/annotations_all.csv")
    name = ann[ann.label == 2].image_name.iloc[0]
    img = sicap_patch("SICAPv2/images/" + name)
    msk = np.array(sicap_mask("SICAPv2/images/" + name))
    colors = np.zeros((*msk.shape, 4), dtype=np.float32)
    colors[msk == 3] = [0.30, 0.69, 0.41, 0.55]
    colors[msk == 4] = [0.86, 0.20, 0.20, 0.55]
    colors[msk == 5] = [0.45, 0.40, 0.75, 0.55]
    axi = fig.add_axes([0.03, 0.56, 0.16, 0.33])
    axi.imshow(img)
    axi.imshow(colors)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title("Training patch + mask", fontsize=8)
    fig.text(0.11, 0.50, "GP3 (green)  GP4 (red)  GP5 (purple)",
             ha="center", fontsize=6.5, color="#444444")

    fig.text(0.24, 0.86, "1. Train on SICAPv2 (anchored)",
             fontsize=9, fontweight="bold", color=C_BLUE)
    fig.text(0.24, 0.72,
             "18,783 tiles, 12,081 labelled; 155 slides with ISUP\n"
             "loss = tile CE + 0.5 * slide CE + auxiliary cancer BCE\n"
             "leave-one-cohort-out, patient-grouped",
             fontsize=7.2, va="top")
    fig.add_artist(FancyArrowPatch((0.335, 0.53), (0.335, 0.44),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=14,
                                   color="#555555", lw=1.4))
    fig.text(0.24, 0.38, "2. Deploy at a new site",
             fontsize=9, fontweight="bold", color=C_ORANGE)
    fig.text(0.24, 0.24,
             "K = 60 labelled slides -> refit probe -> temperature scaling\n"
             "one-sided conformal threshold at alpha = 0.10\n"
             "clear if score <= tau, else defer to a pathologist",
             fontsize=7.2, va="top")

    site = site_data("phikon")["PAR"]
    tab, feats, ymap = site
    keys = tab.slide.astype(str).str.lower().str.replace(
        "_leica", "", regex=False)
    X = np.stack([feats[k].mean(0) for k in keys])
    y = tab.y.to_numpy()
    rng = np.random.default_rng(0)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    cal = np.concatenate([rng.choice(pos, 30, replace=False),
                          rng.choice(neg, 30, replace=False)])
    sc = StandardScaler().fit(X[cal])
    lr = LogisticRegression(C=0.1, max_iter=1000,
                            class_weight="balanced").fit(
        sc.transform(X[cal]), y[cal])
    p = lr.predict_proba(sc.transform(X))[:, 1]
    tau = tau_one_sided(p[cal], y[cal], 0.10)
    hax = fig.add_axes([0.62, 0.18, 0.35, 0.62])
    bins = np.linspace(0, 1, 26)
    hax.hist(p[y == 0], bins=bins, color=C_GREEN, alpha=0.55,
             label="ISUP <= 1")
    hax.hist(p[y == 1], bins=bins, color=C_RED, alpha=0.55,
             label="ISUP >= 2")
    if tau is not None:
        hax.axvline(tau, color="#333333", lw=1.2, ls="--")
        clear = p <= tau
        cleared = float(clear.mean())
        fn = int(((y == 1) & clear).sum())
        fnc = fn / max(int(clear.sum()), 1)
        hax.axvspan(0, tau, color=C_GREEN, alpha=0.08)
        hax.axvspan(tau, 1, color=C_GREY, alpha=0.06)
        hax.text(0.52, 0.72,
                 f"clear (AS): {100*cleared:.0f}%\n"
                 f"FN among cleared: {100*fnc:.1f}%",
                 transform=hax.transAxes, fontsize=7, color="#2E5D3A",
                 fontweight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#CCCCCC", alpha=0.9))
        hax.text(0.78, 0.55, "defer", transform=hax.transAxes,
                 fontsize=7.5, color="#555555", fontweight="bold")
    hax.set_xlabel("Refit site score (P(ISUP >= 2))")
    hax.set_ylabel("Slides")
    hax.legend(frameon=False, fontsize=6.5, loc="upper center",
               bbox_to_anchor=(0.5, 1.14), ncol=2)
    hax.set_title("3. Score distribution on the target site", fontsize=8.5)
    hax.tick_params(labelsize=6.5)
    fig.suptitle("Pipeline: weak-label training and risk-controlled deployment",
                 fontsize=11.5, fontweight="bold", y=0.985)
    fig.savefig(FIG / "V3_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_attention():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb, labels, slides, gg, pat = load_sicap("phikon")
    keep = labels >= 0
    model = train("tile_plus_slide", emb[keep], labels[keep], slides[keep],
                  {s: gg[s] for s in gg}, {"epochs": 10, "batch": 256,
                                           "lam": 0.5}, seed=0)
    model.eval()

    par_lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    par_lab["slide"] = par_lab.slide_id.astype(str).str.lower()
    par_lab["maj"] = par_lab[["isup_1", "isup_r2", "isup_r3"]].median(axis=1)
    picks = []
    for sev in (0, 2, 5):
        cand = par_lab[(par_lab.maj == sev)]
        for s in cand.slide:
            hits = list((ROOT / "features/phikon").glob(f"{s}*.npz"))
            if hits and (ROOT / f"data/par/wsi/{s}_leica.svs").exists():
                picks.append((s, hits[0], sev))
                break
    fig, axes = plt.subplots(len(picks), 6, figsize=(12, 2.6 * len(picks)),
                             gridspec_kw={"width_ratios": [1.7, 1, 1, 1, 1, 1]})
    for row, (s, fid, sev) in enumerate(picks):
        z = np.load(fid)
        x = torch.from_numpy(z["tile_emb"].astype(np.float32)).to(dev)
        with torch.no_grad():
            h = model.trunk(x)
            a = torch.softmax(model.att(h).squeeze(-1), dim=0).cpu().numpy()
        coords = z["coords"]
        path = ROOT / f"data/par/wsi/{s}_leica.svs"
        order = np.argsort(-a)
        kept = []
        for idx in order:
            crop = wsi_crop(path, int(coords[idx, 0]), int(coords[idx, 1]),
                            out_px=224)
            if float(np.mean(np.asarray(crop))) < 225:
                kept.append((idx, crop))
            if len(kept) == 5:
                break
        sl = tiffslide.TiffSlide(str(path))
        w, hgt = sl.level_dimensions[0]
        thumb = sl.get_thumbnail((640, 640)).convert("RGB")
        sl.close()
        ax0 = axes[row, 0]
        ax0.imshow(thumb)
        sx = thumb.size[0] / w
        sy = thumb.size[1] / hgt
        ax0.scatter(coords[:, 0] * sx, coords[:, 1] * sy, s=3,
                    color="#BBBBBB", alpha=0.5)
        kx = [coords[idx, 0] * sx for idx, _ in kept]
        ky = [coords[idx, 1] * sy for idx, _ in kept]
        ax0.scatter(kx, ky, s=26, facecolors="none", edgecolors=C_RED, lw=1.4)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"ISUP {int(sev)}", fontsize=8, rotation=0, labelpad=18)
        for j, (idx, crop) in enumerate(kept):
            ax = axes[row, j + 1]
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"att {a[idx]:.3f}", fontsize=6.5, pad=2)
    fig.suptitle("What the model attends to: top-attended tissue tiles per slide",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "V4_attention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tma_heatmap():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    files = sorted((ROOT / "runs/gp4_dense/tiles/hoptimus0").glob("*.npz"))
    xs, ys = [], []
    for f in files:
        z = np.load(f)
        if "emb_s1" not in z.files:
            continue
        xs.append(z["emb_s1"])
        ys.append(z["frac"][:, 1:4].argmax(1))
    X = np.concatenate(xs)
    ycls = np.concatenate(ys)
    rng = np.random.default_rng(0)
    tr = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.03, max_iter=800,
                            class_weight="balanced").fit(
        sc.transform(X[tr]), ycls[tr])
    mixed = {}
    for f in files:
        z = np.load(f)
        if "emb_s1" not in z.files or len(z["frac"]) < 50:
            continue
        tum = z["frac"][:, 1:4].sum()
        if tum.sum() <= 0:
            continue
        frac4 = float(z["frac"][:, 2].sum() / tum.sum())
        mixed[str(z["core"])] = abs(frac4 - 0.5)
    core = min(mixed, key=mixed.get) if mixed else str(z["core"])
    z = np.load(ROOT / f"runs/gp4_dense/tiles/hoptimus0/{core}.npz")
    p = lr.predict_proba(sc.transform(z["emb_s1"]))
    cls = list(lr.classes_)
    p4 = p[:, cls.index(1)] if 1 in cls else np.zeros(len(p))
    pairs = pd.read_csv(ROOT / "data/tma_pairs.csv")
    row = pairs[pairs.key == core].iloc[0]
    core_img = Image.open(row["image"]).convert("RGB")
    mask = np.array(Image.open(row["mask"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    axes[0].imshow(core_img)
    axes[0].set_title("TMA core", fontsize=8.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[1]
    ax.imshow(core_img, alpha=0.60)
    for xi, yi, pi in zip(z["x"], z["y"], p4):
        ax.add_patch(mpatches.Rectangle((xi, yi), 224, 224,
                                        facecolor=plt.cm.inferno(pi),
                                        edgecolor="none", alpha=0.55))
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Model P(GP4)", fontsize=7)
    ax.set_title("Model prediction per tile", fontsize=8.5)
    ax = axes[2]
    ax.imshow(core_img)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 2] = [0.20, 0.55, 0.90, 0.35]
    ax.imshow(overlay)
    ax.set_title("Expert GP4 mask (blue)", fontsize=8.5)
    fig.suptitle(f"Dense model map vs expert annotation (core {core.upper()})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "V5_tma_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_pipeline()
    fig_attention()
    fig_tma_heatmap()
    print("revised figures written to", FIG)


if __name__ == "__main__":
    main()
