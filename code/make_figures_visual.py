#!/usr/bin/env python3
"""Image-rich figures: cohorts, architecture, pipeline, attention, TMA maps."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import io
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tiffslide

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_phikon_features import choose_level, thumbnail, tissue_mask
from mgod_pilot import load_sicap, train

ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SICAP_ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")

C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_GREY = "#8C8C8C"
C_PURPLE = "#8172B2"

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------- image utils
def sicap_patch(name: str) -> Image.Image:
    zf = zipfile.ZipFile(SICAP_ZIP)
    return Image.open(io.BytesIO(zf.read(name))).convert("RGB")


def sicap_mask(name: str) -> Image.Image:
    zf = zipfile.ZipFile(SICAP_ZIP)
    m = name.replace(".jpg", ".png").replace("/images/", "/masks/")
    return Image.open(io.BytesIO(zf.read(m))).convert("L")


def wsi_tissue_center(path: Path):
    from scipy.ndimage import uniform_filter
    s = tiffslide.TiffSlide(str(path))
    th, _, (sx, sy) = thumbnail(s)
    mask = tissue_mask(th).astype(np.float32)
    s.close()
    if mask.sum() == 0:
        return None
    dens = uniform_filter(mask, size=25)
    iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
    return int(ix / sx), int(iy / sy)


def wsi_crop(path: Path, cx: int, cy: int, target_mpp=0.5, tile=224,
             out_px=512):
    s = tiffslide.TiffSlide(str(path))
    mpp = None
    for key in ("openslide.mpp-x", "tiffslide.mpp-x"):
        try:
            mpp = float(s.properties.get(key))
            break
        except (TypeError, ValueError):
            continue
    w, h = s.level_dimensions[0]
    level, actual = choose_level(s, target_mpp, mpp or target_mpp)
    size = max(8, int(round(tile * target_mpp / actual)))
    half = size // 2
    x0 = int(np.clip(cx - half, 0, w - size))
    y0 = int(np.clip(cy - half, 0, h - size))
    im = s.read_region((x0, y0), level, (size, size)).convert("RGB")
    s.close()
    return im.resize((out_px, out_px))


def wsi_field(path: Path, cx: int, cy: int, size_level0=2048, out_px=512):
    s = tiffslide.TiffSlide(str(path))
    w, h = s.level_dimensions[0]
    half = size_level0 // 2
    x0 = int(np.clip(cx - half, 0, w - size_level0))
    y0 = int(np.clip(cy - half, 0, h - size_level0))
    im = s.read_region((x0, y0), 0, (size_level0, size_level0)).convert("RGB")
    s.close()
    return im.resize((out_px, out_px), Image.LANCZOS)


def wsi_thumbnail(path: Path, size=400):
    s = tiffslide.TiffSlide(str(path))
    th = s.get_thumbnail((size, size)).convert("RGB")
    s.close()
    return th


def find_tcga(slide: str) -> Path | None:
    hits = list(Path(f"{PROJECT_ROOT}/data/tcga_wsi").glob(
        "*.svs"))
    for p in hits:
        if p.stem.lower() == slide.lower():
            return p
    return None


# ------------------------------------------------------------------ figure V3
def fig_cohorts():
    ann = pd.read_csv(f"{DATA_ROOT}/sicapv2/annotations_all.csv")
    g4 = ann[ann.label == 2].image_name.iloc[0]
    sicap_img = sicap_patch("SICAPv2/images/" + g4)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    par_slide = Path(f"{PROJECT_ROOT}/data/par/wsi/c004_leica.svs")
    par_npz = sorted((ROOT / "features/phikon").glob("c004*.npz"))[0]
    pz = np.load(par_npz)
    par_center = (int(pz["coords"][len(pz["coords"]) // 2, 0]),
                  int(pz["coords"][len(pz["coords"]) // 2, 1]))
    par_img = wsi_field(par_slide, *par_center)
    tc = find_tcga("TCGA-2A-A8VL-01Z-00-DX1.2C2BD6EF-EC17-4117-AE89-A22B67AFB233")
    tz = np.load(ROOT / f"features/tcga_phikon/{tc.stem.lower()}.npz")
    tc_center = (int(tz["coords"][len(tz["coords"]) // 2, 0]),
                 int(tz["coords"][len(tz["coords"]) // 2, 1]))
    tc_img = wsi_field(tc, *tc_center)
    d = pd.read_parquet(
        f"{DATA_ROOT}/panda_plus_bench/baseline.parquet")
    g4rows = d[d.label == 2].reset_index(drop=True)
    best = None
    for i in range(min(30, len(g4rows))):
        im = Image.open(io.BytesIO(g4rows.iloc[i].image["bytes"])).convert("RGB")
        score = float(np.mean(np.asarray(im)))  # lower = more tissue
        if best is None or score < best[0]:
            best = (score, im)
    ppb = best[1]
    peso = Image.open(f"{PROJECT_ROOT}/data/peso/png/1.jpg")
    core = Image.open(
        f"{PROJECT_ROOT}/data/arvaniti_images_extracted/"
        "ZT111_4_A/ZT111_4_A_1_12.jpg").convert("RGB")
    core = core.crop((700, 700, 2400, 2400))

    entries = [
        ("SICAPv2 (development)", sicap_img,
         "Needle biopsy patch, 112 um FOV, 4-class labels"),
        ("PAR (external, closest)", par_img,
         "Needle biopsy WSI, 0.5 um/px, 2-3 readers"),
        ("TCGA-PRAD (external)", tc_img,
         "Radical prostatectomy WSI, refined GG labels"),
        ("PANDA-PLUS-Bench", ppb,
         "Biopsy patch, 20x, 8 perturbation conditions"),
        ("PESO (external)", peso,
         "Prostatectomy region, cancer / non-cancer"),
        ("Arvaniti TMA (development, %GP4)", core,
         "TMA core, pixel-level GP3/4/5 masks"),
    ]
    for ax, (title, img, sub) in zip(axes.ravel(), entries):
        ax.imshow(img, cmap=None if img.mode == "RGB" else "gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel(sub, fontsize=6.5, color="#555555")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#DDDDDD")
    fig.suptitle("Six public cohorts used for training, transfer and auditing",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "V1_cohorts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure V4
def fig_architecture():
    fig = plt.figure(figsize=(11.5, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=color, edgecolor=color, alpha=0.10, lw=1.2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=color)
        if sub:
            ax.text(x + w / 2, y + h - 0.52, sub, ha="center", va="top",
                    fontsize=6.8, color="#333333")

    # patch image
    ann = pd.read_csv(f"{DATA_ROOT}/sicapv2/annotations_all.csv")
    g4 = ann[ann.label == 2].image_name.iloc[0]
    img = sicap_patch("SICAPv2/images/" + g4)
    iax = fig.add_axes([0.035, 0.14, 0.14, 0.58])
    iax.imshow(img)
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_visible(True); sp.set_color("#DDDDDD")
    ax.text(0.105, 0.80, "112 um tile", ha="center", fontsize=7)

    box(1.9, 1.95, 2.1, 1.5, "Frozen foundation encoder",
        "Phikon-v1 / H-optimus-0\n768 / 1536-d, no fine-tuning", C_BLUE)
    box(4.4, 1.95, 2.0, 1.5, "Shared trunk",
        "Linear 256 + GELU\n(dropout 0.2)", C_GREY)
    box(6.9, 2.85, 2.3, 1.0, "Tile head",
        "4-class CE: NC / GP3 / GP4 / GP5", C_GREEN)
    box(6.9, 1.60, 2.3, 1.0, "Cancer head (aux)",
        "binary BCE, detected-tissue signal", C_ORANGE)
    box(6.9, 0.35, 2.3, 1.0, "Attention MIL + slide head",
        "softmax attention, ISUP 1-5 CE", C_PURPLE)
    box(9.7, 2.05, 2.1, 1.6, "Outputs",
        "tile probabilities\nP(GP4|tile)\nslide grade / %GP4", C_RED)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#555555", lw=1.2))
    arrow(1.75, 2.55, 1.88, 2.55)
    arrow(4.02, 2.55, 4.38, 2.55)
    arrow(6.42, 2.7, 6.88, 3.25)
    arrow(6.42, 2.55, 6.88, 2.10)
    arrow(6.42, 2.4, 6.88, 0.90)
    arrow(9.25, 3.35, 9.68, 3.05)
    arrow(9.25, 2.10, 9.68, 2.75)
    arrow(9.25, 0.90, 9.68, 2.35)
    ax.set_title("Model architecture: frozen features with shared trunk and task heads",
                 fontsize=11, fontweight="bold", pad=4)
    fig.savefig(FIG / "V2_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure V5
def fig_pipeline():
    from clinical_utility import site_data, tau_one_sided
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=color, edgecolor=color, alpha=0.09, lw=1.1))
        ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color=color)
        if sub:
            ax.text(x + w / 2, y + h - 0.48, sub, ha="center", va="top",
                    fontsize=6.5, color="#333333")

    # training column
    img = sicap_patch(
        "SICAPv2/images/16B0003388_Block_Region_13_10_1_xini_8636_yini_63869.jpg")
    msk = sicap_mask(
        "SICAPv2/images/16B0003388_Block_Region_13_10_1_xini_8636_yini_63869.jpg")
    iax = fig.add_axes([0.03, 0.55, 0.11, 0.30])
    iax.imshow(img); iax.set_xticks([]); iax.set_yticks([])
    ax.text(0.085, 0.50, "patch + mask", ha="center", fontsize=6.5)
    m = np.array(msk)
    colors = np.zeros((*m.shape, 4), dtype=np.float32)
    colors[m == 3] = [0.30, 0.69, 0.41, 0.5]
    colors[m == 4] = [0.77, 0.31, 0.32, 0.5]
    colors[m == 5] = [0.50, 0.45, 0.70, 0.5]
    iax.imshow(colors)
    box(1.4, 2.95, 4.4, 1.35, "Training: SICAPv2 anchored",
        "18,783 tiles, 12,081 labelled; slide ISUP for 155 slides\n"
        "loss = tile CE + 0.5 * slide CE + auxiliary cancer BCE\n"
        "leave-one-cohort-out, patient-grouped", C_BLUE)

    # deployment column
    site = site_data("phikon")["PAR"]
    tab, feats, ymap = site
    keys = tab.slide.astype(str).str.lower().str.replace(
        "_leica", "", regex=False)
    X = np.stack([feats[k].mean(0) for k in keys])
    y = tab.y.to_numpy()
    rng = np.random.default_rng(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    cal = np.concatenate([rng.choice(pos, 30, replace=False),
                          rng.choice(neg, 30, replace=False)])
    sc = StandardScaler().fit(X[cal])
    lr = LogisticRegression(C=0.1, max_iter=1000,
                            class_weight="balanced").fit(sc.transform(X[cal]),
                                                         y[cal])
    p = lr.predict_proba(sc.transform(X))[:, 1]
    tau = tau_one_sided(p[cal], y[cal], 0.10)

    hax = fig.add_axes([0.56, 0.12, 0.38, 0.48])
    bins = np.linspace(0, 1, 26)
    hax.hist(p[y == 0], bins=bins, color=C_GREEN, alpha=0.55,
             label="ISUP <= 1")
    hax.hist(p[y == 1], bins=bins, color=C_RED, alpha=0.55,
             label="ISUP >= 2")
    if tau is not None:
        hax.axvline(tau, color="#333333", lw=1.2, ls="--")
        hax.axvspan(0, tau, color=C_GREEN, alpha=0.08)
        hax.axvspan(tau, 1, color=C_GREY, alpha=0.06)
        hax.text(tau / 2, hax.get_ylim()[1] * 0.92, "clear\n(AS)",
                 ha="center", fontsize=7, color=C_GREEN, fontweight="bold")
        hax.text((tau + 1) / 2, hax.get_ylim()[1] * 0.92, "defer",
                 ha="center", fontsize=7, color="#555555",
                 fontweight="bold")
    hax.set_xlabel("Refit site score (P(ISUP>=2))")
    hax.set_ylabel("Slides")
    hax.legend(frameon=False, fontsize=6.5, loc="upper center")
    hax.set_title("Site calibration: 60 labels, one-sided conformal (alpha = 0.10)",
                  fontsize=8.5)
    hax.tick_params(labelsize=6.5)

    box(1.4, 0.55, 4.4, 1.35, "Deployment at a new site",
        "K=60 labelled slides -> refit probe -> temperature scaling\n"
        "conformal threshold tau at alpha; clear if score <= tau\n"
        "else defer to a pathologist; report workload and FN", C_ORANGE)
    ax.set_title("Algorithm pipeline: training with weak labels and risk-controlled deployment",
                 fontsize=11, fontweight="bold", pad=2)
    fig.savefig(FIG / "V3_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure V6
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
                picks.append((s, hits[0]))
                break
    fig, axes = plt.subplots(len(picks), 6, figsize=(12, 2.5 * len(picks)),
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1, 1]})
    for row, (s, fid) in enumerate(picks):
        sev = int(par_lab[par_lab.slide == s].maj.iloc[0])
        z = np.load(fid)
        x = torch.from_numpy(z["tile_emb"].astype(np.float32)).to(dev)
        with torch.no_grad():
            h = model.trunk(x)
            a = torch.softmax(model.att(h).squeeze(-1), dim=0).cpu().numpy()
        coords = z["coords"]
        order = np.argsort(-a)[:5]
        path = ROOT / f"data/par/wsi/{s}_leica.svs"
        sl = tiffslide.TiffSlide(str(path))
        thumb, _, (sx, sy) = thumbnail(sl)
        sl.close()
        ax0 = axes[row, 0]
        ax0.imshow(thumb)
        ax0.scatter(coords[order, 0] / sx, coords[order, 1] / sy, s=22,
                    facecolors="none", edgecolors=C_RED, lw=1.3)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(f"ISUP {int(sev)}", fontsize=8)
        for j, idx in enumerate(order):
            ax = axes[row, j + 1]
            crop = wsi_crop(path, int(coords[idx, 0]), int(coords[idx, 1]),
                            out_px=224)
            ax.imshow(crop)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"att {a[idx]:.3f}", fontsize=6.5, pad=2)
    fig.suptitle("What the model attends to: top-attended tiles per slide",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "V4_attention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure V7
def fig_tma_heatmap():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    files = sorted((ROOT / "runs/gp4_dense/tiles/hoptimus0").glob("*.npz"))
    xs, ys, cores = [], [], []
    for f in files:
        z = np.load(f)
        if "emb_s1" not in z.files:
            continue
        xs.append(z["emb_s1"]); ys.append(z["frac"][:, 1:4].argmax(1))
        cores.extend([str(z["core"])] * len(z["emb_s1"]))
    X = np.concatenate(xs); ycls = np.concatenate(ys)
    rng = np.random.default_rng(0)
    tr = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.03, max_iter=800,
                            class_weight="balanced").fit(sc.transform(X[tr]),
                                                         ycls[tr])
    # choose a core with many GP4 tiles
    counts = {}
    for f in files:
        z = np.load(f)
        if "emb_s1" in z.files:
            counts[str(z["core"])] = int((z["frac"][:, 2] > 0.3).sum())
    core = max(counts, key=counts.get)
    z = np.load(ROOT / f"runs/gp4_dense/tiles/hoptimus0/{core}.npz")
    p = lr.predict_proba(sc.transform(z["emb_s1"]))
    cls = list(lr.classes_)
    p4 = p[:, cls.index(1)] if 1 in cls else np.zeros(len(p))
    img = Image.open(ROOT / f"data/arvaniti_images_extracted/"
                     f"{core.split('_')[0].upper()}_...").convert("RGB") \
        if False else None
    # locate the core image via tma_pairs
    pairs = pd.read_csv(ROOT / "data/tma_pairs.csv")
    row = pairs[pairs.key == core].iloc[0]
    core_img = Image.open(row["image"]).convert("RGB")
    mask = np.array(Image.open(row["mask"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    axes[0].imshow(core_img); axes[0].set_title(
        f"TMA core {core.upper()}", fontsize=8.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    # model heat
    ax = axes[1]
    ax.imshow(core_img, alpha=0.55)
    for xi, yi, pi in zip(z["x"], z["y"], p4):
        ax.add_patch(mpatches.Rectangle((xi, yi), 224, 224,
                                        facecolor=plt.cm.inferno(pi),
                                        edgecolor="none", alpha=0.45))
    ax.set_title("Model P(GP4) per tile", fontsize=8.5)
    # mask
    ax = axes[2]
    ax.imshow(core_img)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 2] = [0.77, 0.31, 0.32, 0.45]
    ax.imshow(overlay)
    ax.set_title("Pathologist GP4 mask", fontsize=8.5)
    fig.suptitle("Dense attention map vs expert annotation",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "V5_tma_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_cohorts()
    fig_architecture()
    fig_pipeline()
    fig_attention()
    fig_tma_heatmap()
    print("visual figures written to", FIG)


if __name__ == "__main__":
    main()
