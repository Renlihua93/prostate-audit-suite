#!/usr/bin/env python3
"""Extract Phikon tile embeddings from prostate WSIs.

For each slide: build a tissue mask from a thumbnail, sample non-overlapping
224 px tiles at the requested physical resolution, run Phikon (ViT-B) and store
per-tile embeddings plus slide-level mean/std. Tile-level arrays are kept so a
MIL model can be trained later without re-reading the WSIs.

Example:
  set HF_ENDPOINT=https://hf-mirror.com
  py extract_phikon_features.py --files ../data/par/par_train_120.csv \
      --wsi-dir ../data/par/wsi --out ../features/phikon --limit 7
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REGION_MPP = 0.5
TILE_PX = 224


def tissue_mask(thumb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    mask = ((hsv[:, :, 1] > 20) & (gray < 225)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def thumbnail(slide, max_dim: int = 1024):
    w, h = slide.level_dimensions[0]
    level = slide.get_best_level_for_downsample(max(w, h) / max_dim)
    tw, th = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (tw, th)).convert("RGB")
    return np.array(img), level, (tw / w, th / h)


def choose_level(slide, target_mpp: float, mpp: float) -> tuple[int, float]:
    downsample = target_mpp / mpp if mpp else 1.0
    level = 0
    if downsample > 1.0:
        level = slide.get_best_level_for_downsample(downsample)
    actual = mpp * slide.level_downsamples[level]
    return level, actual


def extract_slide(slide, model, device, target_mpp: float, tile_px: int,
                  max_tiles: int, batch: int, seed: int = 0):
    w, h = slide.level_dimensions[0]
    mpp = None
    for key in ("openslide.mpp-x", "tiffslide.mpp-x"):
        try:
            mpp = float(slide.properties.get(key))
            break
        except (TypeError, ValueError):
            continue
    thumb, _, (sx, sy) = thumbnail(slide)
    mask = tissue_mask(thumb)
    tissue_frac = float(mask.mean())

    level, actual_mpp = choose_level(slide, target_mpp, mpp or target_mpp)
    # Read a region covering tile_px * target_mpp microns, then resize to
    # tile_px. The previous inverted ratio read ~31 microns per tile at
    # 0.264 um/px (effective ~0.14 um/px), out of distribution for Phikon.
    size = max(8, int(round(tile_px * target_mpp / actual_mpp)))
    downsample = slide.level_downsamples[level]
    footprint = max(1, int(round(size * downsample)))

    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    if len(ys) > 0:
        order = rng.permutation(len(ys))
        ys, xs = ys[order], xs[order]

    coords = []
    seen = set()
    for ty, tx in zip(ys, xs):
        x0 = int(tx / sx) // footprint * footprint
        y0 = int(ty / sy) // footprint * footprint
        if x0 + footprint > w or y0 + footprint > h:
            continue
        if (x0, y0) in seen:
            continue
        seen.add((x0, y0))
        coords.append((x0, y0))
        if len(coords) >= max_tiles:
            break

    if not coords:
        return None

    embs = []
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = []
        for x0, y0 in chunk:
            region = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
            if region.size != (tile_px, tile_px):
                region = region.resize((tile_px, tile_px), Image.BILINEAR)
            tiles.append(np.array(region, dtype=np.float32) / 255.0)
        x = (np.stack(tiles) - MEAN) / STD
        xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(xb).last_hidden_state[:, 0]
        embs.append(out.float().cpu().numpy())
    emb = np.concatenate(embs)
    coords_arr = np.array(coords, dtype=np.int32)
    return {
        "tile_emb": emb.astype(np.float16),
        "coords": coords_arr,
        "mean": emb.mean(axis=0).astype(np.float32),
        "std": emb.std(axis=0).astype(np.float32),
        "n_tiles": len(coords),
        "tissue_fraction": tissue_frac,
        "level": level,
        "size_px": size,
        "actual_mpp": actual_mpp,
        "thumb": thumb,
        "mask": mask,
    }


def qc_features(res) -> dict:
    thumb = res["thumb"]
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    tissue = res["mask"].astype(bool)
    g = gray[tissue] if tissue.sum() > 50 else gray.ravel()
    hist = np.histogram(g, bins=256, range=(0, 255))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "q_tissue_fraction": res["tissue_fraction"],
        "q_gray_mean": float(g.mean()),
        "q_gray_std": float(g.std()),
        "q_gray_entropy": ent,
        "q_sat_mean": float(hsv[:, :, 1][tissue].mean()) if tissue.sum() > 50 else float(hsv[:, :, 1].mean()),
        "q_laplacian_var": float(lap.var()),
        "q_edge_density": float((cv2.Canny(gray, 50, 150) > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=Path, default=None, help="manifest CSV with slide_id + leica_path")
    ap.add_argument("--wsi-dir", type=Path, default=Path("../data/par/wsi"))
    ap.add_argument("--images", type=Path, default=None, help="directory or file list instead of manifest")
    ap.add_argument("--out", type=Path, default=Path("../features/phikon"))
    ap.add_argument("--model", default="owkin/phikon")
    ap.add_argument("--target-mpp", type=float, default=REGION_MPP)
    ap.add_argument("--tile", type=int, default=TILE_PX)
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import tiffslide
    from transformers import AutoModel

    if args.files:
        df = pd.read_csv(args.files)
        rows = []
        for _, r in df.iterrows():
            name = Path(str(r.get("leica_path", ""))).name
            p = args.wsi_dir / name
            expected = int(r.get("leica_size", 0) or 0)
            if not p.exists():
                continue
            if expected and p.stat().st_size != expected:
                print(f"[skip] {name}: incomplete {p.stat().st_size}/{expected}")
                continue
            rows.append((str(r["slide_id"]).lower(), p))
    elif args.images:
        base = args.images
        paths = sorted([base] if base.is_file() else [p for p in base.rglob("*")
                                                       if p.suffix.lower() in (".svs", ".ndpi", ".tif", ".tiff")])
        rows = [(p.stem.lower(), p) for p in paths]
    else:
        raise SystemExit("Provide --files or --images")
    if args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    table_path = args.out / "slide_features.csv"
    table = pd.read_csv(table_path).to_dict("records") if table_path.exists() else []
    done = {str(r["slide_id"]).lower() for r in table}
    rows = [(sid, p) for sid, p in rows if sid not in done]

    model = AutoModel.from_pretrained(args.model).eval().to(args.device)
    print(f"model={args.model} device={args.device} slides={len(rows)} (skipped {len(done)} done)")

    for i, (sid, path) in enumerate(rows, 1):
        t0 = time.time()
        try:
            slide = tiffslide.TiffSlide(str(path))
            try:
                res = extract_slide(slide, model, args.device, args.target_mpp,
                                    args.tile, args.max_tiles, args.batch,
                                    seed=i)
            finally:
                slide.close()
            if res is None:
                print(f"[skip] {sid}: no tissue tiles", flush=True)
                continue
            np.savez_compressed(args.out / f"{sid}.npz",
                                mean=res["mean"], std=res["std"],
                                coords=res["coords"],
                                tile_emb=res["tile_emb"])
            row = {"slide_id": sid, "n_tiles": res["n_tiles"],
                   "level": res["level"], "size_px": res["size_px"],
                   "actual_mpp": res["actual_mpp"],
                   "tile_emb_file": str((args.out / f"{sid}.npz").name)}
            row.update(qc_features(res))
            table.append(row)
            df = pd.DataFrame(table)
            tmp = table_path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(table_path)
            print(f"[{i}/{len(rows)}] {sid} tiles={res['n_tiles']} "
                  f"mpp={res['actual_mpp']:.3f} {time.time()-t0:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {sid}: {exc}", flush=True)
            continue

    print(f"done: {len(table)} slides in {table_path}")


if __name__ == "__main__":
    main()
