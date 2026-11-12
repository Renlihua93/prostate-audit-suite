#!/usr/bin/env python3
"""WSI tile embedding with any fm_encoders encoder.

Same sampling protocol as extract_phikon_features.py (0.5 um/px, 224 px,
random tissue tiles, max 2000 per slide) but uses the unified Encoder
preprocessing/forward so hoptimus0 / resnet50 / phikon_v2 work too.

Output per slide: npz with mean/std/coords/tile_emb, plus slide_features.csv.
Resumable: slides already in slide_features.csv are skipped.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tiffslide
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_phikon_features import (  # noqa: E402
    choose_level, qc_features, thumbnail, tissue_mask,
)
from fm_encoders import load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def extract_slide(slide, encoder, target_mpp, tile_px, max_tiles, seed=0):
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
    level, actual_mpp = choose_level(slide, target_mpp, mpp or target_mpp)
    size = max(8, int(round(tile_px * target_mpp / actual_mpp)))
    downsample = slide.level_downsamples[level]
    footprint = max(1, int(round(size * downsample)))

    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    if len(ys):
        order = rng.permutation(len(ys))
        ys, xs = ys[order], xs[order]
    coords = []
    seen = set()
    for ty, tx in zip(ys, xs):
        x0 = int(tx / sx) // footprint * footprint
        y0 = int(ty / sy) // footprint * footprint
        if x0 + footprint > w or y0 + footprint > h or (x0, y0) in seen:
            continue
        seen.add((x0, y0))
        coords.append((x0, y0))
        if len(coords) >= max_tiles:
            break
    if not coords:
        return None

    imgs = []
    for x0, y0 in coords:
        region = slide.read_region((x0, y0), level, (size, size))
        region = region.convert("RGB")
        if region.size != (tile_px, tile_px):
            region = region.resize((tile_px, tile_px), Image.BILINEAR)
        imgs.append(region)
    chunks = [encoder.embed(imgs[i:i + 16]).astype(np.float32)
              for i in range(0, len(imgs), 16)]
    emb = np.concatenate(chunks)
    return {"tile_emb": emb, "coords": np.array(coords, dtype=np.int32),
            "mean": emb.mean(0), "std": emb.std(0),
            "n_tiles": len(emb), "level": level, "size_px": (w, h),
            "actual_mpp": actual_mpp, "tissue_frac": float(mask.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", default="hoptimus0")
    ap.add_argument("--target-mpp", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.images.rglob("*")
                   if p.suffix.lower() in (".svs", ".ndpi", ".tif", ".tiff"))
    rows = [(p.stem.lower(), p) for p in paths]
    table_path = args.out / "slide_features.csv"
    table = (pd.read_csv(table_path).to_dict("records")
             if table_path.exists() else [])
    done = {str(r["slide_id"]).lower() for r in table}
    rows = [(s, p) for s, p in rows if s not in done]
    if args.limit:
        rows = rows[: args.limit]

    enc = load_encoder(args.encoder)
    print(f"encoder={args.encoder} slides={len(rows)} done={len(done)}",
          flush=True)
    for i, (sid, path) in enumerate(rows, 1):
        t0 = time.time()
        try:
            slide = tiffslide.TiffSlide(str(path))
            try:
                res = extract_slide(slide, enc, args.target_mpp, args.tile,
                                    args.max_tiles, seed=i)
            finally:
                slide.close()
            if res is None:
                print(f"[skip] {sid}: no tiles", flush=True)
                continue
            np.savez_compressed(args.out / f"{sid}.npz",
                                mean=res["mean"], std=res["std"],
                                coords=res["coords"],
                                tile_emb=res["tile_emb"])
            row = {"slide_id": sid, "n_tiles": res["n_tiles"],
                   "level": res["level"], "size_px": res["size_px"],
                   "actual_mpp": res["actual_mpp"],
                   "tissue_frac": res["tissue_frac"],
                   "tile_emb_file": f"{sid}.npz"}
            table.append(row)
            tmp = table_path.with_suffix(".csv.tmp")
            pd.DataFrame(table).to_csv(tmp, index=False)
            tmp.replace(table_path)
            print(f"[{i}/{len(rows)}] {sid} tiles={res['n_tiles']} "
                  f"{time.time()-t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {sid}: {exc}", flush=True)
            continue
    print("done", len(table))


if __name__ == "__main__":
    main()
