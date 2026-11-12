#!/usr/bin/env python3
"""Streaming WSI feature extraction for disk-constrained cohorts.

For each slide: download to a temporary file, extract tiles at a fixed physical
resolution, embed them with one or more foundation encoders, save a compact
npz, then delete the WSI. Resumable: slides with an existing npz are skipped,
partial downloads resume.

Example:
  HF_ENDPOINT=https://hf-mirror.com py stream_extract.py \
      --manifest ../data/panda-plus/manifest.csv --url-prefix https://... \
      --encoders phikon_v2 hoptimus0 --out ../features/panda_plus_dual
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_phikon_features import (  # noqa: E402
    MEAN, STD, choose_level, qc_features, thumbnail, tissue_mask, TILE_PX,
)

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {
    "phikon": "owkin/phikon",
    "phikon_v2": "owkin/phikon-v2",
    "hoptimus0": "bioptimus/H-optimus-0",
    "hoptimus1": "bioptimus/H-optimus-1",
    "hibou_l": "histai/hibou-L",
}


def log(msg: str, path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download(url: str, dest: Path, log_path: Path, retries: int = 6) -> bool:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file://"):
        src = Path(url[len("file://"):])
        dest.write_bytes(b"") if not dest.exists() else None
        shutil.copyfile(src, dest)
        return True
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120,
                              allow_redirects=True) as r:
                if have and r.status_code == 200:
                    have = 0  # server ignored Range; restart
                r.raise_for_status()
                mode = "ab" if have else "wb"
                with dest.open(mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            log(f"[download] retry {attempt + 1}/{retries}: {exc}", log_path)
            time.sleep(2 ** attempt)
    return False


NAME_TO_DIR = {"phikon": "phikon", "phikon_v2": "phikon-v2",
               "hoptimus0": "h-optimus-0", "hoptimus1": "h-optimus-1",
               "hibou_l": "hibou-L"}


def load_encoder(name: str, device: str):
    """Load a local checkpoint when present, otherwise fall back to the hub id."""
    from transformers import AutoModel
    local = ROOT / "models" / NAME_TO_DIR.get(name, name)
    if (local / "config.json").exists():
        model = AutoModel.from_pretrained(str(local), trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(ENCODERS[name], trust_remote_code=True)
    return model.eval().to(device)


def sample_tiles(slide, target_mpp: float, tile_px: int, max_tiles: int, seed: int):
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
    ds = slide.level_downsamples[level]
    step = max(1, int(round(size)))
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    order = rng.permutation(len(xs))
    coords = []
    seen = set()
    for k in order:
        tx = int(xs[k] / sx / ds) * step
        ty = int(ys[k] / sy / ds) * step
        if tx + size > w or ty + size > h or (tx, ty) in seen:
            continue
        seen.add((tx, ty))
        coords.append((tx * int(ds), ty * int(ds)))
        if len(coords) >= max_tiles:
            break
    return {"level": level, "actual_mpp": actual_mpp, "size": size,
            "coords": coords, "thumb": thumb, "mask": mask,
            "tissue_fraction": float(mask.mean())}


def embed_tiles(slide, coords, level, size, tile_px, model, device, batch):
    embs = []
    for i in range(0, len(coords), batch):
        tiles = []
        for x0, y0 in coords[i:i + batch]:
            region = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
            if region.size != (tile_px, tile_px):
                region = region.resize((tile_px, tile_px), Image.BILINEAR)
            tiles.append(np.array(region, dtype=np.float32) / 255.0)
        x = (np.stack(tiles) - MEAN) / STD
        xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device == "cuda"):
            out = model(xb)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            embs.append(h[:, 0].float().cpu().numpy())
    return np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--id-col", default="slide_id")
    ap.add_argument("--url-col", default=None)
    ap.add_argument("--url-prefix", default="")
    ap.add_argument("--size-col", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoders", nargs="+", default=["phikon_v2"],
                    choices=sorted(ENCODERS))
    ap.add_argument("--tmp-dir", type=Path, default=None)
    ap.add_argument("--target-mpp", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=TILE_PX)
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--keep-wsi", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.tmp_dir or args.out / ".wsi_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "stream_extract.log"
    table_path = args.out / "slide_features.csv"
    table = pd.read_csv(table_path).to_dict("records") if table_path.exists() else []
    done = {str(r[args.id_col]).lower() for r in table}

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    log(f"slides={len(df)} done={len(done)} encoders={args.encoders} device={device}",
        log_path)

    models = {name: load_encoder(name, device) for name in args.encoders}

    import tiffslide
    for i, (_, r) in enumerate(df.iterrows(), 1):
        sid = str(r[args.id_col]).lower()
        npz_path = args.out / f"{sid}.npz"
        if sid in done or npz_path.exists():
            continue
        if free_gb(args.out) < args.min_free_gb:
            log(f"[stop] free disk {free_gb(args.out):.1f} GB < {args.min_free_gb}",
                log_path)
            break
        if args.url_col:
            url = str(r[args.url_col])
        elif args.url_prefix:
            url = args.url_prefix + str(r[args.id_col])
        else:
            raise SystemExit("provide --url-col or --url-prefix")
        local = Path(url.split("?")[0])
        wsi = tmp / local.name
        t0 = time.time()
        log(f"[{i}/{len(df)}] {sid}: downloading {url.split('/')[-1]}", log_path)
        if not download(url, wsi, log_path):
            log(f"[fail] {sid}: download failed", log_path)
            continue
        expected = None
        if args.size_col:
            try:
                expected = int(r[args.size_col])
            except (TypeError, ValueError):
                expected = None
        if expected and wsi.stat().st_size != expected:
            log(f"[fail] {sid}: size {wsi.stat().st_size}/{expected}", log_path)
            wsi.unlink(missing_ok=True)
            continue

        slide = tiffslide.TiffSlide(str(wsi))
        try:
            t = sample_tiles(slide, args.target_mpp, args.tile,
                             args.max_tiles, seed=i)
            if t is None:
                log(f"[skip] {sid}: no tissue", log_path)
                continue
            arrays = {"coords": np.array(t["coords"], dtype=np.int32)}
            for name, model in models.items():
                emb = embed_tiles(slide, t["coords"], t["level"], t["size"],
                                  args.tile, model, device, args.batch)
                arrays[f"tile_{name}"] = emb
                arrays[f"mean_{name}"] = emb.mean(axis=0)
                arrays[f"std_{name}"] = emb.std(axis=0)
        finally:
            slide.close()
        np.savez_compressed(npz_path, **arrays)
        if not args.keep_wsi:
            wsi.unlink(missing_ok=True)

        row = {"slide_id": sid, "n_tiles": len(t["coords"]), "level": t["level"],
               "size_px": t["size"], "actual_mpp": t["actual_mpp"]}
        row.update(qc_features(t))
        table.append(row)
        pd.DataFrame(table).to_csv(table_path.with_suffix(".csv.tmp"), index=False)
        (table_path.with_suffix(".csv.tmp")).replace(table_path)
        log(f"[done] {sid} tiles={len(t['coords'])} {time.time()-t0:.0f}s "
            f"free={free_gb(args.out):.1f}GB", log_path)

    log("finished", log_path)


if __name__ == "__main__":
    main()
