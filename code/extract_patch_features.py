#!/usr/bin/env python3
"""Extract foundation-model features from patch datasets.

Supports two layouts:
  --parquet  a Parquet file with columns image (bytes), label, slide_id
  --zip      a zip of JPEGs plus --annotations (xlsx/csv with image_name and
             one-hot class columns NC/G3/G4/G5)

Example:
  py extract_patch_features.py --parquet ../data/panda_plus_bench/baseline.parquet \
      --encoders phikon_v2 --out ../features/ppbench_phikon_v2
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_encoders import SPECS as ENCODERS, load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_bytes(img) -> bytes:
    if isinstance(img, dict):
        return img.get("bytes") or Path(img["path"]).read_bytes()
    return img


def iter_parquet(path: Path, batch: int):
    df = pd.read_parquet(path)
    for i in range(0, len(df), batch):
        chunk = df.iloc[i:i + batch]
        imgs = [Image.open(io.BytesIO(_to_bytes(b))) for b in chunk.image]
        yield imgs, chunk[["label", "slide_id"]].reset_index(drop=True)


def iter_zip(path: Path, ann_path: Path, batch: int):
    if ann_path.suffix in (".xlsx", ".xls"):
        ann = pd.read_excel(ann_path)
    else:
        ann = pd.read_csv(ann_path)
    class_cols = [c for c in ann.columns if c in ("NC", "G3", "G4", "G5")]
    ann["label"] = ann[class_cols].to_numpy().argmax(axis=1)
    label_map = dict(zip(ann.image_name, ann.label))
    zf = zipfile.ZipFile(path)
    names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".png"))]
    rows = [(n, label_map[Path(n).name]) for n in names if Path(n).name in label_map]
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        imgs = [Image.open(io.BytesIO(zf.read(n))) for n, _ in chunk]
        meta = pd.DataFrame({"label": [l for _, l in chunk],
                             "slide_id": [n.split("_Block")[0].lower() for n, _ in chunk]})
        yield imgs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--annotations", type=Path)
    ap.add_argument("--encoders", nargs="+", default=["phikon_v2"],
                    choices=sorted(ENCODERS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--center-crop", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {n: load_encoder(n, device) for n in args.encoders}

    feats = {n: [] for n in args.encoders}
    labels, slides, seen = [], [], 0
    if args.parquet:
        it = iter_parquet(args.parquet, args.batch)
    elif args.zip and args.annotations:
        it = iter_zip(args.zip, args.annotations, args.batch)
    else:
        raise SystemExit("provide --parquet or --zip + --annotations")

    t0 = time.time()
    for imgs, meta in it:
        if args.center_crop:
            c = args.center_crop
            cropped = []
            for im in imgs:
                im = im.convert("RGB")
                w, h = im.size
                x0, y0 = (w - c) // 2, (h - c) // 2
                cropped.append(im.crop((x0, y0, x0 + c, y0 + c)))
            imgs = cropped
        for n, model in models.items():
            feats[n].append(model.embed(imgs))
        labels.extend(meta.label.tolist())
        slides.extend(meta.slide_id.astype(str).str.lower().tolist())
        seen += len(imgs)
        if seen % (args.batch * 10) == 0:
            print(f"{seen} patches {time.time()-t0:.0f}s", flush=True)
        if args.limit and seen >= args.limit:
            break

    out = {f"emb_{n}": np.concatenate(v) for n, v in feats.items()}
    out["labels"] = np.array(labels)
    out["slide_ids"] = np.array(slides)
    np.savez_compressed(args.out / "patch_features.npz", **out)
    print("saved", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
