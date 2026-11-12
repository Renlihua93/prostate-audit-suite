#!/usr/bin/env python3
"""Tile features for PESO region crops (cancer vs non-cancer regions).

PESO test set: 40 slides, 160 annotated regions exported as 2500x2500 crops
in data/peso/png/<id>.jpg with labels in peso_testset_mapping.csv.
Tiles 224 px, stride 224 (~121 tiles per crop), embedded with fm_encoders.
Assumes 0.5 um/px (20x) as for the other cohorts; recorded as a caveat.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_encoders import load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(f"{DATA_ROOT}/peso") if Path(
    f"{DATA_ROOT}/peso").exists() else ROOT / "data/peso"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"features/peso_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(ROOT / "data/peso_testset_mapping.csv")
    enc = load_encoder(args.encoder)
    done = {p.stem for p in out.glob("*.npz")}
    print(f"encoder={args.encoder} regions={len(mapping)} done={len(done)}",
          flush=True)
    for i, r in mapping.iterrows():
        rid = str(int(r.id))
        if rid in done:
            continue
        p = ROOT / f"data/peso/png/{rid}.jpg"
        if not p.exists():
            print("[skip]", rid, flush=True)
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        imgs = []
        for y in range(0, h - args.tile + 1, args.stride):
            for x in range(0, w - args.tile + 1, args.stride):
                imgs.append(im.crop((x, y, x + args.tile, y + args.tile)))
        chunks = [enc.embed(imgs[j:j + 32])
                  for j in range(0, len(imgs), 32)]
        emb = np.concatenate(chunks).astype(np.float32)
        np.savez_compressed(out / f"{rid}.npz", tile_emb=emb,
                            label=int(r.type == "cancer"),
                            slide=int(r.slide), region_type=str(r.type))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(mapping)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
