#!/usr/bin/env python3
"""Derive slide-level %GP4 from SICAPv2 pixel masks.

Mask values: 0 = non-cancer/background, 3/4/5 = GP3/GP4/GP5 pixels.
Per patch, %GP4 = GP4 / (GP3+GP4+GP5); slide %GP4 pools pixels over all
patches of the slide ("%GP4 of sampled tumour area").
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import collections
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")
OUT = Path(__file__).resolve().parents[1] / "runs/sicap_gp4"
OUT.mkdir(parents=True, exist_ok=True)


def slide_of(name: str) -> str:
    return name.split("_Block")[0].lower()


def main():
    zf = zipfile.ZipFile(ZIP)
    names = [n for n in zf.namelist() if n.endswith(".png")]
    rows = []
    for i, n in enumerate(names):
        a = np.asarray(Image.open(io.BytesIO(zf.read(n))))
        v, c = np.unique(a, return_counts=True)
        d = dict(zip(v.tolist(), c.tolist()))
        g3, g4, g5 = d.get(3, 0), d.get(4, 0), d.get(5, 0)
        tot = g3 + g4 + g5
        rows.append({"image_name": Path(n).name.replace(".png", ".jpg"),
                     "slide": slide_of(Path(n).name),
                     "gp3_px": g3, "gp4_px": g4, "gp5_px": g5,
                     "tumour_px": tot,
                     "gp4_frac": g4 / tot if tot else np.nan})
        if (i + 1) % 2000 == 0:
            print(i + 1, "/", len(names), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "patch_gp4_from_masks.csv", index=False)

    g = df.groupby("slide")
    agg = g[["gp3_px", "gp4_px", "gp5_px", "tumour_px"]].sum()
    agg["n_patches"] = g.size()
    agg["gp4_frac"] = agg.gp4_px / agg.tumour_px.replace(0, np.nan)
    agg = agg.reset_index()

    xl = pd.read_excel(io.BytesIO(
        zf.read("SICAPv2/wsi_labels.xlsx")))
    xl["slide"] = xl.slide_id.astype(str).str.lower()
    merged = agg.merge(xl.drop(columns="slide_id"), on="slide", how="left")
    merged.to_csv(OUT / "slide_gp4_from_masks.csv", index=False)
    print("slides", len(merged), "with tumour", int(merged.tumour_px.gt(0).sum()))
    print(merged[["slide", "n_patches", "tumour_px", "gp4_frac",
                  "Gleason_primary", "Gleason_secondary"]]
          .sort_values("gp4_frac").head(5).to_string())
    print(merged.gp4_frac.describe().to_string())


if __name__ == "__main__":
    main()
