#!/usr/bin/env python3
"""MGOD stress test: SICAP-trained tile head on PANDA-PLUS-Bench conditions.

Trains tile-only and tile+slide-MIL heads (Phikon-v1, 112 um FOV) on all
labelled SICAPv2 patches, then evaluates each PPBench augmentation condition
(AUC G4+, 3-class accuracy). Three seeds.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from mgod_pilot import Head, load_sicap, train

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["baseline", "color_jitter", "combined_aggressive", "gaussian_noise",
         "grayscale", "heavy_geometric", "hed_stain_augmentation",
         "macenko_normalization"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    out = ROOT / f"runs/mgod_ppbench_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"epochs": 10, "batch": 256, "lam": 0.5}

    conds = {}
    for c in CONDS:
        base = "ppbench_phikon" if args.encoder == "phikon" else "ppbench"
        p = ROOT / f"features/{base}/{c}/patch_features.npz"
        if p.exists():
            d = np.load(p)
            conds[c] = (d[f"emb_{args.encoder}"].astype(np.float32),
                        d["labels"].astype(int))

    rows = []
    for seed in args.seeds:
        for mode, lam in (("tile_only", 0.0), ("tile_plus_slide", 0.5)):
            params["lam"] = lam
            model = train(mode, emb, labels, slides, gg, params, seed=seed)
            model.eval()
            for c, (X, y) in conds.items():
                with torch.no_grad():
                    lt, _ = model(torch.from_numpy(X).to(dev))
                    prob = torch.softmax(lt, -1).cpu().numpy()
                score = prob[:, 2:].sum(1)
                pred = prob.argmax(1)
                pred = np.where(pred == 3, 2, pred)  # no GP5 in PPBench
                row = {"seed": seed, "mode": mode, "condition": c,
                       "auc_g4plus": roc_auc_score((y >= 2).astype(int),
                                                   score),
                       "acc3": accuracy_score(y, pred)}
                rows.append(row)
                print(f"seed{seed} {mode} {c}: "
                      f"AUC={row['auc_g4plus']:.3f} acc={row['acc3']:.3f}",
                      flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    piv = df.groupby(["mode", "condition"]).agg(
        auc=("auc_g4plus", "mean"), acc=("acc3", "mean")).round(3)
    print(piv.to_string())
    df.groupby("mode").agg(auc=("auc_g4plus", "mean"),
                           acc=("acc3", "mean")).round(3).to_csv(
        out / "summary.csv")


if __name__ == "__main__":
    main()
