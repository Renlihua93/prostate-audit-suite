#!/usr/bin/env python3
"""PESO external test: region-level cancer detection with SICAP-trained heads.

Models (Phikon-v1): linear probe (C=0.001, raw features) and MGOD head
(tile-CE + SICAP slide MIL). Score per region = mean tile P(any cancer);
AUC over 160 annotated regions, plus slide-level (mean region score per
slide, slide label = any cancer region).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mgod_pilot import load_sicap, train

ROOT = Path(__file__).resolve().parents[1]


def load_peso(encoder):
    d = ROOT / f"features/peso_{encoder}"
    rows = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        rows.append({"id": f.stem, "slide": int(z["slide"]),
                     "y": int(z["label"]),
                     "emb": z["tile_emb"].astype(np.float32)})
    return rows


def region_scores(score_fn, rows):
    return np.array([score_fn(r["emb"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    rows = load_peso(args.encoder)
    y = np.array([r["y"] for r in rows])
    slides_p = np.array([r["slide"] for r in rows])
    print(f"regions={len(rows)} cancer={int(y.sum())} slides={len(set(slides_p))}",
          flush=True)

    results = {}

    # linear probe
    lr = LogisticRegression(C=0.001, max_iter=2000, class_weight="balanced")
    lr.fit(emb[keep], labels[keep])
    cls = list(lr.classes_)
    idx = [i for i, c in enumerate(cls) if c >= 1]
    def lin_score(x):
        p = lr.predict_proba(x)
        return float(p[:, idx].sum(1).mean())
    s_lin = region_scores(lin_score, rows)
    results["linear"] = {"region_auc": float(roc_auc_score(y, s_lin))}

    # MGOD heads (tile-CE + SICAP slide MIL), 3 seeds
    aucs = []
    for seed in args.seeds:
        params = {"epochs": 10, "batch": 256, "lam": 0.5}
        gg_keep = {s: gg[s] for s in set(slides[keep]) if s in gg}
        model = train("tile_plus_slide", emb[keep], labels[keep],
                      slides[keep], gg_keep, params, seed=seed)
        model.eval()
        def mgod_score(x):
            with torch.no_grad():
                lt, _ = model(torch.from_numpy(x).to(dev))
                return float(torch.softmax(lt, -1)[:, 1:].sum(1).mean())
        s = region_scores(mgod_score, rows)
        auc = roc_auc_score(y, s)
        aucs.append(auc)
        print(f"seed{seed} mgod region AUC={auc:.3f}", flush=True)
    results["mgod"] = {"region_auc_mean": float(np.mean(aucs)),
                       "region_auc_sd": float(np.std(aucs))}

    # slide-level sensitivity (any cancer region per slide)
    df = pd.DataFrame({"slide": slides_p, "y": y, "s_lin": s_lin})
    g = df.groupby("slide").agg(y=("y", "max"), s=("s_lin", "mean"))
    results["linear"]["slide_auc"] = float(roc_auc_score(g.y, g.s))
    out = ROOT / f"runs/peso_eval_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
