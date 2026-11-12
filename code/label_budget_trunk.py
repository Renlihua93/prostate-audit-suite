#!/usr/bin/env python3
"""Label-budget test with trunk-pooled representations.

For tile-only (lam=0) and weak-supervised (lam=0.5) heads:
  - pool the trained trunk over each target slide (mean of h)
  - refit a few-shot probe on K site labels (repeated splits)
  - calibrate a one-sided conformal threshold at alpha and evaluate clearing

This is the decisive test of whether weak supervision reduces the number of
site labels needed for safe clearing (previous version used frozen embeddings
and was therefore independent of the trained head).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clinical_utility import site_data, tau_one_sided
from mgod_pilot import load_sicap, train
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def trunk_slide_feats(model, feats, dev):
    out = {}
    with torch.no_grad():
        for s, x in feats.items():
            h = model.trunk(torch.from_numpy(x).to(dev))
            out[s] = h.mean(0).cpu().numpy()
    return out


def simulate(X, y, Ks, alpha, repeats, rng):
    n = len(y)
    rows = []
    for K in Ks:
        for rep in range(repeats):
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            kpos = min(K // 2, len(pos))
            kneg = min(K - kpos, len(neg))
            cal = np.concatenate([rng.choice(pos, kpos, replace=False),
                                  rng.choice(neg, kneg, replace=False)])
            test = np.setdiff1d(np.arange(n), cal)
            if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                continue
            sc = StandardScaler().fit(X[cal])
            lr = LogisticRegression(C=0.1, max_iter=1000,
                                    class_weight="balanced")
            lr.fit(sc.transform(X[cal]), y[cal])
            p_all = lr.predict_proba(sc.transform(X))[:, 1]
            tau = tau_one_sided(p_all[cal], y[cal], alpha)
            row = {"K": K, "rep": rep, "feasible": tau is not None}
            if tau is not None:
                clear = p_all[test] <= tau
                yt = y[test]
                fn = int((clear & (yt == 1)).sum())
                row.update({"cleared": float(clear.mean()),
                            "missed_per100": 100 * fn / len(yt),
                            "fn_clear": fn / max(int(clear.sum()), 1),
                            "sens": 1 - fn / max(int(yt.sum()), 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=[10, 20, 30, 40, 60])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"runs/label_budget_trunk_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    gg_keep = {s: gg[s] for s in set(slides[keep]) if s in gg}
    sites = site_data(args.encoder)
    tab, feats, ymap = sites["PAR"]

    rows_all = []
    for mode, lam in (("tile_only", 0.0), ("tile_plus_slide", 0.5)):
        for seed in args.seeds:
            params = {"epochs": 10, "batch": 256, "lam": lam}
            model = train(mode, emb[keep], labels[keep], slides[keep],
                          gg_keep, params, seed=seed)
            model.eval()
            sf = trunk_slide_feats(model, feats, dev)
            keys = tab.slide.astype(str).str.lower().str.replace(
                "_leica", "", regex=False)
            X = np.stack([sf[k] for k in keys])
            y = tab.y.to_numpy()
            df = simulate(X, y, args.Ks, args.alpha, args.repeats, rng)
            df["mode"] = mode
            df["seed"] = seed
            rows_all.append(df)
            ok = df[df.feasible]
            print(f"{mode} seed{seed}: feasible {len(ok)}/{len(df)}"
                  + (f" cleared={ok.cleared.mean():.3f}"
                     f" missed={ok.missed_per100.mean():.2f}"
                     if len(ok) else ""), flush=True)

    res = pd.concat(rows_all, ignore_index=True)
    res.to_csv(out / "simulation.csv", index=False)
    summ = []
    for (mode, K), g in res.groupby(["mode", "K"]):
        ok = g[g.feasible]
        summ.append({"mode": mode, "K": K,
                     "n": len(g),
                     "feasible": float(g.feasible.mean()),
                     "cleared": float(ok.cleared.mean()) if len(ok) else np.nan,
                     "missed": float(ok.missed_per100.mean()) if len(ok)
                     else np.nan,
                     "fn_clear": float(ok.fn_clear.mean()) if len(ok)
                     else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
