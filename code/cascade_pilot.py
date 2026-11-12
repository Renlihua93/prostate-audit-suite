#!/usr/bin/env python3
"""Cascade conformal triage: detection score gates, grading score clears.

Rule: clear a slide if  s_cancer <= tau_c  OR  s_grade <= tau_g.
Thresholds are chosen on K site labels by grid search over score quantiles,
maximising the number of cleared calibration slides subject to the
Clopper-Pearson upper bound on the false-negative rate of the union <= alpha.

Compared with the single-score rule (grade only) at the same alpha.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clinical_utility import cp_upper, score_slides, site_data, tau_one_sided
from mgod_pilot import load_sicap, train

ROOT = Path(__file__).resolve().parents[1]


def choose_cascade(sg, sca, y, alpha, delta=0.05, min_acc=5,
                   qs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    best = None
    for qc in qs:
        tau_c = float(np.quantile(sca, qc))
        for qg in qs:
            tau_g = float(np.quantile(sg, qg))
            clear = (sca <= tau_c) | (sg <= tau_g)
            n = int(clear.sum())
            if n < min_acc:
                continue
            fn = int((clear & (y == 1)).sum())
            if cp_upper(fn, n, delta) <= alpha:
                if best is None or n > best[0]:
                    best = (n, tau_c, tau_g)
    return None if best is None else (best[1], best[2])


def simulate(sg, sca, y, Ks, alphas, repeats, rng):
    n = len(y)
    rows = []
    for K in Ks:
        for alpha in alphas:
            for rep in range(repeats):
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                cal = np.concatenate([
                    rng.choice(pos, min(K // 2, len(pos)), replace=False),
                    rng.choice(neg, min(K - K // 2, len(neg)), replace=False)])
                test = np.setdiff1d(np.arange(n), cal)
                if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                    continue
                yt = y[test]
                row = {"K": K, "alpha": alpha, "rep": rep, "n_test": len(yt)}
                # single score (grade)
                tau = tau_one_sided(sg[cal], y[cal], alpha)
                if tau is not None:
                    clear = sg[test] <= tau
                    fn = int((clear & (yt == 1)).sum())
                    row.update({"single_feasible": True,
                                "single_cleared": float(clear.mean()),
                                "single_missed": 100 * fn / len(yt),
                                "single_fn_clear": fn / max(int(clear.sum()), 1)})
                else:
                    row["single_feasible"] = False
                # cascade
                th = choose_cascade(sg[cal], sca[cal], y[cal], alpha)
                if th is not None:
                    tau_c, tau_g = th
                    clear = (sca[test] <= tau_c) | (sg[test] <= tau_g)
                    fn = int((clear & (yt == 1)).sum())
                    row.update({"casc_feasible": True,
                                "casc_cleared": float(clear.mean()),
                                "casc_missed": 100 * fn / len(yt),
                                "casc_fn_clear": fn / max(int(clear.sum()), 1)})
                else:
                    row["casc_feasible"] = False
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=[30, 45, 60, 100])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"runs/cascade_pilot_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    gg_keep = {s: gg[s] for s in set(slides[keep]) if s in gg}
    sites = site_data(args.encoder)
    rows_all = []
    for seed in args.seeds:
        params = {"epochs": 10, "batch": 256, "lam": 0.5}
        model = train("tile_plus_slide", emb[keep], labels[keep],
                      slides[keep], gg_keep, params, seed=seed)
        model.eval()
        for name, (tab, feats, ymap) in sites.items():
            sc = score_slides(model, feats, list(ymap), dev)
            tab = tab.copy()
            tab["sg"] = tab.slide.str.lower().map(lambda k: sc[k][0])
            tab["sca"] = tab.slide.str.lower().map(lambda k: sc[k][1])
            tab = tab.dropna(subset=["sg", "sca"])
            if name != "PAR":
                print(f"[skip] {name}: too few negatives for certification",
                      flush=True)
                continue
            df = simulate(tab.sg.to_numpy(), tab.sca.to_numpy(),
                          tab.y.to_numpy(), args.Ks, args.alphas,
                          args.repeats, rng)
            df["site"] = name
            df["seed"] = seed
            rows_all.append(df)
            print(f"seed{seed} scored {len(tab)} slides", flush=True)

    res = pd.concat(rows_all, ignore_index=True)
    res.to_csv(out / "simulation.csv", index=False)
    summ = []
    for (alpha, K), g in res.groupby(["alpha", "K"]):
        s_ok = g[g.single_feasible]
        c_ok = g[g.casc_feasible]
        summ.append({
            "alpha": alpha, "K": K, "n": len(g),
            "single_feas": round(float(g.single_feasible.mean()), 3),
            "single_cleared": round(float(s_ok.single_cleared.mean()), 3)
            if len(s_ok) else np.nan,
            "single_fn_clear": round(float(s_ok.single_fn_clear.mean()), 3)
            if len(s_ok) else np.nan,
            "casc_feas": round(float(g.casc_feasible.mean()), 3),
            "casc_cleared": round(float(c_ok.casc_cleared.mean()), 3)
            if len(c_ok) else np.nan,
            "casc_fn_clear": round(float(c_ok.casc_fn_clear.mean()), 3)
            if len(c_ok) else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
