#!/usr/bin/env python3
"""Decision-targeted active labelling for risk-certified AS triage.

Sequential site labelling: start from 10 random labels, then add 5 slides
per round chosen by
  random       - uniform over the remaining pool
  uncertainty  - |p - 0.5|
  boundary     - |p - tau*| where tau* is the current conformal threshold
                 (i.e., slides whose label most affects the clearing rule)
  risky_tail   - p in the lowest quantile band (candidates for clearing)
At each budget K, refit the probe and record whether a valid one-sided
conformal threshold exists at alpha, plus clearing/FN on the remaining pool.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_utility import site_data, tau_one_sided

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def fit_p(X, y, lab, C=0.1):
    if len(np.unique(y[lab])) < 2:
        return None
    sc = StandardScaler().fit(X[lab])
    lr = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(X[lab]), y[lab])
    return lr.predict_proba(sc.transform(X))[:, 1]


def run(site, Ks, alpha, seeds, rng):
    X, y = site["X"], site["y"]
    n = len(y)
    rows = []
    for seed in seeds:
        for rep in range(20):
            start = rng.choice(n, 10, replace=False)
            if len(np.unique(y[start])) < 2:
                continue
            lab = list(start)
            pool = np.setdiff1d(np.arange(n), start)
            for K in Ks:
                while len(lab) < K and len(pool):
                    p = fit_p(X, y, np.array(lab))
                    if p is None:
                        break
                    if K > 60:
                        pass
                    # choose strategy externally via arg label
                    break
                lab_arr = np.array(lab)
                p = fit_p(X, y, lab_arr)
                if p is None:
                    continue
                tau = tau_one_sided(p[lab_arr], y[lab_arr], alpha)
                test = np.setdiff1d(np.arange(n), lab_arr)
                row = {"K": K, "alpha": alpha, "rep": rep, "seed": seed,
                       "feasible": tau is not None}
                if tau is not None:
                    clear = p[test] <= tau
                    fn = int((clear & (y[test] == 1)).sum())
                    row.update({"cleared": float(clear.mean()),
                                "missed": 100 * fn / len(test),
                                "fn_clear": fn / max(int(clear.sum()), 1)})
                rows.append(row)
                # extend by one round using a placeholder (overwritten in main)
                break
    return pd.DataFrame(rows)


def sequential(X, y, Ks, alpha, strategy):
    n = len(y)
    rng = np.random.default_rng(0)
    out = []
    for rep in range(20):
        lab = list(rng.choice(n, 10, replace=False))
        if len(np.unique(y[lab])) < 2:
            continue
        pool = list(np.setdiff1d(np.arange(n), lab))
        for K in Ks:
            while len(lab) < K and pool:
                p = fit_p(X, y, np.array(lab))
                if p is None:
                    break
                pts = np.array(pool)
                if strategy == "random":
                    pick = rng.choice(len(pool))
                elif strategy == "uncertainty":
                    pick = int(np.argmin(np.abs(p[pts] - 0.5)))
                elif strategy == "boundary":
                    tau = tau_one_sided(p[np.array(lab)], y[np.array(lab)],
                                        alpha)
                    tgt = tau if tau is not None else np.quantile(p, 0.3)
                    pick = int(np.argmin(np.abs(p[pts] - tgt)))
                elif strategy == "risky_tail":
                    pick = int(np.argmin(p[pts]))
                else:
                    raise ValueError(strategy)
                lab.append(pool.pop(pick))
            lab_arr = np.array(lab)
            p = fit_p(X, y, lab_arr)
            if p is None:
                continue
            tau = tau_one_sided(p[lab_arr], y[lab_arr], alpha)
            test = np.setdiff1d(np.arange(n), lab_arr)
            row = {"K": K, "alpha": alpha, "rep": rep, "strategy": strategy,
                   "feasible": tau is not None}
            if tau is not None:
                clear = p[test] <= tau
                fn = int((clear & (y[test] == 1)).sum())
                row.update({"cleared": float(clear.mean()),
                            "missed": 100 * fn / len(test),
                            "fn_clear": fn / max(int(clear.sum()), 1)})
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--Ks", nargs="+", type=int, default=[20, 30, 40, 60, 80])
    args = ap.parse_args()
    outdir = ROOT / f"runs/active_acq_{args.encoder}"
    outdir.mkdir(parents=True, exist_ok=True)
    sites = site_data(args.encoder)
    tab, feats, ymap = sites["PAR"]
    keys = tab.slide.astype(str).str.lower().apply(NORM)
    X = np.stack([feats[k].mean(0) for k in keys])
    site = {"X": X, "y": tab.y.to_numpy()}

    rows = []
    for alpha in args.alphas:
        for strat in ("random", "uncertainty", "boundary", "risky_tail"):
            df = sequential(X, site["y"], args.Ks, alpha, strat)
            rows.append(df)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(outdir / "simulation.csv", index=False)
    summ = res.groupby(["alpha", "strategy", "K"]).agg(
        feasible=("feasible", "mean"), cleared=("cleared", "mean"),
        missed=("missed", "mean"), fn_clear=("fn_clear", "mean"),
        n=("rep", "count")).round(3)
    summ.to_csv(outdir / "summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
