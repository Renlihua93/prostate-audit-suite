#!/usr/bin/env python3
"""Valid active-labelling test: random audit set for conformal calibration.

Total budget K is split into:
  audit  = K/2 slides, sampled uniformly (used ONLY for conformal calibration)
  fit    = K/2 slides, sampled by
             random        - uniform from the remaining pool
             uncertainty   - closest to p=0.5 under a pilot model
The probe is fit on the fit set; the conformal threshold is calibrated on the
audit set (exchangeable with the held-out test pool); evaluation is on the
remaining slides. This isolates the effect of decision-targeted labelling
without the selection bias of calibrating on the selected labels.
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


def fit_p(X, y, idx):
    if len(np.unique(y[idx])) < 2:
        return None
    sc = StandardScaler().fit(X[idx])
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(X[idx]), y[idx])
    return lr.predict_proba(sc.transform(X))[:, 1]


def run(site, Ks, alpha, reps, rng):
    X, y = site["X"], site["y"]
    n = len(y)
    rows = []
    for K in Ks:
        half = K // 2
        for rep in range(reps):
            audit = rng.choice(n, half, replace=False)
            pool = np.setdiff1d(np.arange(n), audit)
            if len(np.unique(y[audit])) < 2:
                continue
            # pilot model on audit only, used for uncertainty ranking of pool
            p_pilot = fit_p(X, y, audit)
            if p_pilot is None:
                continue
            for arm in ("random_fit", "uncertainty_fit"):
                if arm == "random_fit":
                    fit = rng.choice(pool, half, replace=False)
                else:
                    unc = np.abs(p_pilot[pool] - 0.5)
                    fit = pool[np.argsort(unc)[:half]]
                p = fit_p(X, y, fit)
                if p is None:
                    continue
                tau = tau_one_sided(p[audit], y[audit], alpha)
                test = np.setdiff1d(pool, fit)
                row = {"K": K, "alpha": alpha, "rep": rep, "arm": arm,
                       "feasible": tau is not None, "n_test": len(test)}
                if tau is not None:
                    clear = p[test] <= tau
                    yt = y[test]
                    fn = int((clear & (yt == 1)).sum())
                    row.update({"cleared": float(clear.mean()),
                                "missed": 100 * fn / len(yt),
                                "fn_clear": fn / max(int(clear.sum()), 1),
                                "sens": 1 - fn / max(int(yt.sum()), 1)})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--Ks", nargs="+", type=int, default=[80, 120])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.10])
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    out = ROOT / f"runs/audit_active_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sites = site_data(args.encoder)
    tab, feats, ymap = sites["PAR"]
    keys = tab.slide.astype(str).str.lower().apply(NORM)
    site = {"X": np.stack([feats[k].mean(0) for k in keys]),
            "y": tab.y.to_numpy()}
    print(f"PAR n={len(site['y'])} pos={int(site['y'].sum())}", flush=True)
    rows = []
    for alpha in args.alphas:
        rows.append(run(site, args.Ks, alpha, args.reps, rng))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "simulation.csv", index=False)
    summ = res.groupby(["alpha", "K", "arm"]).agg(
        feasible=("feasible", "mean"), cleared=("cleared", "mean"),
        missed=("missed", "mean"), fn_clear=("fn_clear", "mean"),
        sens=("sens", "mean"), n=("rep", "count")).round(3)
    summ.to_csv(out / "summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
