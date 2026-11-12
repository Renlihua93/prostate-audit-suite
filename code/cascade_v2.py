#!/usr/bin/env python3
"""Cascade v2: benign gate + low-grade gate, both refit on K site labels.

Scores (both LR on mean tile embeddings, fit on the K labelled slides):
  p_benign  = P(ISUP == 0)
  p_grade   = P(ISUP >= 2)

Clearing rule:  clear if p_grade <= tau_g  OR  p_benign >= 1 - tau_b.
Thresholds chosen on the calibration set to maximise cleared count subject
to the Clopper-Pearson upper bound on the union FN rate <= alpha.
Compared with the single-score rule (p_grade only) at the same alpha.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_utility import cp_upper, site_data, tau_one_sided

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def fit_scores(X, y_grade, y_benign, cal):
    sc = StandardScaler().fit(X[cal])
    out = []
    for yy in (y_grade, y_benign):
        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight="balanced")
        lr.fit(sc.transform(X[cal]), yy[cal])
        out.append(lr.predict_proba(sc.transform(X))[:, 1])
    return out  # p_grade, p_benign


def choose_cascade(pg, pb, y, alpha, delta=0.05, min_acc=5,
                   qs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    best = None
    for qg in qs:
        tg = float(np.quantile(pg, qg))
        for qb in qs:
            tb = float(np.quantile(pb, 1 - qb))  # high-benign quantile
            clear = (pg <= tg) | (pb >= tb)
            n = int(clear.sum())
            if n < min_acc:
                continue
            fn = int((clear & (y == 1)).sum())
            if cp_upper(fn, n, delta) <= alpha:
                if best is None or n > best[0]:
                    best = (n, tg, tb)
    return None if best is None else (best[1], best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=[30, 45, 60, 100])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"runs/cascade_v2_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    sites = site_data(args.encoder)
    tab, feats, ymap = sites["PAR"]
    keys = tab.slide.astype(str).str.lower().apply(NORM)
    X = np.stack([feats[k].mean(0) for k in keys])
    y = tab.y.to_numpy()
    sev = tab.maj.to_numpy()
    y_benign = (sev == 0).astype(int)
    n = len(y)

    rows = []
    for K in args.Ks:
        for alpha in args.alphas:
            for rep in range(args.repeats):
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                cal = np.concatenate([
                    rng.choice(pos, min(K // 2, len(pos)), replace=False),
                    rng.choice(neg, min(K - K // 2, len(neg)), replace=False)])
                test = np.setdiff1d(np.arange(n), cal)
                if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                    continue
                if len(np.unique(y_benign[cal])) < 2:
                    continue
                pg, pb = fit_scores(X, y, y_benign, cal)
                yt = y[test]
                row = {"K": K, "alpha": alpha, "rep": rep}
                tau = tau_one_sided(pg[cal], y[cal], alpha)
                if tau is not None:
                    clear = pg[test] <= tau
                    fn = int((clear & (yt == 1)).sum())
                    row.update({"single_feasible": True,
                                "single_cleared": float(clear.mean()),
                                "single_missed": 100 * fn / len(yt),
                                "single_fn_clear": fn / max(int(clear.sum()), 1)})
                else:
                    row["single_feasible"] = False
                th = choose_cascade(pg[cal], pb[cal], y[cal], alpha)
                if th is not None:
                    tg, tb = th
                    clear = (pg[test] <= tg) | (pb[test] >= tb)
                    fn = int((clear & (yt == 1)).sum())
                    row.update({"casc_feasible": True,
                                "casc_cleared": float(clear.mean()),
                                "casc_missed": 100 * fn / len(yt),
                                "casc_fn_clear": fn / max(int(clear.sum()), 1)})
                else:
                    row["casc_feasible"] = False
                rows.append(row)

    res = pd.DataFrame(rows)
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
            "single_missed": round(float(s_ok.single_missed.mean()), 2)
            if len(s_ok) else np.nan,
            "casc_feas": round(float(g.casc_feasible.mean()), 3),
            "casc_cleared": round(float(c_ok.casc_cleared.mean()), 3)
            if len(c_ok) else np.nan,
            "casc_missed": round(float(c_ok.casc_missed.mean()), 2)
            if len(c_ok) else np.nan,
            "casc_fn_clear": round(float(c_ok.casc_fn_clear.mean()), 3)
            if len(c_ok) else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
