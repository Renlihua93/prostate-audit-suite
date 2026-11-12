#!/usr/bin/env python3
"""Sensitivity of the deployment operating point to cohort prevalence and to
the reader-label rule (first reader vs median of readers)."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_utility import site_data, tau_one_sided

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/prevalence_label_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def simulate(X, y, K=60, alpha=0.10, reps=100, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for _ in range(reps):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos) < 5 or len(neg) < 5:
            continue
        cal = np.concatenate([rng.choice(pos, min(30, len(pos)), replace=False),
                              rng.choice(neg, min(30, len(neg)), replace=False)])
        test = np.setdiff1d(np.arange(n), cal)
        if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
            continue
        sc = StandardScaler().fit(X[cal])
        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight="balanced").fit(
            sc.transform(X[cal]), y[cal])
        p = lr.predict_proba(sc.transform(X))[:, 1]
        tau = tau_one_sided(p[cal], y[cal], alpha)
        row = {"feasible": tau is not None}
        if tau is not None:
            clear = p[test] <= tau
            yt = y[test]
            fn = int((clear & (yt == 1)).sum())
            row.update({"cleared": float(clear.mean()),
                        "fn_clear": fn / max(int(clear.sum()), 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def resample_prevalence(X, y, target, n, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    npos = int(round(n * target))
    nneg = n - npos
    idx = np.concatenate([rng.choice(pos, npos, replace=True),
                          rng.choice(neg, nneg, replace=True)])
    return X[idx], y[idx]


def main():
    site = site_data("phikon")["PAR"]
    tab, feats, _ = site
    keys = tab.slide.astype(str).str.lower().apply(NORM)
    X = np.stack([feats[k].mean(0) for k in keys])
    y_maj = tab.y.to_numpy()

    # reader-1 labels
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide_l"] = lab.slide_id.astype(str).str.lower().apply(NORM)
    r1 = lab.dropna(subset=["isup_1"]).set_index("slide_l").isup_1
    y_r1 = np.array([int(r1.get(s, -1)) for s in keys])
    ok = y_r1 >= 0
    print(f"reader-1 labels available for {ok.sum()}/{len(keys)}", flush=True)

    rows = []
    for tag, X_, y_ in (("majority", X, y_maj),
                        ("reader1", X[ok], (y_r1[ok] >= 2).astype(int))):
        d = simulate(X_, y_)
        if d.empty:
            d = pd.DataFrame(columns=["feasible", "cleared", "fn_clear"])
        okdf = d[d.feasible]
        rows.append({"analysis": "label_rule", "setting": tag,
                     "feasible": float(d.feasible.mean()),
                     "cleared": float(okdf.cleared.mean()) if len(okdf) else np.nan,
                     "fn_clear": float(okdf.fn_clear.mean()) if len(okdf) else np.nan,
                     "n": len(d)})
    rng = np.random.default_rng(0)
    for target in (0.10, 0.20, 0.30, 0.40, 0.50):
        Xr, yr = resample_prevalence(X, y_maj, target, 1000, rng)
        d = simulate(Xr, yr)
        if d.empty:
            d = pd.DataFrame(columns=["feasible", "cleared", "fn_clear"])
        okdf = d[d.feasible]
        rows.append({"analysis": "prevalence", "setting": f"p={target:.2f}",
                     "feasible": float(d.feasible.mean()),
                     "cleared": float(okdf.cleared.mean()) if len(okdf) else np.nan,
                     "fn_clear": float(okdf.fn_clear.mean()) if len(okdf) else np.nan,
                     "n": len(d)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
