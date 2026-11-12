#!/usr/bin/env python3
"""Recalibration of the site-adapted slide probe.

Per repeated split (K calibration / rest test):
  - fit the few-shot probe on the calibration set
  - fit temperature scaling (1 param) and isotonic regression on the
    calibration set's cross-fitted probabilities
  - evaluate on the test set: AUC (unchanged), calibration slope,
    intercept, Brier, and reliability; compare raw vs temperature vs isotonic
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from clinical_utility import site_data
from sensitivity_analysis import build

ROOT = Path(__file__).resolve().parents[1]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_model(X, y):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(X), y)
    return sc, lr


def predict(sc, lr, X):
    return lr.predict_proba(sc.transform(X))[:, 1]


def crossfit_cal(sc, lr, X, y):
    """Out-of-fold probabilities inside the calibration set."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        s2, m2 = fit_model(X[tr], y[tr])
        oof[va] = predict(s2, m2, X[va])
    return oof


def fit_temperature(p, y):
    def nll(T):
        q = 1 / (1 + np.exp(-logit(p) / T))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    r = minimize_scalar(nll, bounds=(0.05, 20), method="bounded")
    return float(r.x)


def apply_temperature(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def cal_metrics(p, y):
    out = {"auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p))}
    lg = logit(p)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y)
    out["slope"] = float(lr.coef_[0, 0])
    out["intercept"] = float(lr.intercept_[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"runs/recalibration_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    sites = build(site_data(args.encoder))
    site = sites["PAR"]
    X, y = site["X"], site["y"]
    n = len(y)

    rows, acc = [], {"raw": [], "temperature": [], "isotonic": []}
    for rep in range(args.reps):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cal = np.concatenate([rng.choice(pos, args.K // 2, replace=False),
                              rng.choice(neg, args.K - args.K // 2,
                                         replace=False)])
        test = np.setdiff1d(np.arange(n), cal)
        sc, lr = fit_model(X[cal], y[cal])
        oof = crossfit_cal(sc, lr, X[cal], y[cal])
        T = fit_temperature(oof, y[cal])
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y[cal])

        p_te = predict(sc, lr, X[test])
        y_te = y[test]
        variants = {"raw": p_te,
                    "temperature": apply_temperature(p_te, T),
                    "isotonic": iso.predict(p_te)}
        for k, p in variants.items():
            m = cal_metrics(p, y_te)
            m.update({"rep": rep, "T": T})
            acc[k].append(m)
    for k, v in acc.items():
        df = pd.DataFrame(v)
        df.to_csv(out / f"metrics_{k}.csv", index=False)
        print(k, {c: round(float(df[c].mean()), 3)
                  for c in ("auc", "slope", "intercept", "brier")})
    print("mean temperature:", round(float(np.mean(
        [m["T"] for m in acc["raw"]])), 2))


if __name__ == "__main__":
    main()
