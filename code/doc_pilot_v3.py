#!/usr/bin/env python3
"""DOC v3: two-sided risk-controlled triage with sequential labelling.

Accept slides as low-risk if score <= tau_low and as high-risk if
score >= tau_high, controlling BOTH tails with Clopper-Pearson upper bounds
on accepted error at confidence 1-delta. Maximise accepted fraction subject
to both bounds <= alpha. Refit the slide probe after each labelling budget.

Sites: PAR (majority of 2-3 readers), TCGA (refined GG).
Selection: random (bootstrap repeats) / uncertainty / diversity.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from doc_pilot_v2 import site_tables, slide_features

ROOT = Path(__file__).resolve().parents[1]


def cp_upper(errors, n, delta):
    if n == 0:
        return 1.0
    if errors == n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def choose_thresholds(scores, y, alpha, delta, min_acc=5):
    order = np.argsort(scores)
    s, yy = scores[order], y[order]
    n = len(s)
    best = None
    # prefix: low-risk accepted; suffix: high-risk accepted
    for i in range(0, n + 1):
        nlo = i
        err_lo = int(yy[:i].sum())  # true positive among low-risk = FN
        if nlo and nlo < min_acc:
            continue
        if nlo and cp_upper(err_lo, nlo, delta) > alpha:
            continue
        # largest suffix satisfying the high-risk bound
        for j in range(n - i, 0, -1):
            nhi = j
            err_hi = int((1 - yy[n - j:]).sum())  # true negative = FP
            if nhi < min_acc:
                break
            if cp_upper(err_hi, nhi, delta) <= alpha:
                if best is None or nlo + nhi > best[0]:
                    lo = s[i - 1] if i else -np.inf
                    hi = s[n - j] if j < n else np.inf
                    best = (nlo + nhi, lo, hi)
                break
    if best is None:
        return None
    return best[1], best[2]


def run_site(name, tab, feats_dir, Ks, strategies, alpha, delta, repeats):
    feats = slide_features(feats_dir)
    tab = tab.copy()
    tab["slide_l"] = tab.slide.astype(str).str.lower()
    tab = tab[tab.slide_l.isin(feats)].reset_index(drop=True)
    X = np.stack([feats[s] for s in tab.slide_l])
    y = tab.y.to_numpy()
    dis = tab.readers_disagree.to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for K in Ks:
        for strat in strategies:
            for rep in range(repeats if strat == "random" else 1):
                if strat == "random":
                    idx = rng.choice(len(y), K, replace=False)
                else:
                    sc0 = StandardScaler().fit(X)
                    lr0 = LogisticRegression(C=0.1, max_iter=800,
                                             class_weight="balanced")
                    lr0.fit(sc0.transform(X), y)
                    p0 = lr0.predict_proba(sc0.transform(X))[:, 1]
                    if strat == "uncertainty":
                        idx = np.argsort(np.abs(p0 - 0.5))[:K]
                    else:
                        sel = [int(np.argmin(np.abs(p0 - 0.5)))]
                        while len(sel) < K:
                            d = np.min(np.abs(p0[:, None]
                                              - p0[sel][None, :]), axis=1)
                            sel.append(int(np.argmax(d)))
                        idx = np.array(sel)
                if len(np.unique(y[idx])) < 2:
                    continue
                sc = StandardScaler().fit(X[idx])
                lr = LogisticRegression(C=0.1, max_iter=800,
                                        class_weight="balanced")
                lr.fit(sc.transform(X[idx]), y[idx])
                s_tr = lr.predict_proba(sc.transform(X[idx]))[:, 1]
                th = choose_thresholds(s_tr, y[idx], alpha, delta)
                if th is None:
                    rows.append({"site": name, "K": K, "strategy": strat,
                                 "accepted": 0.0, "deferral": 1.0,
                                 "acc_err_lo": np.nan, "acc_err_hi": np.nan,
                                 "disagree_deferred": np.nan})
                    continue
                lo, hi = th
                p = lr.predict_proba(sc.transform(X))[:, 1]
                acc_lo = p <= lo
                acc_hi = p >= hi
                acc = acc_lo | acc_hi
                rows.append({
                    "site": name, "K": K, "strategy": strat,
                    "accepted": float(acc.mean()),
                    "deferral": float(1 - acc.mean()),
                    "acc_err_lo": (float(y[acc_lo].mean())
                                   if acc_lo.sum() else np.nan),
                    "acc_err_hi": (float(1 - y[acc_hi].mean())
                                   if acc_hi.sum() else np.nan),
                    "disagree_deferred": (float((~acc[np.asarray(
                        dis == True)]).mean())  # noqa: E712
                        if name == "PAR" else np.nan),
                    "n_acc": int(acc.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/doc_pilot")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    Ks = [20, 30, 40, 60, 80]
    out = []
    for alpha in args.alphas:
        for name, (tab, fd) in site_tables().items():
            res = run_site(name, tab, fd, Ks,
                           ["random", "uncertainty", "diversity"],
                           alpha, args.delta, args.repeats)
            res["alpha"] = alpha
            out.append(res)
            piv = res.groupby(["strategy", "K"]).agg(
                accepted=("accepted", "mean"),
                acc_err_lo=("acc_err_lo", "mean"),
                acc_err_hi=("acc_err_hi", "mean"),
                disagree_deferred=("disagree_deferred", "mean")).round(3)
            print(f"\n=== {name} alpha={alpha} ===")
            print(piv.to_string())
    pd.concat(out).to_csv(args.out / "doc_pilot_v3.csv", index=False)


if __name__ == "__main__":
    main()
