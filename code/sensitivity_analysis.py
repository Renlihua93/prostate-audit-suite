#!/usr/bin/env python3
"""Site-size feasibility, calibration, decision curves, subgroup sensitivity.

A. Site-size analysis: bootstrap sites of size n_site (from the expanded PAR /
   TCGA slides), calibrate a one-sided conformal threshold at alpha on a
   fraction of the site, and report the probability that a valid threshold
   exists, plus clearing/miss rates when it does.
B. Calibration of the refit slide probe (reliability, slope, Brier).
C. Decision curves (AI-only biopsy rule) vs read-all / treat-all.
D. Subgroups: clearing and FN by reference severity (benign / ISUP1 / >=2).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from clinical_utility import net_benefit, site_data, tau_one_sided

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def build(sites):
    out = {}
    for name, (tab, feats, ymap) in sites.items():
        tab = tab.copy()
        keys = tab.slide.astype(str).str.lower().apply(NORM)
        X = np.stack([feats[k].mean(0) for k in keys])
        out[name] = {"tab": tab.reset_index(drop=True), "X": X,
                     "y": tab.y.to_numpy(),
                     "sev": (tab.maj.to_numpy() if name == "PAR"
                             else tab.GG.astype(int).to_numpy()),
                     "dis": (tab.disagree.to_numpy()
                             if "disagree" in tab
                             else np.array([np.nan] * len(tab)))}
    return out


def fit_refit(X, y, cal, **kw):
    if len(np.unique(y[cal])) < 2:
        return None, None
    sc = StandardScaler().fit(X[cal])
    lr = LogisticRegression(C=kw.get("C", 0.1), max_iter=1000,
                            class_weight="balanced")
    lr.fit(sc.transform(X[cal]), y[cal])
    return lr, sc


def site_size(site, sizes, frac, alpha, reps, rng):
    X, y = site["X"], site["y"]
    n = len(y)
    rows = []
    for ns in sizes:
        for rep in range(reps):
            idx = rng.choice(n, ns, replace=ns > n)
            pos = idx[y[idx] == 1]
            neg = idx[y[idx] == 0]
            K = int(round(ns * frac))
            kpos = min(K // 2, len(pos))
            kneg = min(K - kpos, len(neg))
            cal = np.concatenate([rng.choice(pos, kpos, replace=False),
                                  rng.choice(neg, kneg, replace=False)])
            test = np.setdiff1d(idx, cal)
            if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                continue
            lr, sc = fit_refit(X, y, cal)
            if lr is None:
                continue
            p = lr.predict_proba(sc.transform(X))[:, 1]
            tau = tau_one_sided(p[cal], y[cal], alpha)
            row = {"n_site": ns, "frac": frac, "alpha": alpha, "rep": rep,
                   "feasible": tau is not None}
            if tau is not None:
                clear = p[test] <= tau
                yt = y[test]
                fn = int((clear & (yt == 1)).sum())
                row.update({"cleared": float(clear.mean()),
                            "fn_clear": fn / max(int(clear.sum()), 1),
                            "missed_per100": 100 * fn / len(yt),
                            "sens": 1 - fn / max(int(yt.sum()), 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def calibration(site, K, reps, rng, C=0.1):
    X, y = site["X"], site["y"]
    n = len(y)
    preds, obs, aucs = [], [], []
    for rep in range(reps):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cal = np.concatenate([rng.choice(pos, K // 2, replace=False),
                              rng.choice(neg, K - K // 2, replace=False)])
        test = np.setdiff1d(np.arange(n), cal)
        if len(np.unique(y[test])) < 2:
            continue
        lr, sc = fit_refit(X, y, cal, C=C)
        if lr is None:
            continue
        p = lr.predict_proba(sc.transform(X))[test, 1]
        preds.append(p)
        obs.append(y[test])
        aucs.append(roc_auc_score(y[test], p))
    if not preds:
        return {}
    p = np.concatenate(preds)
    o = np.concatenate(obs)
    from sklearn.metrics import log_loss
    slope = 1.0
    eps = 1e-6
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr_cal = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], o)
    slope = float(lr_cal.coef_[0, 0])
    bins = pd.qcut(p, 10, duplicates="drop")
    rel = pd.DataFrame({"p": p, "o": o, "bin": bins}).groupby(
        "bin", observed=True).agg(pred=("p", "mean"), obs=("o", "mean"),
                                  n=("o", "size"))
    return {"auc": float(np.mean(aucs)), "slope": slope,
            "intercept": float(lr_cal.intercept_[0]),
            "brier": float(brier_score_loss(o, p)),
            "reliability": rel.reset_index(drop=True).to_dict("records")}


def dca(site, K, alpha, reps, rng, pts):
    X, y = site["X"], site["y"]
    n = len(y)
    rows = []
    for rep in range(reps):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cal = np.concatenate([rng.choice(pos, K // 2, replace=False),
                              rng.choice(neg, K - K // 2, replace=False)])
        test = np.setdiff1d(np.arange(n), cal)
        if len(np.unique(y[test])) < 2:
            continue
        lr, sc = fit_refit(X, y, cal)
        p = lr.predict_proba(sc.transform(X))[test, 1]
        yt = y[test]
        prev = float(yt.mean())
        for pt in pts:
            odds = pt / (1 - pt)
            rows.append({"rep": rep, "pt": pt,
                         "ai_only": net_benefit(yt, (p >= pt).astype(int), pt),
                         "read_all": prev,
                         "treat_all": prev - (1 - prev) * odds})
    return pd.DataFrame(rows)


def subgroups(site, K, alpha, reps, rng):
    X, y = site["X"], site["y"]
    sev = site["sev"]
    dis = site["dis"]
    n = len(y)
    rows = []
    for rep in range(reps):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cal = np.concatenate([rng.choice(pos, K // 2, replace=False),
                              rng.choice(neg, K - K // 2, replace=False)])
        test = np.setdiff1d(np.arange(n), cal)
        lr, sc = fit_refit(X, y, cal)
        if lr is None:
            continue
        p_all = lr.predict_proba(sc.transform(X))[:, 1]
        tau = tau_one_sided(p_all[cal], y[cal], alpha)
        if tau is None:
            continue
        p = p_all[test]
        clear = p <= tau
        for grp, mask in (("benign/ISUP0", sev[test] == 0),
                          ("ISUP1", sev[test] == 1),
                          ("ISUP>=2", sev[test] >= 2)):
            if mask.sum() == 0:
                continue
            rows.append({"rep": rep, "group": grp,
                         "n": int(mask.sum()),
                         "cleared": float(clear[mask].mean())})
        if np.any(dis[test] == True):  # noqa: E712
            m = np.asarray(dis[test] == True)  # noqa: E712
            rows.append({"rep": rep, "group": "readers_disagree",
                         "n": int(m.sum()),
                         "cleared": float(clear[m].mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[120, 160, 200, 250])
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"runs/sensitivity_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    sites = build(site_data(args.encoder))
    for name, site in sites.items():
        print(f"=== {name}: n={len(site['y'])} pos={int(site['y'].sum())} "
              f"neg={int((site['y'] == 0).sum())}", flush=True)
        if name != "PAR":
            print(f"[skip] {name}: too few negatives for conformal "
                  f"sensitivity analysis", flush=True)
            continue
        for alpha in args.alphas:
            r = site_size(site, args.sizes, args.frac, alpha,
                          args.reps, rng)
            for col in ("cleared", "fn_clear", "missed_per100", "sens"):
                if col not in r:
                    r[col] = np.nan
            r["site"] = name
            r.to_csv(out / f"site_size_{name}_a{alpha}.csv", index=False)
            g = r.groupby("n_site").agg(
                feasible=("feasible", "mean"),
                cleared=("cleared", "mean"),
                fn_clear=("fn_clear", "mean"),
                missed=("missed_per100", "mean")).round(3)
            print(f"alpha={alpha} site-size feasibility:\n{g.to_string()}",
                  flush=True)
        cal = calibration(site, args.K, 50, rng)
        print(f"calibration (K={args.K}): "
              f"auc={cal.get('auc', float('nan')):.3f} "
              f"slope={cal.get('slope', float('nan')):.2f} "
              f"brier={cal.get('brier', float('nan')):.3f}", flush=True)
        (out / f"calibration_{name}.json").write_text(
            json.dumps(cal, indent=2, default=float))
        dc = dca(site, args.K, 0.10, 50, rng, [0.1, 0.2, 0.3, 0.4, 0.5])
        dc.to_csv(out / f"dca_{name}.csv", index=False)
        sub = subgroups(site, args.K, 0.10, 50, rng)
        sub.to_csv(out / f"subgroups_{name}.csv", index=False)
        if len(sub):
            print(sub.groupby("group").cleared.agg(["mean", "count"])
                  .round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
