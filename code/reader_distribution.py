#!/usr/bin/env python3
"""Reader-distribution grading: predict the pathologist grade distribution.

Training data: each available reader grade for a slide is a row (multi-rater
augmentation), so a classifier's probabilities approximate the reader
distribution. Evaluated with patient-grouped CV against:
  - observed reader disagreement (nunique > 1)
  - entropy of the empirical reader histogram
  - boundary-mass deferral capture (P(ISUP>=2) near 0.5)
Compared with a majority-label model.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from clinical_utility import site_data

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def prep(encoder):
    sites = site_data(encoder)
    tab, feats, ymap = sites["PAR"]
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide_l"] = lab.slide_id.astype(str).str.lower()
    feats_l = {NORM(k): v for k, v in feats.items()}
    rows = []
    for _, r in lab.iterrows():
        k = NORM(r.slide_l)
        if k not in feats_l:
            continue
        vals = [r.isup_1, r.isup_r2, r.isup_r3]
        vals = [int(v) for v in vals if pd.notna(v)]
        if not vals:
            continue
        rows.append({"slide": k, "group": k.rstrip("abcdefghijklmnopqrstuvwxyz"),
                     "X": feats_l[k].mean(0), "grades": vals})
    return rows


def hist(vals, n=6):
    h = np.zeros(n)
    for v in vals:
        h[int(v)] += 1
    return h / h.sum()


def entropy(p):
    p = np.clip(p, 1e-9, 1)
    return float(-(p * np.log(p)).sum())


def run(rows, C=0.05):
    X = np.stack([r["X"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    n = len(rows)
    prob_long = np.zeros((n, 6))
    prob_maj = np.zeros((n, 6))
    for tr, va in GroupKFold(n_splits=5).split(X, groups=groups):
        Xtr = [X[i] for i in tr for _ in rows[i]["grades"]]
        ytr = [g for i in tr for g in rows[i]["grades"]]
        Xva = [X[i] for i in tr for _ in rows[i]["grades"]]
        sc = StandardScaler().fit(np.stack(Xtr))
        lr = LogisticRegression(C=C, max_iter=1500, class_weight="balanced")
        lr.fit(sc.transform(np.stack(Xtr)), ytr)
        prob_long[va] = lr.predict_proba(sc.transform(X[va]))
        # majority model
        ymaj = np.array([int(np.median(rows[i]["grades"])) for i in tr])
        lr2 = LogisticRegression(C=C, max_iter=1500, class_weight="balanced")
        lr2.fit(sc.transform(X[tr]), ymaj)
        p2 = np.zeros((len(va), 6))
        cls = list(lr2.classes_)
        p2[:, cls] = lr2.predict_proba(sc.transform(X[va]))
        prob_maj[va] = p2

    dis = np.array([len(set(r["grades"])) > 1 for r in rows])
    ent_obs = np.array([entropy(hist(r["grades"])) for r in rows])
    ent_long = np.array([entropy(p) for p in prob_long])
    ent_maj = np.array([entropy(p) for p in prob_maj])
    js = np.array([jensenshannon(hist(r["grades"]), p, base=2)
                   for r, p in zip(rows, prob_long)])
    p_ge2 = prob_long[:, 2:].sum(1)
    out = {
        "n": n, "n_disagree": int(dis.sum()),
        "auc_entropy_long_vs_disagreement": float(roc_auc_score(dis, ent_long))
        if dis.any() else None,
        "auc_entropy_majority_vs_disagreement": float(roc_auc_score(dis, ent_maj))
        if dis.any() else None,
        "spearman_entropy_vs_observed": float(spearmanr(ent_long, ent_obs).statistic),
        "mean_js": float(js.mean()),
        "mean_entropy_long": float(ent_long.mean()),
        "mean_entropy_majority": float(ent_maj.mean()),
    }
    # deferral capture at fixed rates, ranking by entropy
    order = np.argsort(-ent_long)
    for rate in (0.1, 0.2, 0.3):
        k = int(rate * n)
        cap = float(dis[order[:k]].mean()) if k else float("nan")
        cap_maj = float(dis[np.argsort(-ent_maj)[:k]].mean()) if k else float("nan")
        out[f"capture@{int(rate*100)}_long"] = cap
        out[f"capture@{int(rate*100)}_majority"] = cap_maj
        out[f"capture@{int(rate*100)}_random"] = float(dis.mean())
    # boundary-mass deferral (P(ge2) in [0.3,0.7])
    band = (p_ge2 >= 0.3) & (p_ge2 <= 0.7)
    out["boundary_rate"] = float(band.mean())
    out["boundary_capture"] = float(dis[band].mean()) if band.any() else None
    return out, prob_long, prob_maj, dis, ent_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.out or ROOT / f"runs/reader_distribution_{args.encoder}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = prep(args.encoder)
    print(f"slides with features+labels: {len(rows)}", flush=True)
    res, pl, pm, dis, ent_obs = run(rows)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in res.items()}, indent=2))
    np.savez(outdir / "predictions.npz", prob_long=pl, prob_majority=pm,
             disagreement=dis, entropy_observed=ent_obs)
    (outdir / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
