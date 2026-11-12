#!/usr/bin/env python3
"""DOC pilot v2: sequential labelling + refit + conformal threshold.

At each budget K: sample K labelled slides (random / uncertainty / diversity),
refit a slide-level linear probe on their mean tile embeddings, then pick the
one-sided acceptance threshold tau by Clopper-Pearson control of accepted
false-negative rate at alpha. Evaluate deferral rate and accepted error on
the full site, plus capture of reader-disagreement cases (PAR).
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

ROOT = Path(__file__).resolve().parents[1]


def cp_upper(errors, n, delta=0.05):
    if n == 0:
        return 1.0
    if errors == n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def choose_tau(scores, y, alpha, delta, min_acc=5):
    order = np.argsort(scores)
    best = None
    n_acc = errs = 0
    for i in order:
        n_acc += 1
        errs += int(y[i] == 1)
        if n_acc < min_acc:
            continue
        if cp_upper(errs, n_acc, delta) <= alpha:
            best = (scores[i], n_acc)
    return None if best is None else best[0]


def slide_features(feats_dir):
    out = {}
    for f in sorted(Path(feats_dir).glob("*.npz")):
        z = np.load(f)
        out[f.stem.lower()] = z["tile_emb"].astype(np.float32).mean(0)
    return out


def site_tables():
    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    disagree = r.apply(lambda x: (x.dropna().nunique() > 1)
                       if x.notna().sum() >= 2 else np.nan, axis=1)
    par = par.merge(maj.rename("y_raw"), left_on="slide", right_index=True)
    par = par.merge(disagree.rename("readers_disagree"), left_on="slide",
                    right_index=True)
    par = par.dropna(subset=["y_raw"])
    par["y"] = (par.y_raw >= 2).astype(int)

    tc = pd.read_csv(ROOT / "runs/tcga_external/tcga_scores.csv")
    tc["y"] = (tc.gg.astype(int) >= 2).astype(int)
    tc["y_raw"] = tc.gg.astype(int)
    tc["readers_disagree"] = np.nan
    return {"PAR": (par, ROOT / "features/phikon"),
            "TCGA": (tc, ROOT / "features/tcga_phikon")}


def run_site(name, tab, feats_dir, strategies, Ks, alpha, delta, repeats):
    feats = slide_features(feats_dir)
    tab = tab.copy(); tab["slide_l"] = tab.slide.astype(str).str.lower()
    tab = tab[tab.slide_l.isin(feats)].reset_index(drop=True)
    X = np.stack([feats[s] for s in tab.slide_l])
    y = tab.y.to_numpy()
    y_raw = tab.y_raw.to_numpy()
    dis = tab.readers_disagree.to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for K in Ks:
        if K >= len(y):
            continue
        for strat in strategies:
            for rep in range(repeats if strat == "random" else 1):
                if strat == "random":
                    idx = rng.choice(len(y), K, replace=False)
                    Xtr, ytr = X[idx], y[idx]
                else:
                    sc0 = StandardScaler().fit(X)
                    lr0 = LogisticRegression(C=0.1, max_iter=800,
                                             class_weight="balanced")
                    lr0.fit(sc0.transform(X), y)
                    p0 = lr0.predict_proba(sc0.transform(X))[:, 1]
                    if strat == "uncertainty":
                        sel = np.argsort(np.abs(p0 - 0.5))[:K]
                    else:  # diversity on score
                        sel = [int(np.argmin(np.abs(p0 - 0.5)))]
                        while len(sel) < K:
                            d = np.min(np.abs(p0[:, None] - p0[sel][None, :]),
                                       axis=1)
                            sel.append(int(np.argmax(d)))
                        sel = np.array(sel)
                    Xtr, ytr = X[sel], y[sel]
                if len(np.unique(ytr)) < 2:
                    rows.append({"site": name, "K": K, "strategy": strat,
                                 "deferral": 1.0, "acc_err": np.nan,
                                 "fn_on_all": np.nan, "disagree_deferred": np.nan})
                    continue
                sc = StandardScaler().fit(Xtr)
                lr = LogisticRegression(C=0.1, max_iter=800,
                                        class_weight="balanced")
                lr.fit(sc.transform(Xtr), ytr)
                s_tr = lr.predict_proba(sc.transform(Xtr))[:, 1]
                tau = choose_tau(s_tr, ytr, alpha, delta)
                p = lr.predict_proba(sc.transform(X))[:, 1]
                if tau is None:
                    rows.append({"site": name, "K": K, "strategy": strat,
                                 "deferral": 1.0, "acc_err": np.nan,
                                 "fn_on_all": np.nan, "disagree_deferred": np.nan})
                    continue
                acc = p <= tau
                defer = ~acc
                acc_err = float(y[acc].mean()) if acc.sum() else np.nan
                dd = (float(defer[np.asarray(dis == True)].mean())
                      if name == "PAR" else np.nan)  # noqa: E712
                rows.append({"site": name, "K": K, "strategy": strat,
                             "deferral": float(defer.mean()),
                             "acc_err": acc_err,
                             "n_acc": int(acc.sum()),
                             "disagree_deferred": dd})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/doc_pilot")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    Ks = [10, 15, 20, 30, 40, 60, 80]
    out = []
    for name, (tab, fd) in site_tables().items():
        res = run_site(name, tab, fd, ["random", "uncertainty", "diversity"],
                       Ks, args.alpha, args.delta, args.repeats)
        out.append(res)
        piv = res.groupby(["strategy", "K"]).agg(
            deferral=("deferral", "mean"), acc_err=("acc_err", "mean"),
            disagree_deferred=("disagree_deferred", "mean")).round(3)
        print(f"\n=== {name} (alpha={args.alpha}) ===")
        print(piv.to_string())
    pd.concat(out).to_csv(args.out / "doc_pilot_v2.csv", index=False)


if __name__ == "__main__":
    main()
