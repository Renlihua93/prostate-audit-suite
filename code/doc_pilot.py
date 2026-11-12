#!/usr/bin/env python3
"""DOC pilot: reader-consensus, sequential risk-controlled triage.

Source probe: SICAPv2 (Phikon-v1), scores from runs/par_external/par_scores.csv
and runs/tcga_external/tcga_scores.csv.

Per site:
  - majority ISUP (PAR 2-3 readers; TCGA single label)
  - one-sided acceptance: accept as low-risk (for AS) if score <= tau;
    false-negative (accepted but true ISUP>=2) rate controlled by a
    Clopper-Pearson upper bound at 1-delta from the labelled subset
  - sequential labelling strategies: random / uncertainty / diversity
  - report deferral rate and accepted error vs K labels
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def cp_upper(errors: int, n: int, delta: float = 0.05) -> float:
    if n == 0:
        return 1.0
    if errors == n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def choose_tau(scores, y, alpha=0.05, delta=0.05, min_acc=5):
    """Largest acceptance (tau) with CP upper bound on accepted FN <= alpha."""
    order = np.argsort(scores)
    s_sorted, y_sorted = scores[order], y[order]
    best = None
    n_acc = 0
    errs = 0
    for i, (s, yi) in enumerate(zip(s_sorted, y_sorted)):
        n_acc += 1
        errs += int(yi == 1)  # accepted but ISUP>=2 = false negative
        if n_acc < min_acc:
            continue
        if cp_upper(errs, n_acc, delta) <= alpha:
            if best is None or n_acc > best[1]:
                best = (s, n_acc)
    return None if best is None else best[0]


def run_site(name, scores, y, strategies, Ks, alpha=0.05, delta=0.05,
             repeats=50, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    order_all = np.argsort(-scores)  # high score first (uncertain-ish end)
    rows = []
    for K in Ks:
        if K > n:
            continue
        for strat in strategies:
            reps = repeats if strat == "random" else 1
            for rep in range(reps):
                if strat == "random":
                    idx = rng.choice(n, K, replace=False)
                elif strat == "uncertainty":
                    unc = np.abs(scores - np.median(scores))
                    idx = np.argsort(-unc)[:K]
                elif strat == "diversity":
                    # farthest-point sampling on 1-D score
                    idx = [int(np.argmax(np.abs(scores - np.mean(scores))))]
                    while len(idx) < K:
                        d = np.min(np.abs(scores[:, None]
                                           - scores[idx][None, :]), axis=1)
                        idx.append(int(np.argmax(d)))
                    idx = np.array(idx)
                else:
                    raise ValueError(strat)
                tau = choose_tau(scores[idx], y[idx], alpha, delta)
                if tau is None:
                    rows.append({"site": name, "K": K, "strategy": strat,
                                 "tau": np.nan, "deferral": 1.0,
                                 "acc_err": np.nan, "stopped": False})
                    continue
                accept = scores <= tau
                acc_err = float(y[accept].mean()) if accept.sum() else np.nan
                rows.append({"site": name, "K": K, "strategy": strat,
                             "tau": float(tau),
                             "deferral": float(1 - accept.mean()),
                             "acc_err": acc_err,
                             "stopped": True})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/doc_pilot")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- PAR: majority over 2-3 readers
    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    rr = lab[["slide", "isup_1", "isup_r2", "isup_r3"]]
    maj = rr.set_index("slide").apply(
        lambda r: np.nanmedian(r.dropna().astype(float)), axis=1)
    agree = rr.set_index("slide").apply(
        lambda r: (r.dropna().astype(float).max()
                   - r.dropna().astype(float).min() == 0)
        if r.notna().sum() >= 2 else np.nan, axis=1)
    par = par.merge(maj.rename("maj_isup"), left_on="slide", right_index=True,
                    how="inner")
    par = par.merge(agree.rename("unanimous"), left_on="slide", right_index=True,
                    how="left")
    par = par.dropna(subset=["maj_isup"])
    y_par = (par.maj_isup.astype(int) >= 2).astype(int).to_numpy()
    s_par = par.g4_score.to_numpy()
    print(f"PAR slides={len(par)} ISUP>=2={int(y_par.sum())} "
          f"AUC={roc_auc_score(y_par, s_par):.3f} "
          f"multi-reader={(par.unanimous.notna()).sum()}", flush=True)

    # ---- TCGA
    tc = pd.read_csv(ROOT / "runs/tcga_external/tcga_scores.csv")
    y_tc = (tc.gg.astype(int) >= 2).astype(int).to_numpy()
    s_tc = tc.g4_score.to_numpy()
    print(f"TCGA slides={len(tc)} GG>=2={int(y_tc.sum())} "
          f"AUC={roc_auc_score(y_tc, s_tc):.3f}", flush=True)

    Ks = [5, 10, 15, 20, 30, 40, 60, 80]
    res = pd.concat([
        run_site("PAR", s_par, y_par, ["random", "uncertainty", "diversity"],
                 Ks, args.alpha, args.delta),
        run_site("TCGA", s_tc, y_tc, ["random", "uncertainty", "diversity"],
                 Ks, args.alpha, args.delta)])
    res.to_csv(args.out / "doc_pilot.csv", index=False)
    piv = res.groupby(["site", "strategy", "K"]).agg(
        deferral=("deferral", "mean"), acc_err=("acc_err", "mean")).round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
