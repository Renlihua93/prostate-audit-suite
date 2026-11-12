#!/usr/bin/env python3
"""Ordinal threshold calibration for grade agreement (QWK) across cohorts.

Slide score = expected grade from the 5-class slide head (SICAP-trained).
Cutpoints mapping the score to grades 1-5 come from
  source   - SICAP training score quantiles matched to grade proportions
  site-K   - K labelled target slides, greedy coordinate ascent on QWK
Evaluation: QWK on held-out PAR slides (majority ISUP 1-5; benign excluded).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

from mgod_pilot import load_sicap, train

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def expected_grade(model, feats, dev):
    out = {}
    with torch.no_grad():
        for s, x in feats.items():
            _, ls = model(torch.from_numpy(x).to(dev))
            p = torch.softmax(ls, -1).cpu().numpy()
            out[s] = float((p * np.arange(1, 6)).sum())
    return out


def apply_cuts(score, cuts):
    pred = np.ones(len(score), dtype=int)
    for c in cuts:
        pred += (score > c).astype(int)
    return pred.astype(float)


def qwk(y, pred):
    return cohen_kappa_score(np.asarray(y).astype(int),
                             np.asarray(pred).astype(int),
                             weights="quadratic")


def source_cuts(scores, grades):
    """Quantiles of source scores matching cumulative grade proportions."""
    props = np.array([(grades <= k).mean() for k in (2, 3, 4, 5)])
    return [float(np.quantile(scores, p)) for p in props]


def site_cuts(scores, y, iters=3, cand=25):
    base = np.quantile(scores, [0.25, 0.5, 0.75, 0.9]).tolist()
    best = qwk(y, apply_cuts(scores, base))
    cuts = base
    grid = np.quantile(scores, np.linspace(0.05, 0.98, cand))
    for _ in range(iters):
        improved = False
        for i in range(4):
            for g in grid:
                trial = list(cuts)
                trial[i] = float(g)
                trial = sorted(trial)
                s = qwk(y, apply_cuts(scores, trial))
                if s > best + 1e-9:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    return cuts


def simulate(scores, y, Ks, reps, rng):
    n = len(y)
    rows = []
    for K in Ks:
        for rep in range(reps):
            cal = rng.choice(n, K, replace=False)
            test = np.setdiff1d(np.arange(n), cal)
            if len(np.unique(y[cal])) < 2:
                continue
            cuts = site_cuts(scores[cal], y[cal])
            rows.append({"K": K, "rep": rep, "qwk_site": qwk(
                y[test], apply_cuts(scores[test], cuts))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=[20, 40, 60])
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    emb, labels, slides = emb[keep], labels[keep], slides[keep]

    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: float(np.floor(np.nanmedian(
        x.dropna().astype(float)) + 0.5)) if x.notna().any() else np.nan,
        axis=1)
    par = par.merge(maj.rename("maj"), left_on="slide", right_index=True)
    par = par.dropna(subset=["maj"])
    par = par[par.maj >= 1]  # benign excluded: source head has grades 1-5
    feats_dir = (ROOT / "features/phikon" if args.encoder == "phikon"
                 else ROOT / "features/hoptimus0_par")
    feats = {}
    for f in sorted(feats_dir.glob("*.npz")):
        feats[NORM(f.stem)] = np.load(f)["tile_emb"].astype(np.float32)
    keys = par.slide.str.lower().apply(NORM)
    keep2 = keys.isin(list(feats))
    par = par[keep2].reset_index(drop=True)
    keys = keys[keep2]
    y = par.maj.to_numpy().astype(float)
    print(f"PAR graded slides (ISUP>=1): {len(y)}", flush=True)

    slides_uniq = np.array(sorted(gg))
    patients = np.array([pat[s] for s in slides_uniq])
    rows = []
    for seed in args.seeds:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2,
                                        random_state=seed).split(
            slides_uniq, groups=patients))
        g_tr = set(slides_uniq[tr])
        mask = np.array([slides[i] in g_tr for i in range(len(labels))])
        model = train("tile_plus_slide", emb[mask], labels[mask],
                      slides[mask], {s: gg[s] for s in g_tr},
                      {"epochs": 10, "batch": 256, "lam": 0.5}, seed=seed)
        model.eval()
        sc = expected_grade(model, feats, dev)
        s_par = np.array([sc[k] for k in keys])
        src_scores, src_grades = [], []
        with torch.no_grad():
            for s in g_tr:
                x = emb[slides == s]
                if len(x) == 0:
                    continue
                _, ls = model(torch.from_numpy(x).to(dev))
                p = torch.softmax(ls, -1).cpu().numpy()
                src_scores.append(float((p * np.arange(1, 6)).sum()))
                src_grades.append(float(gg[s]))
        cuts_src = source_cuts(np.array(src_scores), np.array(src_grades))
        base_qwk = float(qwk(y, apply_cuts(s_par, cuts_src)))
        sim = simulate(s_par, y, args.Ks, args.reps, rng)
        sim["seed"] = seed
        sim["qwk_source"] = base_qwk
        rows.append(sim)
        print(f"seed{seed}: source-threshold QWK={base_qwk:.3f}",
              flush=True)
    res = pd.concat(rows, ignore_index=True)
    out = ROOT / f"runs/ordinal_qwk_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    summ = res.groupby("K").agg(qwk_site=("qwk_site", "mean"),
                                qwk_source=("qwk_source", "mean")).round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()
