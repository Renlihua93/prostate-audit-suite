#!/usr/bin/env python3
"""Clinical utility simulation for risk-controlled AS triage.

Pipeline per seed:
  1. Train MGOD head (SICAP tile-CE + SICAP slide MIL) on all SICAP slides.
  2. Score every PAR/TCGA slide zero-shot: mean tile P(GP4/GP5).
  3. Repeated split: sample K calibration slides (stratified by label),
     choose two-sided conformal thresholds on them at alpha;
     evaluate on the remaining held-out slides.

Reported on held-out slides:
  accepted fraction (auto-cleared / auto-escalated), deferral,
  sensitivity/PPV for ISUP>=2, missed high-risk per 100 slides,
  reader-disagreement deferral (PAR), and decision-curve net benefit
  (AI-only vs AI+defer vs read-all vs treat-all).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from doc_pilot_v3 import choose_thresholds
from mgod_pilot import load_sicap, train
from mgod_loco import load_slides

ROOT = Path(__file__).resolve().parents[1]


def site_data(encoder):
    norm = lambda x: str(x).lower().replace("_leica", "").replace(
        "_grundium", "")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="	")
    lab["slide"] = lab.slide_id.astype(str)
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    dis = r.apply(lambda x: bool(x.dropna().nunique() > 1)
                  if x.notna().sum() >= 2 else np.nan, axis=1)
    par = lab[["slide"]].copy()
    par["maj"] = maj.reindex(par.slide).to_numpy()
    par["disagree"] = dis.reindex(par.slide).to_numpy()
    par = par.dropna(subset=["maj"])
    par["y"] = (par.maj >= 2).astype(int)
    par_dir = (ROOT / "features/phikon" if encoder == "phikon"
               else ROOT / "features/hoptimus0_par")
    par_f, par_y = load_slides(par_dir, pd.DataFrame(
        {"slide": par.slide.astype(str).str.lower(), "y": par.y}))
    have = set(par_f)
    par = par[par.slide.astype(str).str.lower().apply(norm).isin(
        {norm(s) for s in have})].reset_index(drop=True)

    tc = pd.read_csv(ROOT / "data/refined_tcga_prad_matched.csv")
    tc["slide"] = tc.slide_name.astype(str).str.lower().str.replace(
        ".svs", "", regex=False)
    tc = tc.dropna(subset=["GG"]).drop_duplicates("slide")
    tc["y"] = (tc.GG.astype(int) >= 2).astype(int)
    tc["disagree"] = np.nan
    tc_dir = (ROOT / "features/tcga_phikon" if encoder == "phikon"
              else ROOT / "features/hoptimus0_tcga")
    tc_f, tc_y = load_slides(tc_dir, pd.DataFrame(
        {"slide": tc.slide, "y": tc.y}))
    tc = tc[tc.slide.astype(str).str.lower().apply(norm).isin(
        {norm(s) for s in tc_f})].reset_index(drop=True)
    return {"PAR": (par, par_f, par_y), "TCGA": (tc, tc_f, tc_y)}


def score_slides(model, feats, slides, dev):
    out = {}
    with torch.no_grad():
        for s in slides:
            lt, _ = model(torch.from_numpy(feats[s]).to(dev))
            p = torch.softmax(lt, -1)
            out[s] = (float(p[:, 2:].sum(1).mean().item()),
                      float((1 - p[:, 0]).mean().item()))
    return out


def net_benefit(y, treat, w):
    tp = int(((treat == 1) & (y == 1)).sum())
    fp = int(((treat == 1) & (y == 0)).sum())
    n = len(y)
    return tp / n - fp / n * (w / (1 - w))


def cp_upper(errors, n, delta):
    from scipy.stats import beta
    if n == 0:
        return 1.0
    if errors == n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def tau_one_sided(scores, y, alpha, delta=0.05, min_acc=5):
    order = np.argsort(scores)
    best = None
    n_acc = errs = 0
    for i in order:
        n_acc += 1
        errs += int(y[i] == 1)
        if n_acc < min_acc:
            continue
        if cp_upper(errs, n_acc, delta) <= alpha:
            best = scores[i]
    return best


def refit_scores(X, y, cal):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y[cal])) < 2:
        return None
    sc = StandardScaler().fit(X[cal])
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(X[cal]), y[cal])
    return lr.predict_proba(sc.transform(X))[:, 1]


def gate_rule(sg, sc, y, cal, alpha, delta=0.05, min_acc=5):
    best = None
    for qg in np.linspace(0.1, 0.6, 11):
        tg = np.quantile(sg[cal], qg)
        for qc in np.linspace(0.1, 0.9, 17):
            tc_ = np.quantile(sc[cal], qc)
            clear = (sg[cal] <= tg) & (sc[cal] <= tc_)
            n_acc = int(clear.sum())
            if n_acc < min_acc:
                continue
            errs = int((clear & (y[cal] == 1)).sum())
            if cp_upper(errs, n_acc, delta) <= alpha:
                if best is None or n_acc > best[0]:
                    best = (n_acc, tg, tc_)
    return None if best is None else (best[1], best[2])


def simulate(scores, X, y, dis, Ks, alphas, repeats, rng, quantiles,
             cancer=None):
    n = len(y)
    rows = []
    for K in Ks:
        for alpha in alphas:
            for rep in range(repeats):
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                kpos = min(K // 2, len(pos))
                kneg = min(K - kpos, len(neg))
                cal = np.concatenate([
                    rng.choice(pos, kpos, replace=False),
                    rng.choice(neg, kneg, replace=False)])
                test = np.setdiff1d(np.arange(n), cal)
                if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                    continue
                sc_all = refit_scores(X, y, cal)
                if sc_all is None:
                    continue
                s, yt = sc_all[test], y[test]
                tau = tau_one_sided(sc_all[cal], y[cal], alpha)
                gate = (gate_rule(sc_all, cancer, y, cal, alpha)
                        if cancer is not None else None)
                row = {"K": K, "alpha": alpha, "rep": rep,
                       "n_test": len(yt), "feasible": tau is not None}
                if gate is not None:
                    tg, tc_ = gate
                    gclear = (s <= tg) & (cancer[test] <= tc_)
                    gfn = int((gclear & (yt == 1)).sum())
                    row.update({
                        "gate_cleared": float(gclear.mean()),
                        "gate_missed": 100 * gfn / len(yt),
                        "gate_sens": 1 - gfn / max(int(yt.sum()), 1)})
                if tau is not None:
                    clear = s <= tau
                    fn = int((clear & (yt == 1)).sum())
                    row.update({
                        "cleared": float(clear.mean()),
                        "deferral": float(1 - clear.mean()),
                        "missed_per100": 100 * fn / len(yt),
                        "sens_detected": 1 - fn / max(int(yt.sum()), 1),
                        "disagree_deferred": (float((~clear)[np.asarray(
                            dis[test] == True)].mean())  # noqa: E712
                            if dis is not None and np.any(dis[test] == True)  # noqa: E712
                            else np.nan)})
                # unconstrained tradeoff at calibration quantiles
                for q in quantiles:
                    tau_q = float(np.quantile(sc_all[cal], q))
                    clear = s <= tau_q
                    fn = int((clear & (yt == 1)).sum())
                    row[f"q{int(q * 100)}_cleared"] = float(clear.mean())
                    row[f"q{int(q * 100)}_missed"] = 100 * fn / len(yt)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mode", default="tile_plus_slide",
                    choices=["tile_only", "tile_plus_slide"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--Ks", nargs="+", type=int, default=[30, 50, 80])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    gg_keep = {s: gg[s] for s in set(slides[keep]) if s in gg}
    sites = site_data(args.encoder)
    rng = np.random.default_rng(0)
    all_rows = []
    for seed in args.seeds:
        params = {"epochs": 10, "batch": 256,
                  "lam": 0.0 if args.mode == "tile_only" else 0.5}
        model = train(args.mode, emb[keep], labels[keep],
                      slides[keep], gg_keep, params, seed=seed)
        model.eval()
        for name, (tab, feats, ymap) in sites.items():
            sc = score_slides(model, feats, list(ymap), dev)
            tab = tab.copy()
            tab["score"] = tab.slide.str.lower().map(lambda k: sc[k][0])
            tab["cancer"] = tab.slide.str.lower().map(lambda k: sc[k][1])
            tab = tab.dropna(subset=["score"])
            y = tab.y.to_numpy()
            dis = (tab.disagree.to_numpy()
                   if "disagree" in tab else np.array([np.nan] * len(tab)))
            Xs = np.stack([feats[s2].mean(0) for s2 in tab.slide.str.lower()])
            df = simulate(tab.score.to_numpy(), Xs, y, dis, args.Ks,
                          args.alphas, args.repeats, rng, quantiles,
                          cancer=tab.cancer.to_numpy())
            if df.empty or "feasible" not in df.columns:
                print(f"[skip] {name}: no certifiable simulation cells",
                      flush=True)
                continue
            df["site"] = name
            df["seed"] = seed
            all_rows.append(df)
            ok = df[df.feasible]
            if len(ok):
                from sklearn.metrics import roc_auc_score
                print(name, "seed", seed, "AUC",
                      round(float(roc_auc_score(y, tab.score)), 3),
                      "feasible", f"{len(ok)}/{len(df)}", flush=True)

    res = pd.concat(all_rows, ignore_index=True)
    out = args.out or ROOT / f"runs/clinical_utility_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "simulation.csv", index=False)
    ok = res[res.feasible]
    if len(ok):
        summ = ok.groupby(["site", "alpha", "K"]).agg(
            cleared=("cleared", "mean"), deferral=("deferral", "mean"),
            sens_detected=("sens_detected", "mean"),
            missed_per100=("missed_per100", "mean"),
            disagree_deferred=("disagree_deferred", "mean"),
            n_rep=("rep", "count")).round(3)
        print(summ.to_string())
        summ.to_csv(out / "summary.csv")
    print("done")


if __name__ == "__main__":
    main()
