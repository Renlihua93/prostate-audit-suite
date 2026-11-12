#!/usr/bin/env python3
"""Pitfall effect-size matrix for the evaluation-audit paper.

Quantifies, on the existing six-cohort assets, how standard evaluation
shortcuts change conclusions:
  1. source-tuned hyperparameters (C on SICAP macro-F1 vs fixed C=0.001 raw)
  2. calibration-set selection bias (active vs audit-valid conformal)
  3. label aggregation ties (median 1.5/2.5 and the 1-vs-2 boundary)
  4. straw-man baseline (tile-only vs standard MIL)
  5. unsupervised alignment sign flips (cohort centering)
Written to runs/pitfall_audit/matrix.csv.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pitfall_audit"
OUT.mkdir(parents=True, exist_ok=True)
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def load_par_scores(encoder="phikon"):
    SIC = {
        "phikon": "features/sicap_phikon_c224/patch_features.npz",
        "hoptimus0": "features/sicap_c224/patch_features.npz",
        "resnet50": "features/sicap_c224/patch_features.npz",
    }
    PAR = {"phikon": "features/phikon", "hoptimus0": "features/hoptimus0_par",
           "resnet50": "features/resnet50_par"}
    TCGA = {"phikon": "features/tcga_phikon",
            "hoptimus0": "features/hoptimus0_tcga",
            "resnet50": "features/resnet50_tcga"}
    sic = np.load(ROOT / SIC[encoder])
    k = sic["labels"] >= 0
    Xs, ys = sic[f"emb_{encoder}"][k], sic["labels"][k]
    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    par = par.merge(maj.rename("maj"), left_on="slide", right_index=True)
    par = par.dropna(subset=["maj"])
    feats_dir = ROOT / PAR[encoder]
    feats = {}
    for f in sorted(feats_dir.glob("*.npz")):
        feats[NORM(f.stem)] = np.load(f)["tile_emb"].astype(np.float32)
    keys = par.slide.str.lower().apply(NORM)
    keep = keys.isin(list(feats))
    par, keys = par[keep], keys[keep]
    Xp = np.stack([feats[k2].mean(0) for k2 in keys])
    yp = (par.maj.to_numpy() >= 2).astype(int)
    # TCGA
    tc = pd.read_csv(ROOT / "data/refined_tcga_prad_matched.csv")
    tc["slide"] = tc.slide_name.astype(str).str.lower().str.replace(
        ".svs", "", regex=False)
    tc = tc.dropna(subset=["GG"]).drop_duplicates("slide")
    tdir = ROOT / TCGA[encoder]
    tf = {}
    for f in sorted(tdir.glob("*.npz")):
        tf[NORM(f.stem)] = np.load(f)["tile_emb"].astype(np.float32)
    tk = tc.slide.apply(NORM)
    keep = tk.isin(list(tf))
    tc, tk = tc[keep], tk[keep]
    Xt = np.stack([tf[k2].mean(0) for k2 in tk])
    yt = (tc.GG.astype(int).to_numpy() >= 2).astype(int)
    return Xs, ys, (Xp, yp), (Xt, yt)


def hyperparameter_effect():
    Xs, ys, (Xp, yp), (Xt, yt) = load_par_scores("phikon")
    out = {}
    for tag, scaler, C in (("source_tuned", True, 0.01),
                           ("fixed_raw", False, 0.001)):
        F = StandardScaler().fit(Xs).transform if scaler else (lambda x: x)
        lr = LogisticRegression(C=C, max_iter=1500,
                                class_weight="balanced").fit(F(Xs), ys)
        idx = [i for i, c in enumerate(lr.classes_) if c >= 2]
        for name, (X, y) in (("PAR", (Xp, yp)), ("TCGA", (Xt, yt))):
            auc = roc_auc_score(y, lr.predict_proba(F(X))[:, idx].sum(1))
            out[f"auc_{name}_{tag}"] = float(auc)
    out["delta_PAR"] = out["auc_PAR_fixed_raw"] - out["auc_PAR_source_tuned"]
    out["delta_TCGA"] = out["auc_TCGA_fixed_raw"] - out["auc_TCGA_source_tuned"]
    return out


def tie_effect():
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    r = lab[["isup_1", "isup_r2", "isup_r3"]]
    n = 0
    flips = 0
    for _, row in r.iterrows():
        vals = [float(v) for v in row if pd.notna(v)]
        if len(vals) < 2:
            continue
        n += 1
        med = np.median(vals)
        if med != int(med):  # tie produces x.5
            low = int(np.floor(med))
            high = int(np.ceil(med))
            if low < 2 <= high:
                flips += 1
    return {"n_multireader": n, "ties": int(sum(
        1 for _, row in r.iterrows()
        if pd.notna(row).sum() >= 2
        and np.median([float(v) for v in row if pd.notna(v)])
        != int(np.median([float(v) for v in row if pd.notna(v)])))),
        "ties_crossing_ge2": flips}


def strawman_effect():
    d = json.load(open(ROOT / "runs/mgod_pilot_phikon/results.json"))["summary"]
    d2 = json.load(open(ROOT / "runs/mgod_pilot_hoptimus0/results.json"))["summary"]
    out = {}
    for tag, dd in (("phikon", d), ("hoptimus0", d2)):
        out[f"tcga_tile_only_{tag}"] = dd["tile_only"]["tcga_auc_ge2"]["mean"]
        out[f"tcga_weak_{tag}"] = dd["tile_plus_slide"]["tcga_auc_ge2"]["mean"]
        out[f"tcga_weak_gain_{tag}"] = (out[f"tcga_weak_{tag}"]
                                        - out[f"tcga_tile_only_{tag}"])
    return out


def selection_bias_effect():
    naive = pd.read_csv(ROOT / "runs/active_acq_phikon/simulation.csv")
    naive = naive[(naive.alpha == 0.10) & (naive.K == 80)]
    audit = pd.read_csv(ROOT / "runs/audit_active_phikon/simulation.csv")
    audit = audit[(audit.K == 80)]
    out = {"naive_fn_clear_uncertainty": float(
        naive[naive.strategy == "uncertainty"].fn_clear.mean()),
        "naive_feasible": float(
            naive[naive.strategy == "uncertainty"].feasible.mean()),
        "audit_fn_clear_uncertainty": float(
            audit[audit.arm == "uncertainty_fit"].fn_clear.mean()),
        "audit_feasible": float(
            audit[audit.arm == "uncertainty_fit"].feasible.mean())}
    return out


def alignment_effect():
    d = pd.read_csv(ROOT / "runs/domain_adapt/results.csv")
    out = {}
    for site in ("tcga", "par"):
        g = d[d.cohort == site].set_index("setting").auc_ge2
        out[f"{site}_zero"] = float(g.get("zero_shot", np.nan))
        out[f"{site}_centering"] = float(g.get("cohort_centering", np.nan))
        out[f"{site}_delta"] = (out[f"{site}_centering"]
                                - out[f"{site}_zero"])
    return out


def main():
    rows = []
    for name, fn in (("hyperparameter", hyperparameter_effect),
                     ("ties", tie_effect),
                     ("strawman", strawman_effect),
                     ("selection_bias", selection_bias_effect),
                     ("alignment", alignment_effect)):
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001
            res = {"error": str(exc)}
        for k, v in res.items():
            rows.append({"pitfall": name, "metric": k, "value": v})
        print(name, json.dumps(res, indent=2, default=str), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matrix.csv", index=False)
    print("wrote", OUT / "matrix.csv")


if __name__ == "__main__":
    main()
