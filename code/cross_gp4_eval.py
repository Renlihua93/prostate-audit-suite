#!/usr/bin/env python3
"""Cross-cohort %GP4 transfer: Arvaniti TMA cores <-> SICAPv2 slides.

Both feature sets are H-optimus-0 embeddings of 112 um FOV tiles, so this is
the cleanest cross-cohort %GP4 test available. Tile classifier is a 3-class
softmax over GP3/GP4/GP5; cohort %GP4 pools predicted GP4 probability over
tumor probability, weighted by labelled area (Arvaniti) or tile count
(SICAPv2).

Arvaniti ground truth: mask area fractions per tile -> core %GP4.
SICAPv2 ground truth: pixel masks -> slide %GP4 (see runs/sicap_gp4).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from gp4_dense_probe import frac_targets, load_tiles

ROOT = Path(__file__).resolve().parents[1]


def core_gp4(core, lab, p):
    """Area-weighted %GP4 per Arvaniti core from tile probabilities p[3]."""
    df = pd.DataFrame({"core": core, "w": lab,
                       "tumor": p.sum(1), "g4": p[:, 1]})
    df["wt"] = df.w * df.tumor
    df["wg"] = df.w * df.g4
    g = df.groupby("core")[["wt", "wg"]].sum()
    g["pred"] = 100 * g.wg / g.wt.clip(lower=1e-9)
    return g[["pred"]].reset_index()


def metrics(pred, true):
    out = {"n": int(len(true)),
           "mae": float(np.mean(np.abs(pred - true))),
           "spearman": float(spearmanr(pred, true).statistic)}
    y = (true >= 10).astype(int)
    if y.min() != y.max():
        out["auc_ge10"] = float(roc_auc_score(y, pred))
    band = true <= 20
    if band.sum() >= 5:
        out["mae_band20"] = float(np.mean(np.abs(pred[band] - true[band])))
    return out


def main():
    out_dir = ROOT / "runs/cross_gp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_tiles(ROOT / "runs/gp4_dense/tiles/hoptimus0")
    Xa = d["emb"][1]
    cores, frac, lab = d["core"], d["frac"], d["lab"]
    p_true, gp4_true = frac_targets(frac)
    hard = p_true.argmax(1)
    core_true = pd.DataFrame({"core": cores, "lab": lab,
                              "f": p_true.sum(1), "g4": p_true[:, 1]})
    core_true["wf"] = core_true.lab * core_true.f
    core_true["wg"] = core_true.lab * core_true.g4
    ct = core_true.groupby("core")[["wf", "wg"]].sum().reset_index()
    ct["true"] = 100 * ct.wg / ct.wf.clip(lower=1e-9)

    sic = np.load(ROOT / "features/sicap_c224/patch_features.npz")
    Xs = sic["emb_hoptimus0"]
    ys = sic["labels"].astype(int)
    ss = pd.Series(sic["slide_ids"].astype(str)).str.split("/") \
        .str[-1].to_numpy()
    sl = pd.read_csv(ROOT / "runs/sicap_gp4/slide_gp4_from_masks.csv")
    sl = sl[sl.tumour_px > 0]

    results = {}

    # ---- direction 1: Arvaniti -> SICAP
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(Xa, groups=cores))
    sc = StandardScaler().fit(Xa[tr])
    lr = LogisticRegression(C=0.03, max_iter=1000, tol=1e-3,
                            class_weight="balanced")
    lr.fit(sc.transform(Xa[tr]), hard[tr])
    # sanity: Arvaniti held-out cores
    p_te = lr.predict_proba(sc.transform(Xa[te]))
    ct_te = core_gp4(cores[te], lab[te], p_te)
    ct_te["true"] = ct_te.core.map(dict(zip(ct.core, ct.true)))
    results["arvaniti_to_arvaniti"] = metrics(ct_te.pred.to_numpy(),
                                              ct_te.true.to_numpy())
    # transfer: SICAP slides
    p_s = lr.predict_proba(sc.transform(Xs))
    tmp = pd.DataFrame({"slide": ss, "tumor": p_s.sum(1), "g4": p_s[:, 1]})
    agg = tmp.groupby("slide").agg(tumor=("tumor", "sum"),
                                   g4=("g4", "sum")).reset_index()
    agg["pred"] = 100 * agg.g4 / agg.tumor
    ext = agg.merge(sl[["slide", "gp4_frac"]], on="slide")
    ext["true"] = 100 * ext.gp4_frac
    results["arvaniti_to_sicap"] = metrics(ext.pred.to_numpy(),
                                           ext.true.to_numpy())

    # ---- direction 2: SICAP -> Arvaniti
    k = ys >= 1
    sc2 = StandardScaler().fit(Xs[k])
    lr2 = LogisticRegression(C=0.1, max_iter=1000, tol=1e-3,
                             class_weight="balanced")
    lr2.fit(sc2.transform(Xs[k]), (ys[k] - 1))
    p_a = lr2.predict_proba(sc2.transform(Xa))
    # classes are GP3/GP4/GP5; reorder to GP4 index 1
    pa = core_gp4(cores, lab, p_a)
    pa["true"] = pa.core.map(dict(zip(ct.core, ct.true)))
    results["sicap_to_arvaniti"] = metrics(pa.pred.to_numpy(),
                                           pa.true.to_numpy())

    (out_dir / "results.json").write_text(json.dumps(results, indent=2),
                                          encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
