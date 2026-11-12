#!/usr/bin/env python3
"""%GP4 transfer metrics at 5/10/20% thresholds and in-band MAE."""

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
OUT = ROOT / "runs/cross_gp4"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(pred, true, name):
    out = {"direction": name, "n": int(len(true)),
           "mae": float(np.mean(np.abs(pred - true))),
           "spearman": float(spearmanr(pred, true).statistic)}
    for thr in (5, 10, 20):
        y = (true >= thr).astype(int)
        if y.min() != y.max():
            out[f"auc_ge{thr}"] = float(roc_auc_score(y, pred))
    for band in (10, 20):
        m = true <= band
        if m.sum() >= 5:
            out[f"mae_le{band}"] = float(np.mean(np.abs(
                pred[m] - true[m])))
            out[f"n_le{band}"] = int(m.sum())
    return out


def main():
    rows = []
    # Arvaniti cores, tile probe
    d = load_tiles(ROOT / "runs/gp4_dense/tiles/hoptimus0")
    Xa = d["emb"][1]
    cores, frac, lab = d["core"], d["frac"], d["lab"]
    p_true, gp4_true = frac_targets(frac)
    hard = p_true.argmax(1)
    ct = pd.DataFrame({"core": cores, "lab": lab, "f": p_true.sum(1),
                       "g4": p_true[:, 1]})
    ct["wt"] = ct.lab * ct.f
    ct["wg"] = ct.lab * ct.g4
    cg = ct.groupby("core")[["wt", "wg"]].sum()
    cg["true"] = 100 * cg.wg / cg.wt.clip(lower=1e-9)
    cg = cg.reset_index()

    gss = GroupShuffleSplit(1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(Xa, groups=cores))
    sc = StandardScaler().fit(Xa[tr])
    lr = LogisticRegression(C=0.03, max_iter=1000, tol=1e-3,
                            class_weight="balanced").fit(
        sc.transform(Xa[tr]), hard[tr])

    # Arvaniti held-out cores
    p_te = lr.predict_proba(sc.transform(Xa[te]))
    df = pd.DataFrame({"core": cores[te], "w": lab[te],
                       "tumor": p_te.sum(1), "g4": p_te[:, 1]})
    df["wt"] = df.w * df.tumor
    df["wg"] = df.w * df.g4
    g = df.groupby("core")[["wt", "wg"]].sum()
    g["pred"] = 100 * g.wg / g.wt.clip(lower=1e-9)
    g = g.reset_index()
    g["true"] = g.core.map(dict(zip(cg.core, cg.true)))
    rows.append(metrics(g.pred.to_numpy(), g.true.to_numpy(),
                        "arvaniti_holdout"))

    # Arvaniti -> SICAP slides
    sic = np.load(ROOT / "features/sicap_c224/patch_features.npz")
    Xs = sic["emb_hoptimus0"].astype(np.float32)
    ys = sic["labels"].astype(int)
    ss = pd.Series(sic["slide_ids"].astype(str)).str.split("/").str[-1] \
        .to_numpy()
    p_s = lr.predict_proba(sc.transform(Xs))
    tmp = pd.DataFrame({"slide": ss, "tumor": p_s.sum(1), "g4": p_s[:, 1]})
    agg = tmp.groupby("slide").agg(tumor=("tumor", "sum"),
                                   g4=("g4", "sum")).reset_index()
    agg["pred"] = 100 * agg.g4 / agg.tumor.clip(lower=1e-9)
    sl = pd.read_csv(ROOT / "runs/sicap_gp4/slide_gp4_from_masks.csv")
    sl = sl[sl.tumour_px > 0]
    ext = agg.merge(sl[["slide", "gp4_frac"]], on="slide")
    ext["true"] = 100 * ext.gp4_frac
    rows.append(metrics(ext.pred.to_numpy(), ext.true.to_numpy(),
                        "arvaniti_to_sicap"))
    ext[["slide", "pred", "true"]].to_csv(OUT / "scores_arvaniti_to_sicap.csv",
                                          index=False)

    # SICAP -> Arvaniti cores
    k = ys >= 1
    sc2 = StandardScaler().fit(Xs[k])
    lr2 = LogisticRegression(C=0.1, max_iter=1000, tol=1e-3,
                             class_weight="balanced").fit(
        sc2.transform(Xs[k]), ys[k] - 1)
    p_a = lr2.predict_proba(sc2.transform(Xa))
    df = pd.DataFrame({"core": cores, "w": lab,
                       "tumor": p_a.sum(1), "g4": p_a[:, 1]})
    df["wt"] = df.w * df.tumor
    df["wg"] = df.w * df.g4
    ga = df.groupby("core")[["wt", "wg"]].sum()
    ga["pred"] = 100 * ga.wg / ga.wt.clip(lower=1e-9)
    ga = ga.reset_index()
    ga["true"] = ga.core.map(dict(zip(cg.core, cg.true)))
    rows.append(metrics(ga.pred.to_numpy(), ga.true.to_numpy(),
                        "sicap_to_arvaniti"))
    ga[["core", "pred", "true"]].to_csv(OUT / "scores_sicap_to_arvaniti.csv",
                                        index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "thresholds.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
