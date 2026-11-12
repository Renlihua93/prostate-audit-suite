#!/usr/bin/env python3
"""Additional experiments/figures: hyperparameter curve, calibration before/after,
K-shot triage curves, domain distance, alpha x K heatmap."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
import glob
import tiffslide  # noqa: F401  (kept for parity with env)

from pitfall_matrix import load_par_scores

ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "figures"
OUT = ROOT / "runs/extra_experiments"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B2"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def hp_curve():
    cs = np.logspace(-4, 0, 9)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows = []
    for ax, enc in zip(axes, ("phikon", "hoptimus0")):
        Xs, ys, (Xp, yp), (Xt, yt) = load_par_scores(enc)
        for tag, target, y, color, mk in (
                ("PAR", "PAR", yp, C[0], "o"), ("TCGA", "TCGA", yt, C[1], "s")):
            X = Xp if target == "PAR" else Xt
            aucs = []
            for c in cs:
                sc = StandardScaler().fit(Xs)
                lr = LogisticRegression(C=c, max_iter=1500,
                                        class_weight="balanced").fit(
                    sc.transform(Xs), ys)
                idx = [i for i, cc in enumerate(lr.classes_) if cc >= 2]
                aucs.append(roc_auc_score(
                    y, lr.predict_proba(sc.transform(X))[:, idx].sum(1)))
                rows.append({"encoder": enc, "target": target, "C": c,
                             "auc": aucs[-1]})
            ax.semilogx(cs, aucs, mk + "-", color=color, lw=1.2,
                        label=target)
        ax.set_title(enc)
        ax.set_xlabel("Regularisation C (standardised features)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("AUC (ISUP/GG >= 2)")
    fig.suptitle("Continuous hyperparameter sensitivity", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "S_hyperparameter_curve.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "hyperparameter_curve.csv", index=False)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def temperature(p, y):
    def nll(T):
        q = np.clip(1 / (1 + np.exp(-logit(p) / T)), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def calibration_curves():
    from clinical_utility import site_data
    site = site_data("phikon")["PAR"]
    tab, feats, _ = site
    keys = tab.slide.astype(str).str.lower().str.replace(
        "_leica", "", regex=False)
    X = np.stack([feats[k].mean(0) for k in keys])
    y = tab.y.to_numpy()
    rng = np.random.default_rng(0)
    raw_acc = np.zeros((10, 2)); cal_acc = np.zeros((10, 2))
    raw_cnt = np.zeros(10); cal_cnt = np.zeros(10)
    for rep in range(50):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cal = np.concatenate([rng.choice(pos, 30, replace=False),
                              rng.choice(neg, 30, replace=False)])
        test = np.setdiff1d(np.arange(len(y)), cal)
        sc = StandardScaler().fit(X[cal])
        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight="balanced").fit(
            sc.transform(X[cal]), y[cal])
        # cross-fit calibration probabilities for temperature
        oof = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, va in skf.split(X[cal], y[cal]):
            s2 = StandardScaler().fit(X[cal][tr])
            m2 = LogisticRegression(C=0.1, max_iter=1000,
                                    class_weight="balanced").fit(
                s2.transform(X[cal][tr]), y[cal][tr])
            oof[cal[va]] = m2.predict_proba(s2.transform(X[cal][va]))[:, 1]
        T = temperature(oof[cal], y[cal])
        p = lr.predict_proba(sc.transform(X[test]))[:, 1]
        pc = 1 / (1 + np.exp(-logit(p) / T))
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        idxc = np.digitize(pc, bins) - 1
        for b in range(10):
            if (idx == b).any():
                raw_acc[b] += [p[idx == b].sum(),
                               y[test][idx == b].sum()]
                raw_cnt[b] += (idx == b).sum()
            if (idxc == b).any():
                cal_acc[b] += [pc[idxc == b].sum(),
                               y[test][idxc == b].sum()]
                cal_cnt[b] += (idxc == b).sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([0, 1], [0, 1], ls="--", color=C[4], lw=0.8)
    for acc, cnt, lab, col in ((raw_acc, raw_cnt, "Raw", C[0]),
                               (cal_acc, cal_cnt, "Temperature-scaled", C[1])):
        ok = cnt > 0
        ax.plot(acc[ok, 0] / cnt[ok], acc[ok, 1] / cnt[ok], "o-",
                color=col, lw=1.2, ms=4, label=lab)
    ax.set_xlabel("Predicted P(ISUP >= 2)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability before/after temperature scaling", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "S_calibration_before_after.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def kshot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for enc, col in (("phikon", C[0]), ("hoptimus0", C[1])):
        frames = []
        for f in (ROOT / f"runs/label_budget/{enc}_tile_plus_slide/simulation.csv",
                  ROOT / f"runs/clinical_utility_{enc}/simulation.csv"):
            if f.exists():
                frames.append(pd.read_csv(f))
        d = pd.concat(frames, ignore_index=True)
        d = d[d.alpha == 0.10]
        feas = d.groupby("K").feasible.mean()
        axes[0].plot(feas.index, 100 * feas.values, "o-", color=col,
                     label=enc)
        ok = d[d.feasible]
        g = ok.groupby("K").agg(cleared=("cleared", "mean"),
                                sens=("sens_detected", "mean"))
        axes[1].plot(g.index, g.cleared, "o--", color=col, label=enc + " cleared")
        axes[1].plot(g.index, g.sens, "s:", color=col, label=enc + " sens")
    axes[0].set_xlabel("Site labels K")
    axes[0].set_ylabel("Certification rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].set_title("Feasibility versus site labels", fontsize=8.5)
    axes[1].set_xlabel("Site labels K")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Conditional clearing and sensitivity", fontsize=8.5)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "S_kshot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def domain_distance():
    rng = np.random.default_rng(0)
    def load_sicap(enc):
        p = (ROOT / "features/sicap_phikon_c224/patch_features.npz"
             if enc == "phikon" else ROOT / "features/sicap_c224/patch_features.npz")
        return np.load(p)[f"emb_{enc}"].astype(np.float32)
    def load_slides(dirname):
        return np.concatenate([np.load(f)["tile_emb"].astype(np.float32)
                               for f in glob.glob(str(Path(dirname) / "*.npz"))])
    def load_ppbench(enc):
        p = (ROOT / "features/ppbench_phikon/baseline/patch_features.npz"
             if enc == "phikon" else ROOT / "features/ppbench/baseline/patch_features.npz")
        return np.load(p)[f"emb_{enc}"].astype(np.float32)
    def load_peso(enc):
        return np.concatenate([np.load(f)["tile_emb"].astype(np.float32)
                               for f in glob.glob(str(ROOT / f"features/peso_{enc}/*.npz"))])
    def energy(a, b, n=600):
        a = a[rng.choice(len(a), n, replace=False)].astype(np.float64)
        b = b[rng.choice(len(b), n, replace=False)].astype(np.float64)
        return 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()
    rows = []
    for enc in ("phikon", "hoptimus0"):
        s = load_sicap(enc)
        targets = {
            "PAR": load_slides(ROOT / ("features/phikon" if enc == "phikon"
                                       else "features/hoptimus0_par")),
            "TCGA": load_slides(ROOT / ("features/tcga_phikon" if enc == "phikon"
                                        else "features/hoptimus0_tcga")),
            "PPBench": load_ppbench(enc),
            "PESO": load_peso(enc),
        }
        for k, v in targets.items():
            rows.append({"encoder": enc, "target": k, "energy": energy(s, v)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "domain_distance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    order = ["PAR", "PPBench", "TCGA", "PESO"]
    x = np.arange(len(order))
    for ax, enc, col in ((axes[0], "phikon", C[0]),
                         (axes[1], "hoptimus0", C[1])):
        vals = [df[(df.encoder == enc) & (df.target == t)].energy.mean()
                for t in order]
        ax.bar(x, vals, 0.55, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(enc, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Energy distance to SICAPv2")
    fig.tight_layout()
    fig.savefig(FIG / "S_domain_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def alpha_k_heatmap():
    d = pd.read_csv(ROOT / "runs/clinical_utility_phikon/simulation.csv")
    ok = d[d.feasible]
    for key, name in (("cleared", "cleared"), ("missed_per100", "missed")):
        g = ok.groupby(["alpha", "K"])[key].mean().unstack()
        fig, ax = plt.subplots(figsize=(4.3, 3.0))
        im = ax.imshow(g.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels([str(c) for c in g.columns])
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels([str(i) for i in g.index])
        ax.set_xlabel("K site labels")
        ax.set_ylabel("alpha")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < np.nanmax(g.values) * 0.6 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(key, fontsize=7)
        ax.set_title(f"Deployment {name} (Phikon-v1)", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"S_heatmap_{name}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    hp_curve()
    calibration_curves()
    kshot_curves()
    domain_distance()
    alpha_k_heatmap()
    print("extra experiments written", OUT)


if __name__ == "__main__":
    main()
