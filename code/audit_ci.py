#!/usr/bin/env python3
"""Bootstrap CIs for the audit effects and H-optimus-0 replication.

- Hyperparameter sensitivity and alignment sign, for both encoders, with
  paired slide-level bootstrap CIs (2000 resamples).
- Certification-rate CIs (Clopper-Pearson) and tie-rule CI.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pitfall_matrix import load_par_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/audit_ci"
OUT.mkdir(parents=True, exist_ok=True)


def cp_ci(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def boot_delta(s_a, s_b, y, rng, n_boot=2000):
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s_a[idx])
                   - roc_auc_score(y[idx], s_b[idx]))
    out = np.array(out)
    return float(np.mean(out)), float(np.percentile(out, 2.5)), \
        float(np.percentile(out, 97.5))


def hyper_scores(enc):
    Xs, ys, (Xp, yp), (Xt, yt) = load_par_scores(enc)
    res = {}
    for tag, scaler, C in (("tuned", True, 0.01), ("fixed", False, 0.001)):
        F = StandardScaler().fit(Xs).transform if scaler else (lambda x: x)
        lr = LogisticRegression(C=C, max_iter=1500,
                                class_weight="balanced").fit(F(Xs), ys)
        idx = [i for i, c in enumerate(lr.classes_) if c >= 2]
        res[tag] = {name: lr.predict_proba(F(X))[:, idx].sum(1)
                    for name, X in (("PAR", Xp), ("TCGA", Xt))}
    return res, (yp, yt)


def align_scores(enc):
    """Tile-level centering (matches the original claim), per slide means."""
    import pandas as pd
    from pathlib import Path
    sic_path = (ROOT / "features/sicap_phikon_c224/patch_features.npz"
                if enc == "phikon" else
                ROOT / "features/sicap_c224/patch_features.npz")
    par_dir = {"phikon": "features/phikon",
               "hoptimus0": "features/hoptimus0_par",
               "resnet50": "features/resnet50_par"}[enc]
    tcga_dir = {"phikon": "features/tcga_phikon",
                "hoptimus0": "features/hoptimus0_tcga",
                "resnet50": "features/resnet50_tcga"}[enc]
    sic = np.load(sic_path)
    keep = sic["labels"] >= 0
    Xs, ys = sic[f"emb_{enc}"][keep], sic["labels"][keep]
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(C=0.01, max_iter=1500,
                            class_weight="balanced").fit(sc.transform(Xs), ys)
    idx = [i for i, c in enumerate(lr.classes_) if c >= 2]

    def load_slides(dirname):
        feats = {}
        for f in sorted(Path(dirname).glob("*.npz")):
            z = np.load(f)
            key = f.stem.lower().replace("_leica", "").replace(
                "_grundium", "")
            feats[key] = z["tile_emb"].astype(np.float32)
        return feats

    # labels
    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="	")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    par = par.merge(maj.rename("maj"), left_on="slide", right_index=True)
    par = par.dropna(subset=["maj"])
    yp_map = {k.lower(): int(v >= 2) for k, v in
              zip(par.slide, par.maj)}
    tc = pd.read_csv(ROOT / "data/refined_tcga_prad_matched.csv")
    tc["slide"] = tc.slide_name.astype(str).str.lower().str.replace(
        ".svs", "", regex=False)
    tc = tc.dropna(subset=["GG"]).drop_duplicates("slide")
    yt_map = {s.lower(): int(g >= 2) for s, g in zip(tc.slide, tc.GG)}

    out = {}
    for name, dirname, ymap in (
            ("PAR", ROOT / par_dir, yp_map),
            ("TCGA", ROOT / tcga_dir, yt_map)):
        feats = load_slides(dirname)
        slides = [s for s in feats if s in ymap]
        all_tiles = np.concatenate([feats[s] for s in slides])
        tsc = StandardScaler().fit(all_tiles)
        zero, cent, y = [], [], []
        for s in slides:
            x = feats[s]
            p0 = lr.predict_proba(sc.transform(x))[:, idx].sum(1).mean()
            p1 = lr.predict_proba(tsc.transform(x))[:, idx].sum(1).mean()
            zero.append(p0)
            cent.append(p1)
            y.append(ymap[s])
        out[name] = (np.array(zero), np.array(cent), np.array(y))
    return out


def main():
    rng = np.random.default_rng(0)
    rows = []
    for enc in ("phikon", "hoptimus0", "resnet50"):
        hs, (yp, yt) = hyper_scores(enc)
        for name, y in (("PAR", yp), ("TCGA", yt)):
            d, lo, hi = boot_delta(hs["fixed"][name], hs["tuned"][name], y,
                                   rng)
            rows.append({"analysis": "hyperparameter", "encoder": enc,
                         "target": name, "delta": d, "lo": lo, "hi": hi})
        al = align_scores(enc)
        for name in ("PAR", "TCGA"):
            zero, cent, y = al[name]
            d, lo, hi = boot_delta(cent, zero, y, rng)
            rows.append({"analysis": "alignment", "encoder": enc,
                         "target": name, "delta": d, "lo": lo, "hi": hi})
    # certification rates
    naive = pd.read_csv(ROOT / "runs/active_acq_phikon/simulation.csv")
    naive = naive[(naive.strategy == "uncertainty") & (naive.K == 80)
                  & (naive.alpha == 0.10)]
    audit = pd.read_csv(ROOT / "runs/audit_active_phikon/simulation.csv")
    audit = audit[(audit.arm == "uncertainty_fit") & (audit.K == 80)]
    for tag, d in (("naive", naive), ("audit", audit)):
        k = int(d.feasible.sum())
        n = len(d)
        lo, hi = cp_ci(k, n)
        rows.append({"analysis": "certification", "encoder": "phikon",
                     "target": tag, "delta": k / n, "lo": lo, "hi": hi,
                     "n": n})
    # tie flips (decision-level subset)
    lo, hi = cp_ci(3, 264)
    rows.append({"analysis": "tie_flips", "encoder": "phikon",
                 "target": "PAR_features", "delta": 3 / 264, "lo": lo,
                 "hi": hi, "n": 264})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ci.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
