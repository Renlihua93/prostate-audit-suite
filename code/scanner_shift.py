#!/usr/bin/env python3
"""Scanner-shift supplementary analysis using PAR Leica/Grundium rescan pairs."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NORM = lambda x: str(x).lower().replace("_leica", "").replace(
    "_grundium", "")


def load_dir(dirname):
    out = {}
    for f in sorted(Path(dirname).glob("*.npz")):
        z = np.load(f)
        out[NORM(f.stem)] = z["tile_emb"].astype(np.float32)
    return out


def main():
    leica = load_dir(ROOT / "features/phikon")
    grund = load_dir(ROOT / "features/phikon_grundium")
    pairs = sorted(set(leica) & set(grund))
    print(f"paired slides: {len(pairs)}", flush=True)

    sic = np.load(ROOT / "features/sicap_phikon_c224/patch_features.npz")
    keep = sic["labels"] >= 0
    Xs, ys = sic["emb_phikon"][keep], sic["labels"][keep]
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(C=0.01, max_iter=1500,
                            class_weight="balanced").fit(sc.transform(Xs), ys)
    idx = [i for i, c in enumerate(lr.classes_) if c >= 2]

    def slide_score(tiles):
        return float(lr.predict_proba(sc.transform(tiles))[:, idx].sum(1).mean())

    # embedding similarity
    means_l = np.stack([leica[s].mean(0) for s in pairs])
    means_g = np.stack([grund[s].mean(0) for s in pairs])
    cos_same = np.sum(means_l * means_g, 1) / (
        np.linalg.norm(means_l, axis=1) * np.linalg.norm(means_g, axis=1))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    while np.any(perm == np.arange(len(pairs))):
        perm = rng.permutation(len(pairs))
    cos_diff = np.sum(means_l * means_g[perm], 1) / (
        np.linalg.norm(means_l, axis=1)
        * np.linalg.norm(means_g[perm], axis=1))
    print(f"same-slide cross-scanner cosine: {cos_same.mean():.3f} "
          f"(cross-slide {cos_diff.mean():.3f})", flush=True)

    # labels
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.lower()
    maj = lab.set_index("slide")[["isup_1", "isup_r2", "isup_r3"]] \
        .apply(lambda x: np.nanmedian(x.dropna().astype(float))
               if x.notna().any() else np.nan, axis=1)
    ymap = {s: int(v >= 2) for s, v in maj.items() if pd.notna(v)}

    rows = []
    for s in pairs:
        if s not in ymap:
            continue
        rows.append({"slide": s, "y": ymap[s],
                     "leica": slide_score(leica[s]),
                     "grundium": slide_score(grund[s])})
    df = pd.DataFrame(rows)
    print(f"labeled pairs: {len(df)}", flush=True)
    if len(df) >= 10 and df.y.nunique() == 2:
        auc_l = roc_auc_score(df.y, df.leica)
        auc_g = roc_auc_score(df.y, df.grundium)
        d = auc_g - auc_l
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(df), len(df))
            if df.y.iloc[i].nunique() < 2:
                continue
            boot.append(roc_auc_score(df.y.iloc[i], df.grundium.iloc[i])
                        - roc_auc_score(df.y.iloc[i], df.leica.iloc[i]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        corr = float(np.corrcoef(df.leica, df.grundium)[0, 1])
        print(f"AUC Leica {auc_l:.3f}, Grundium {auc_g:.3f}, "
              f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}], "
              f"score correlation {corr:.3f}", flush=True)
        df.to_csv(ROOT / "runs/scanner_shift_scores.csv", index=False)


if __name__ == "__main__":
    main()
