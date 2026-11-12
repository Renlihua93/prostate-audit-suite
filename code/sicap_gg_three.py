#!/usr/bin/env python3
"""4-class Gleason-pattern classification on NormalizedSICAPv2 / CrowdGleason.

Patches are 512 px JPEGs labelled one-hot NC/G3/G4/G5 (plus G4C cribriform
flag). The official patient-level train/val/test split is used: C is picked on
val, then train+val is refit and evaluated once on test.

Feature extraction runs once per encoder and is cached under runs/sicap_gg/.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fm_encoders import load_encoder

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(f"{DATA_ROOT}/crowdgleason")
IMG_DIR = DATA / "sicapv2_images"
CLASSES = ["NC", "G3", "G4", "G5"]


def ensure_images() -> None:
    if IMG_DIR.exists() and len(list(IMG_DIR.glob("*.jpg"))) > 18000:
        return
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA / "NormalizedSICAPv2.zip") as z:
        z.extractall(IMG_DIR)


def make_clf(C, dim, seed=42):
    return Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=min(128, dim), random_state=seed)),
        ("lr", LogisticRegression(C=C, max_iter=2000, tol=1e-3,
                                  class_weight="balanced")),
    ])


def embed_all(encoder_name: str, names: list[str], batch: int, cache: Path):
    if cache.exists():
        z = np.load(cache)
        if list(z["names"]) == names:
            print(f"  loaded {cache.name} {z['emb'].shape}")
            return z["emb"]
    enc = load_encoder(encoder_name)
    paths = [IMG_DIR / n for n in names]
    embs = []
    import time
    t0 = time.time()
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p) for p in paths[i:i + batch]]
        embs.append(enc.embed(imgs))
        for im in imgs:
            im.close()
        if i % (batch * 50) == 0:
            print(f"  {i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    emb = np.concatenate(embs)
    np.savez_compressed(cache, emb=emb, names=np.array(names))
    del enc
    import torch
    torch.cuda.empty_cache()
    return emb


def binary_scores(df_test, proba, classes):
    out = {}
    idx = {int(c): i for i, c in enumerate(classes)}

    def col(k):
        return proba[:, idx[k]] if k in idx else np.zeros(len(proba))

    tumour = 1 - col(0)
    lab_tum = (df_test.label != 0).to_numpy()
    if lab_tum.min() != lab_tum.max():
        out["auroc_tumour"] = float(roc_auc_score(lab_tum, tumour))
    g4p = col(2) + col(3)
    lab_g4 = df_test.label.isin([2, 3]).to_numpy()
    if lab_g4.min() != lab_g4.max():
        out["auroc_g4plus"] = float(roc_auc_score(lab_g4, g4p))
    if "G4C" in df_test:
        lab_c = df_test.G4C.to_numpy().astype(int)
        if lab_c.min() != lab_c.max():
            out["auroc_g4c"] = float(roc_auc_score(lab_c, g4p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path,
                    default=DATA / "sicap_annotations_all.csv")
    ap.add_argument("--encoders", nargs="+",
                    default=["phikon_v2", "hoptimus0", "resnet50"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs/sicap_gg")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--C-grid", nargs="+", type=float,
                    default=[0.01, 0.03, 0.1, 0.3, 1.0])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_images()

    df = pd.read_csv(args.annotations)
    df["label"] = df[CLASSES].to_numpy().argmax(axis=1)
    names = df.image_name.tolist()

    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    te = (df.split == "test").to_numpy()
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"cases={df.case.nunique() if 'case' in df else 'n/a'}")

    results = []
    for name in args.encoders:
        print(f"== {name} ==")
        emb = embed_all(name, names, args.batch, args.out / f"feat_{name}.npz")
        dim = emb.shape[1]
        best = None
        for C in args.C_grid:
            clf = make_clf(C, dim)
            clf.fit(emb[tr], df.label[tr])
            pred = clf.predict(emb[va])
            f1 = f1_score(df.label[va], pred, average="macro")
            if best is None or f1 > best[1]:
                best = (C, f1)
        C = best[0]
        clf = make_clf(C, dim)
        trva = tr | va
        clf.fit(emb[trva], df.label[trva])
        proba = clf.predict_proba(emb[te])
        pred = clf.predict(emb[te])
        y = df.label[te].to_numpy()
        res = {
            "encoder": name, "C": C, "val_macro_f1": float(best[1]),
            "test_accuracy": float(accuracy_score(y, pred)),
            "test_macro_f1": float(f1_score(y, pred, average="macro")),
            "test_qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        }
        res.update(binary_scores(df[te].reset_index(drop=True), proba,
                                 list(clf.classes_)))
        cm = confusion_matrix(y, pred, labels=list(clf.classes_))
        results.append(res)
        pd.DataFrame({"image_name": df.image_name[te].to_numpy(),
                      "y": y, "pred": pred,
                      **{f"p{c}": proba[:, list(clf.classes_).index(c)]
                         if c in clf.classes_ else 0.0
                         for c in range(len(CLASSES))}}).to_csv(
            args.out / f"test_pred_{name}.csv", index=False)
        pd.DataFrame(cm).to_csv(args.out / f"confusion_{name}.csv", index=False)
        print(json.dumps(res, indent=2), flush=True)

    (args.out / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\n== test summary ==")
    for r in results:
        print(f"{r['encoder']:>10}  acc={r['test_accuracy']:.3f} "
              f"F1={r['test_macro_f1']:.3f} QWK={r['test_qwk']:.3f} "
              f"AUROC(tum)={r.get('auroc_tumour', float('nan')):.3f} "
              f"AUROC(G4+)={r.get('auroc_g4plus', float('nan')):.3f} "
              f"AUROC(G4C)={r.get('auroc_g4c', float('nan')):.3f}")


if __name__ == "__main__":
    main()
