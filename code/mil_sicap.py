#!/usr/bin/env python3
"""Slide-level attention MIL on SICAPv2 with patient-grouped CV.

Tasks (multi-task):
  - ISUP grade group 1-5 (cross-entropy, class-weighted)
  - slide %GP4 from pixel masks (MSE on tumour slides, target 0-100)

Evaluation: QWK, macro-F1, AUROC ISUP>=2 and >=3; %GP4 MAE (pp),
Spearman, AUC at %GP4>=10%, near-threshold band (true<=20) MAE.
Out-of-fold predictions are saved.

Inputs: features/sicap_raw/patch_features.npz (emb_<enc>, labels, slide_ids)
        runs/sicap_gp4/slide_gp4_from_masks.csv (gp4_frac, tumour_px)
        SICAPv2/wsi_labels.xlsx (slide -> patient, Gleason grades)
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (cohen_kappa_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SICAP_ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")


class ABMIL(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, n_cls: int = 5,
                 dropout: float = 0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        self.att = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(),
                                 nn.Linear(128, 1))
        self.cls = nn.Linear(hidden, n_cls)
        self.reg = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.enc(x)
        a = torch.softmax(self.att(h).squeeze(-1), dim=0)
        z = (a[:, None] * h).sum(0)
        return self.cls(z), self.reg(z).squeeze(-1), a


def load_slides(enc: str, gp4_csv: Path):
    d = np.load(ROOT / "features/sicap_raw/patch_features.npz")
    emb = d[f"emb_{enc}"]
    slides = pd.Series(d["slide_ids"].astype(str)).str.split("/") \
        .str[-1].to_numpy()
    order = np.argsort(slides, kind="stable")
    emb, slides = emb[order], slides[order]
    uniq = np.unique(slides)
    idx = {s: np.where(slides == s)[0] for s in uniq}
    feats = {s: torch.from_numpy(emb[i]).float() for s, i in idx.items()}

    x = pd.read_excel(io.BytesIO(zipfile.ZipFile(SICAP_ZIP).read(
        "SICAPv2/wsi_labels.xlsx")))
    x["slide"] = x.slide_id.astype(str).str.lower()
    p, s_ = x.Gleason_primary, x.Gleason_secondary
    x["gg"] = np.where(p == 3, np.where(s_ == 3, 1, 2),
                       np.where(p == 4,
                                np.where(s_ == 3, 3, np.where(s_ == 4, 4, 5)),
                                5))
    g = pd.read_csv(gp4_csv)
    x = x.merge(g[["slide", "gp4_frac", "tumour_px", "n_patches"]],
                on="slide", how="left")
    x["gp4"] = (x.gp4_frac * 100).clip(0, 100)
    x = x[x.slide.isin(feats)].reset_index(drop=True)
    return x, feats


def sample_slide(feats, idx, rng, max_patches=200):
    f = feats[idx]
    if len(f) > max_patches:
        sel = rng.choice(len(f), max_patches, replace=False)
        f = f[np.sort(sel)]
    return f


def run_fold(tr, va, x, feats, dim, args, rng, seed):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ABMIL(dim, args.hidden, 5, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    cw = np.bincount(x.gg.values - 1, minlength=5).astype(float)
    cw = torch.tensor((cw.sum() / (5 * np.clip(cw, 1, None))),
                      dtype=torch.float, device=dev)
    ce = nn.CrossEntropyLoss(weight=cw)
    mse = nn.MSELoss()
    best, best_state = 1e18, None
    tumour = x.tumour_px.fillna(0).to_numpy() > 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in perm:
            f = sample_slide(feats, x.slide.iloc[i], rng).to(dev)
            y = torch.tensor([x.gg.iloc[i] - 1], device=dev)
            logits, r, _ = model(f)
            loss = ce(logits[None], y)
            if tumour[i]:
                tgt = torch.tensor([x.gp4.iloc[i] / 100.0],
                                   dtype=torch.float, device=dev)
                loss = loss + args.lam * mse(r[None], tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in va:
                f = feats[x.slide.iloc[i]].to(dev)
                y = torch.tensor([x.gg.iloc[i] - 1], device=dev)
                logits, r, _ = model(f)
                vl += float(ce(logits[None], y))
            vl /= len(va)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in
                                    model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in va:
            logits, r, _ = model(feats[x.slide.iloc[i]].to(dev))
            p = torch.softmax(logits, -1).cpu().numpy()
            preds.append((float(r.cpu()), p))
    prob = np.stack([p for _, p in preds])
    reg = np.array([r for r, _ in preds])
    return prob, reg


def metrics(x_va, prob, reg):
    gg = x_va.gg.to_numpy()
    pred = prob.argmax(1) + 1
    out = {"qwk": cohen_kappa_score(gg, pred, weights="quadratic"),
           "macro_f1": f1_score(gg, pred, average="macro")}
    for thr, name in ((2, "isup_ge2"), (3, "isup_ge3")):
        y = (gg >= thr).astype(int)
        if y.min() != y.max():
            out[f"auc_{name}"] = roc_auc_score(
                y, prob[:, thr - 1:].sum(1))
    t = x_va.tumour_px.fillna(0).to_numpy() > 0
    if t.sum() >= 5:
        true, pr = x_va.gp4.to_numpy()[t], reg[t] * 100
        out["gp4_mae"] = float(np.mean(np.abs(pr - true)))
        out["gp4_spearman"] = float(spearmanr(pr, true).statistic)
        y10 = (true >= 10).astype(int)
        if y10.min() != y10.max():
            out["gp4_auc10"] = roc_auc_score(y10, pr)
        band = true <= 20
        out["gp4_band20_mae"] = float(np.mean(np.abs(pr[band] - true[band])))
        out["n_tumour"] = int(t.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hoptimus0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/sicap_mil")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x, feats = load_slides(args.encoder,
                           ROOT / "runs/sicap_gp4/slide_gp4_from_masks.csv")
    dim = next(iter(feats.values())).shape[1]
    print(f"slides={len(x)} patients={x.patient_id.nunique()} dim={dim}",
          flush=True)

    all_rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                    random_state=seed)
        oof_gg = np.zeros(len(x), dtype=int) - 1
        oof_prob = np.zeros((len(x), 5))
        oof_reg = np.full(len(x), np.nan)
        for k, (tr, va) in enumerate(sgkf.split(
                x, x.gg, groups=x.patient_id)):
            prob, reg = run_fold(tr, va, x, feats, dim, args, rng, seed * 10 + k)
            oof_prob[va] = prob
            oof_reg[va] = reg
            oof_gg[va] = prob.argmax(1) + 1
            print(f"  seed{seed} fold{k} val={len(va)} "
                  f"qwk={cohen_kappa_score(x.gg.iloc[va], oof_gg[va], weights='quadratic'):.3f}",
                  flush=True)
        m = metrics(x, oof_prob, oof_reg)
        m["seed"] = seed
        all_rows.append(m)
        print(seed, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                for k, v in m.items()}), flush=True)
        pred = x[["slide", "patient_id", "gg", "gp4", "tumour_px"]].copy()
        pred["pred_gg"] = oof_gg
        pred["pred_gp4"] = oof_reg * 100
        for j in range(5):
            pred[f"p{j + 1}"] = oof_prob[:, j]
        pred.to_csv(args.out / f"oof_{args.encoder}_seed{seed}.csv",
                    index=False)

    res = pd.DataFrame(all_rows)
    res.to_csv(args.out / f"summary_{args.encoder}.csv", index=False)
    keys = [c for c in res.columns if c != "seed"]
    print("\nmean:", json.dumps({c: round(float(res[c].mean()), 3)
                                 for c in keys}))


if __name__ == "__main__":
    main()
