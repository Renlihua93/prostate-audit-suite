#!/usr/bin/env python3
"""Auxiliary detection multi-task regularisation for cross-cohort grading.

Heads on frozen features: tile 4-class (NC/G3/G4/G5), binary cancer
(status > 0), attention MIL slide head (ISUP 1-5). Modes:
  base = tile CE + slide MIL
  aux  = base + lambda * BCE(cancer)
Evaluation: patient-grouped SICAP split; PAR/TCGA AUC (mean tile P(G4+)).
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from mgod_pilot import load_sicap
from mgod_loco import load_slides

ROOT = Path(__file__).resolve().parents[1]


class Head(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Dropout(0.2))
        self.tile = nn.Linear(hidden, 4)
        self.cancer = nn.Linear(hidden, 1)
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.slide = nn.Linear(hidden, 5)

    def forward(self, x):
        h = self.trunk(x)
        a = torch.softmax(self.att(h).squeeze(-1), dim=0)
        z = (a[:, None] * h).sum(0)
        return self.tile(h), self.cancer(h).squeeze(-1), self.slide(z[None])[0]


def train(emb, labels, slides, gg, lam_aux, epochs, seed):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(emb.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    cancer = (labels > 0).astype(np.float32)
    idx_all = np.where(labels >= 0)[0]
    sl_ids = np.array(sorted(gg))
    tiles_by = {s: emb[slides == s] for s in sl_ids}
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(idx_all)
        for b in range(0, len(perm), 256):
            idx = perm[b:b + 256]
            x = torch.from_numpy(emb[idx]).to(dev)
            yt = torch.from_numpy(labels[idx]).long().to(dev)
            yc = torch.from_numpy(cancer[idx]).to(dev)
            lt, lc, _ = model(x)
            loss = ce(lt, yt)
            if lam_aux > 0:
                loss = loss + lam_aux * bce(lc, yc)
            s = sl_ids[rng.integers(len(sl_ids))]
            xf = torch.from_numpy(tiles_by[s]).to(dev)
            _, _, ls = model(xf)
            loss = loss + 0.5 * ce(ls[None], torch.tensor([gg[s] - 1],
                                                          device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def slide_scores(model, feats, dev):
    out = {}
    with torch.no_grad():
        for s, x in feats.items():
            lt, _, _ = model(torch.from_numpy(x).to(dev))
            p = torch.softmax(lt, -1)
            out[s] = float(p[:, 2:].sum(1).mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    keep = labels >= 0
    emb, labels, slides = emb[keep], labels[keep], slides[keep]

    sites = {}
    for name, d in (("PAR", "features/phikon" if args.encoder == "phikon"
                     else "features/hoptimus0_par"),
                    ("TCGA", "features/tcga_phikon" if args.encoder == "phikon"
                     else "features/hoptimus0_tcga")):
        if name == "PAR":
            par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
            lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
            lab["slide"] = lab.slide_id.astype(str).str.upper()
            r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index(
                "slide")
            maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                          if x.notna().any() else np.nan, axis=1)
            par = par.merge(maj.rename("maj"), left_on="slide",
                            right_index=True).dropna(subset=["maj"])
            lab_df = pd.DataFrame({"slide": par.slide,
                                   "y": (par.maj >= 2).astype(int)})
        else:
            tc = pd.read_csv(ROOT / "data/refined_tcga_prad_matched.csv")
            tc["slide"] = tc.slide_name.astype(str).str.lower().str.replace(
                ".svs", "", regex=False)
            tc = tc.dropna(subset=["GG"]).drop_duplicates("slide")
            lab_df = pd.DataFrame({"slide": tc.slide,
                                   "y": (tc.GG.astype(int) >= 2).astype(int)})
        feats, y = load_slides(ROOT / d, lab_df)
        sites[name] = (feats, y)

    slides_uniq = np.array(sorted(gg))
    patients = np.array([pat[s] for s in slides_uniq])
    rows = []
    for lam in (0.0, 0.5):
        for seed in args.seeds:
            tr, te = next(GroupShuffleSplit(1, test_size=0.2,
                                            random_state=seed).split(
                slides_uniq, groups=patients))
            g_tr, g_te = set(slides_uniq[tr]), set(slides_uniq[te])
            mask = np.array([slides[i] in g_tr for i in range(len(labels))])
            model = train(emb[mask], labels[mask], slides[mask],
                          {s: gg[s] for s in g_tr}, lam, args.epochs, seed)
            model.eval()
            row = {"lambda": lam, "seed": seed}
            for name, (feats, ymap) in sites.items():
                sc = slide_scores(model, feats, dev)
                yy = np.array([ymap[s] for s in ymap])
                ss = np.array([sc[s] for s in ymap])
                row[name] = float(roc_auc_score(yy, ss))
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("lambda").agg({c: ["mean", "std"] for c in ("PAR",
                                                                 "TCGA")})
    print(summ.round(3).to_string())
    out = ROOT / f"runs/aux_det_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main()
