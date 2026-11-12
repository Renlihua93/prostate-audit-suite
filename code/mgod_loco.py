#!/usr/bin/env python3
"""MGOD leave-one-cohort-out / multi-source validation.

Modes (binary slide target ISUP/GG >= 2 everywhere):
  A tile_only        SICAP tile-CE only, no slide supervision
  B sicap_mil        SICAP tile-CE + SICAP slide BCE
  C +PAR_mil         B + PAR slide BCE (weak source), evaluated on TCGA + SICAP
  D +TCGA_mil        B + TCGA slide BCE (weak source), evaluated on PAR + SICAP
  E par_mil_only     PAR slide BCE only (no tile labels), evaluated on TCGA + SICAP
  F tcga_mil_only    TCGA slide BCE only, evaluated on PAR + SICAP

Evaluation: mean tile P(G4+) per slide, AUC for ISUP/GG>=2.
Runs for Phikon-v1 and H-optimus-0, 3 seeds, patient-grouped SICAP split.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SICAP_ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")


class Head(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Dropout(0.2))
        self.tile = nn.Linear(hidden, 4)
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.slide = nn.Linear(hidden, 5)
        self.weak = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.tile(h)
        a = torch.softmax(self.att(h).squeeze(-1), dim=0)
        z = (a[:, None] * h).sum(0)
        return logits, self.slide(z[None])[0], self.weak(z[None])[0]


def load_sicap(encoder):
    path = (ROOT / "features/sicap_phikon_c224/patch_features.npz"
            if encoder == "phikon"
            else ROOT / "features/sicap_c224/patch_features.npz")
    d = np.load(path)
    emb = d[f"emb_{encoder}"].astype(np.float32)
    labels = d["labels"].astype(int)
    slides = pd.Series(d["slide_ids"].astype(str)).str.split("/").str[-1] \
        .to_numpy()
    x = pd.read_excel(io.BytesIO(zipfile.ZipFile(SICAP_ZIP).read(
        "SICAPv2/wsi_labels.xlsx")))
    x["slide"] = x.slide_id.astype(str).str.lower()
    p, s = x.Gleason_primary, x.Gleason_secondary
    x["gg"] = np.where(p == 3, np.where(s == 3, 1, 2),
                       np.where(p == 4,
                                np.where(s == 3, 3, np.where(s == 4, 4, 5)),
                                5))
    gg = dict(zip(x.slide, x.gg))
    pat = dict(zip(x.slide, x.patient_id.astype(str)))
    return emb, labels, slides, gg, pat


def load_slides(feats_dir, labels):
    feats, y = {}, {}
    for f in sorted(Path(feats_dir).glob("*.npz")):
        key = f.stem.lower().replace("_leica", "").replace("_grundium", "")
        feats[key] = np.load(f)["tile_emb"].astype(np.float32)
    for _, r in labels.iterrows():
        s = str(r.slide).lower()
        if s in feats:
            y[s] = int(r.y)
    return feats, y


def slide_score(model, tiles, dev):
    with torch.no_grad():
        lt, _, _ = model(torch.from_numpy(tiles).to(dev))
        return float(torch.softmax(lt, -1)[:, 2:].sum(1).mean().item())


def auc_slide(model, feats, y, dev):
    sc = np.array([slide_score(model, feats[s], dev) for s in y])
    return roc_auc_score(np.array(list(y.values())), sc)


def train(cohorts, feats_by, s_emb, s_lab, s_slides, s_gg, dev,
          params, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(s_emb.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    idx_all = np.where(s_lab >= 0)[0]
    weak = [(name, s, y) for name in cohorts if name != "sicap"
            for s, y in cohorts[name].items()]
    sicap_slides = None
    if "sicap" in cohorts and cohorts["sicap"]:
        gg_by = cohorts["sicap"]
        sicap_slides = [(s, int(g)) for s, g in gg_by.items()]
        sicap_slides = [(s, g) for s, g in sicap_slides if g >= 1]
    for ep in range(params["epochs"]):
        model.train()
        perm = rng.permutation(idx_all)
        for b in range(0, len(perm), params["batch"]):
            idx = perm[b:b + params["batch"]]
            if params["tile_ce"]:
                x = torch.from_numpy(s_emb[idx]).to(dev)
                yt = torch.from_numpy(s_lab[idx]).long().to(dev)
                lt, _, _ = model(x)
                loss = ce(lt, yt)
            else:
                loss = None
            if params["lam"] > 0 and sicap_slides:
                s5, g5 = sicap_slides[rng.integers(len(sicap_slides))]
                xf = torch.from_numpy(feats_by["sicap"][s5]).to(dev)
                _, ls5, _ = model(xf)
                l_s = ce(ls5[None], torch.tensor([g5 - 1], device=dev))
                loss = l_s if loss is None else loss + params["lam"] * l_s
            if params["lam"] > 0 and weak:
                name, s, y = weak[rng.integers(len(weak))]
                xf = torch.from_numpy(feats_by[name][s]).to(dev)
                _, _, lw = model(xf)
                l_w = params["lam"] * bce(
                    lw, torch.tensor([float(y)], device=dev))
                loss = l_w if loss is None else loss + params["lam"] * l_w
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    emb, labels, slides, gg, pat = load_sicap(args.encoder)
    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    par = par.merge(maj.rename("maj"), left_on="slide", right_index=True)
    par = par.dropna(subset=["maj"])
    par_dir = (ROOT / "features/phikon" if args.encoder == "phikon"
               else ROOT / "features/hoptimus0_par")
    par_f, par_y = load_slides(par_dir, pd.DataFrame(
        {"slide": par.slide, "y": (par.maj >= 2).astype(int)}))
    tc = pd.read_csv(ROOT / "runs/tcga_external/tcga_scores.csv")
    tc_dir = (ROOT / "features/tcga_phikon" if args.encoder == "phikon"
              else ROOT / "features/hoptimus0_tcga")
    tc_f, tc_y = load_slides(tc_dir, pd.DataFrame(
        {"slide": tc.slide, "y": (tc.gg.astype(int) >= 2).astype(int)}))

    slides_uniq = np.array(sorted(gg))
    patients = np.array([pat[s] for s in slides_uniq])
    res = {}
    for seed in args.seeds:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2,
                                        random_state=seed).split(
            slides_uniq, groups=patients))
        g_tr, g_te = set(slides_uniq[tr]), set(slides_uniq[te])
        mask_tr = np.array([slides[i] in g_tr for i in range(len(labels))])
        emb_tr, lab_tr, sl_tr = emb[mask_tr], labels[mask_tr], slides[mask_tr]
        gg_tr = {s: gg[s] for s in gg if s in g_tr}
        g_te_list = sorted(g_te)

        def eval_sicap(model):
            sc = []
            yy = []
            for s in g_te_list:
                sc.append(slide_score(model, emb[slides == s], dev))
                yy.append(int(gg[s] >= 2))
            return roc_auc_score(yy, sc)

        modes = {
            "A_tile_only": {},
            "B_sicap_mil": {"sicap": gg_tr},
            "C_plus_PAR_mil": {"sicap": gg_tr, "PAR": par_y},
            "D_plus_TCGA_mil": {"sicap": gg_tr, "TCGA": tc_y},
            "E_par_mil_only": {"sicap": {}, "PAR": par_y},
            "F_tcga_mil_only": {"sicap": {}, "TCGA": tc_y},
        }
        feats_by = {"sicap": {s: emb[slides == s] for s in gg_tr},
                    "PAR": par_f, "TCGA": tc_f}
        for mode, cohorts in modes.items():
            lam = 0.0 if mode == "A_tile_only" else 0.5
            tile_ce = mode not in ("E_par_mil_only", "F_tcga_mil_only")
            params = {"epochs": args.epochs, "batch": 256, "lam": lam,
                      "tile_ce": tile_ce}
            # weak-label pools must contain only slides with features
            pools = {k: {s: y for s, y in v.items() if s in feats_by[k]}
                     for k, v in cohorts.items()}
            model = train(pools, feats_by, emb_tr, lab_tr, sl_tr, gg_tr,
                          dev, params, seed)
            model.eval()
            row = {"mode": mode, "seed": seed,
                   "sicap": eval_sicap(model)}
            if "PAR" not in cohorts:
                row["par"] = auc_slide(model, par_f, par_y, dev)
            if "TCGA" not in cohorts:
                row["tcga"] = auc_slide(model, tc_f, tc_y, dev)
            res.setdefault(mode, []).append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)

    out = ROOT / f"runs/mgod_loco_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    for mode, rows in res.items():
        keys = [k for k in ("sicap", "par", "tcga") if k in rows[0]]
        print(mode, {k: {"mean": float(np.mean([r[k] for r in rows])),
                         "sd": float(np.std([r[k] for r in rows]))}
                     for k in keys})


if __name__ == "__main__":
    main()
