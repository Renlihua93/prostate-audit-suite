#!/usr/bin/env python3
"""MGOD pilot: does slide-level (weak) supervision teach a tile head to
transfer better across cohorts?

Shared tile head on frozen Phikon-v1 embeddings (768-d), two training modes:
  A. tile-CE only on SICAPv2 patch labels
  B. tile-CE + slide ISUP MIL loss (attention pooling) on SICAPv2 slides

Evaluation:
  - SICAP held-out slides (patient-grouped split): macro-F1, QWK, AUC>=2
  - PAR slides (120) and TCGA slides (113): AUC>=2 via mean tile p(G4+)

All cohorts use 112 um FOV tiles.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SICAP_ZIP = Path(f"{DATA_ROOT}/sicapv2/SICAPv2.zip")


class Head(nn.Module):
    def __init__(self, dim, hidden=256, n_tile=4, n_grade=5):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Dropout(0.2))
        self.tile = nn.Linear(hidden, n_tile)
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.slide = nn.Linear(hidden, n_grade)

    def forward(self, x):  # x: [n_tiles, dim]
        h = self.trunk(x)
        logits_tile = self.tile(h)
        a = torch.softmax(self.att(h).squeeze(-1), dim=0)
        z = (a[:, None] * h).sum(0)
        return logits_tile, self.slide(z[None])[0]


def load_sicap(encoder="phikon"):
    if encoder == "phikon":
        path = ROOT / "features/sicap_phikon_c224/patch_features.npz"
    else:
        path = ROOT / "features/sicap_c224/patch_features.npz"
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


def load_target(feats_dir, labels):
    feats, y = {}, {}
    for f in sorted(Path(feats_dir).glob("*.npz")):
        key = f.stem.lower().replace("_leica", "").replace("_grundium", "")
        z = np.load(f)
        feats[key] = z["tile_emb"].astype(np.float32)
    for _, r in labels.iterrows():
        s = str(r.slide).lower()
        if s in feats:
            y[s] = int(r.y)
    return feats, y


def slide_score(model, tiles, dev, pool="mean"):
    with torch.no_grad():
        lt, ls = model(torch.from_numpy(tiles).to(dev))
        p = torch.softmax(lt, -1)[:, 2:].sum(1).cpu().numpy()  # P(G4 or G5)
        return float(p.mean()) if pool == "mean" else float(p.max())


def auc_feats(model, feats, y, dev):
    sc = np.array([slide_score(model, feats[s], dev) for s in y])
    yy = np.array([y[s] for s in y])
    return roc_auc_score(yy, sc)


def train(mode, emb, labels, slides, gg, params, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sl_ids = np.array(sorted(gg))
    idx_all = np.where(labels >= 0)[0]
    tiles_by_slide = {s: emb[slides == s] for s in sl_ids}
    tile_lab_by_idx = labels
    model = Head(emb.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_t = nn.CrossEntropyLoss()
    ce_s = nn.CrossEntropyLoss()
    for ep in range(params["epochs"]):
        model.train()
        perm = rng.permutation(idx_all)
        for b in range(0, len(perm), params["batch"]):
            idx = perm[b:b + params["batch"]]
            x = torch.from_numpy(emb[idx]).to(dev)
            yt = torch.from_numpy(tile_lab_by_idx[idx]).long().to(dev)
            lt, _ = model(x)
            loss = ce_t(lt, yt)
            # one slide per step for MIL
            s = sl_ids[rng.integers(len(sl_ids))]
            xf = torch.from_numpy(tiles_by_slide[s]).to(dev)
            _, ls = model(xf)
            loss = loss + params["lam"] * ce_s(ls[None], torch.tensor(
                [gg[s] - 1], device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--par-dir", default=None)
    ap.add_argument("--tcga-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    par_dir = Path(args.par_dir) if args.par_dir else (
        ROOT / "features/phikon" if args.encoder == "phikon"
        else ROOT / "features/hoptimus0_par")
    tcga_dir = Path(args.tcga_dir) if args.tcga_dir else (
        ROOT / "features/tcga_phikon" if args.encoder == "phikon"
        else ROOT / "features/hoptimus0_tcga")
    out = ROOT / f"runs/mgod_pilot_{args.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    emb, labels, slides, gg, pat = load_sicap(args.encoder)

    par = pd.read_csv(ROOT / "runs/par_external/par_scores.csv")
    lab = pd.read_csv(ROOT / "data/par/slide_labels.tsv", sep="\t")
    lab["slide"] = lab.slide_id.astype(str).str.upper()
    r = lab[["slide", "isup_1", "isup_r2", "isup_r3"]].set_index("slide")
    maj = r.apply(lambda x: np.nanmedian(x.dropna().astype(float))
                  if x.notna().any() else np.nan, axis=1)
    par = par.merge(maj.rename("maj"), left_on="slide", right_index=True)
    par = par.dropna(subset=["maj"])
    par_lab = pd.DataFrame({"slide": par.slide.str.lower(),
                            "y": (par.maj >= 2).astype(int)})
    par_f, par_y = load_target(par_dir, par_lab)

    tc = pd.read_csv(ROOT / "runs/tcga_external/tcga_scores.csv")
    tc_lab = pd.DataFrame({"slide": tc.slide.astype(str),
                           "y": (tc.gg.astype(int) >= 2).astype(int)})
    tc_f, tc_y = load_target(tcga_dir, tc_lab)

    params = {"epochs": args.epochs, "batch": 256, "lam": 0.5}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    slides_uniq = np.array(sorted(gg))
    patients = np.array([pat[s] for s in slides_uniq])

    results = {"tile_only": [], "tile_plus_slide": []}
    for seed in args.seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=seed)
        tr, te = next(gss.split(slides_uniq, groups=patients))
        g_tr, g_te = set(slides_uniq[tr]), set(slides_uniq[te])
        for mode, lam in (("tile_only", 0.0), ("tile_plus_slide", 0.5)):
            params["lam"] = lam
            mask_tr = np.array([slides[i] in g_tr for i in range(len(labels))])
            emb_tr, lab_tr = emb[mask_tr], labels[mask_tr]
            slides_tr = slides[mask_tr]
            gg_tr = {s: gg[s] for s in gg if s in g_tr}
            model = train(mode, emb_tr, lab_tr, slides_tr, gg_tr, params,
                          seed=seed)
            model.eval()
            y_te = np.array([gg[s] for s in sorted(g_te)])
            sc_in = np.array([slide_score(model, emb[slides == s], dev)
                              for s in sorted(g_te)])
            res = {"auc_ge2": roc_auc_score((y_te >= 2).astype(int), sc_in),
                   "par_auc_ge2": auc_feats(model, par_f, par_y, dev),
                   "tcga_auc_ge2": auc_feats(model, tc_f, tc_y, dev)}
            if mode == "tile_plus_slide":
                preds = []
                with torch.no_grad():
                    for s in sorted(g_te):
                        _, ls = model(torch.from_numpy(
                            emb[slides == s]).to(dev))
                        preds.append(int(ls.argmax().item()) + 1)
                res["qwk"] = cohen_kappa_score(y_te, preds,
                                               weights="quadratic")
                res["macro_f1"] = f1_score(y_te, preds, average="macro")
            results[mode].append(res)
            print(f"seed{seed} {mode}",
                  json.dumps({k: round(v, 3) for k, v in res.items()}),
                  flush=True)

    summary = {m: {k: {"mean": float(np.mean([r[k] for r in v])),
                      "sd": float(np.std([r[k] for r in v]))}
                   for k in v[0]} for m, v in results.items()}
    (out / "results.json").write_text(json.dumps(
        {"runs": results, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
