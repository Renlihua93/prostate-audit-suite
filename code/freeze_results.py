#!/usr/bin/env python3
"""Compile frozen results for the manuscript from runs/* artefacts."""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/frozen_2026-11-08"
OUT.mkdir(parents=True, exist_ok=True)


def mgod_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        p = ROOT / f"runs/mgod_pilot_{enc}/results.json"
        if not p.exists():
            continue
        d = json.load(open(p))["summary"]
        for mode in ("tile_only", "tile_plus_slide"):
            r = d[mode]
            rows.append({"encoder": enc, "mode": mode,
                         **{k: round(v["mean"], 3) for k, v in r.items()}})
    return pd.DataFrame(rows)


def loco_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        p = ROOT / f"runs/mgod_loco_{enc}/results.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        for mode, runs in d.items():
            row = {"encoder": enc, "mode": mode}
            for k in ("sicap", "par", "tcga"):
                if runs and k in runs[0]:
                    row[k] = round(float(pd.Series(
                        [r[k] for r in runs]).mean()), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def ppbench_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        p = ROOT / f"runs/mgod_ppbench_{enc}/summary.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["encoder"] = enc
            rows.append(d)
    return pd.concat(rows) if rows else pd.DataFrame()


def peso_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        p = ROOT / f"runs/peso_eval_{enc}/results.json"
        if p.exists():
            d = json.load(open(p))
            rows.append({"encoder": enc,
                         "linear_region_auc": round(
                             d["linear"]["region_auc"], 3),
                         "linear_slide_auc": round(
                             d["linear"]["slide_auc"], 3),
                         "mgod_region_auc": round(
                             d["mgod"]["region_auc_mean"], 3),
                         "mgod_sd": round(d["mgod"]["region_auc_sd"], 3)})
    return pd.DataFrame(rows)


def utility_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        base = ROOT / f"runs/clinical_utility_{enc}/simulation.csv"
        d = pd.read_csv(base)
        if enc == "hoptimus0":
            s34 = ROOT / "runs/clinical_utility_hoptimus0/simulation_s34.csv"
            s012 = ROOT / "runs/clinical_utility_hoptimus0/simulation_s012.csv"
            if s012.exists():
                d = pd.concat([pd.read_csv(s012), d], ignore_index=True)
        ok = d[d.feasible]
        if len(ok):
            g = ok.groupby(["alpha", "K"]).agg(
                cleared=("cleared", "mean"),
                missed=("missed_per100", "mean"),
                sens=("sens_detected", "mean"),
                dis_defer=("disagree_deferred", "mean"),
                n=("rep", "count")).reset_index()
            g["encoder"] = enc
            rows.append(g)
    return pd.concat(rows) if rows else pd.DataFrame()


def sensitivity_table():
    p = ROOT / "runs/sensitivity_phikon/site_size_PAR_a0.05.csv"
    q = ROOT / "runs/sensitivity_phikon/site_size_PAR_a0.1.csv"
    rows = []
    for f, a in ((p, 0.05), (q, 0.10)):
        if f.exists():
            d = pd.read_csv(f)
            g = d.groupby("n_site").agg(
                feasible=("feasible", "mean"),
                cleared=("cleared", "mean"),
                fn_clear=("fn_clear", "mean")).reset_index()
            g["alpha"] = a
            rows.append(g)
    return pd.concat(rows) if rows else pd.DataFrame()


def recalibration_table():
    rows = []
    for enc in ("phikon", "hoptimus0"):
        for variant in ("raw", "temperature", "isotonic"):
            p = ROOT / f"runs/recalibration_{enc}/metrics_{variant}.csv"
            if p.exists():
                d = pd.read_csv(p)
                rows.append({"encoder": enc, "variant": variant,
                             "auc": round(d.auc.mean(), 3),
                             "slope": round(d.slope.mean(), 3),
                             "brier": round(d.brier.mean(), 3)})
    return pd.DataFrame(rows)


def main():
    tables = {
        "MGOD": mgod_table(),
        "LOCO": loco_table(),
        "PPBench": ppbench_table(),
        "PESO": peso_table(),
        "DOC": utility_table(),
        "SiteSize": sensitivity_table(),
        "Recalibration": recalibration_table(),
    }
    md = ["# Frozen results (2026-11-08)\n"]
    for name, d in tables.items():
        d.to_csv(OUT / f"{name}.csv", index=False)
        md.append(f"\n## {name}\n")
        md.append("```\n" + d.round(3).to_string(index=False) + "\n```")
    (OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
