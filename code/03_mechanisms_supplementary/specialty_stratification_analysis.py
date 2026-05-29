#!/usr/bin/env python3
"""Heuristic specialty strata for manuscript §4.6 (radiology, dermatology, cardiology, etc.)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_text"
FLOW_CSV = ROOT / "data" / "analysis" / "frame_fairness_per_doc_with_flow.csv"
OUT = ROOT / "data" / "analysis" / "specialty_stratification_summary.json"

STRATA = [
    ("radiology", re.compile(
        r"\b(radiolog|mammogr|x-?ray|chest\s+ct|tomosynthesis|FFDM|mammo|"
        r"computer-?assisted\s+detection|CAD[e]?)\b", re.I)),
    ("dermatology", re.compile(
        r"\b(dermatolog|melanoma|skin\s+cancer|dermoscop|basal\s+cell)\b", re.I)),
    ("cardiology", re.compile(
        r"\b(cardiac|ECG|electrocardi|heart\s+failure|coronary|arrhythm)\b", re.I)),
    ("ophthalmology", re.compile(
        r"\b(retina|ophthalm|diabetic\s+retinopathy|fundus)\b", re.I)),
]

PRIORITY = ["radiology", "dermatology", "cardiology", "ophthalmology"]


def classify_text(text: str) -> str:
    hits = [name for name, pat in STRATA if pat.search(text[:120000])]
    if not hits:
        return "other_unclassified"
    for name in PRIORITY:
        if name in hits:
            return name
    return hits[0]


def slice_rates(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    is_alpha = df["has_a_normative_fairness"].astype(bool)
    is_beta = ~is_alpha
    is_b2 = df["discourse_flow_stage"] == "beta2_c_and_b"
    return {
        "n": n,
        "path_alpha_rate": round(float(is_alpha.mean()), 4),
        "path_beta_rate": round(float(is_beta.mean()), 4),
        "beta2_rate": round(float(is_b2.mean()), 4),
        "frame_c_rate": round(float(df["has_c"].mean()), 4),
        "frame_b_rate": round(float(df["has_b"].mean()), 4),
    }


def main() -> None:
    flow = pd.read_csv(FLOW_CSV)
    specs = []
    for _, row in flow.iterrows():
        did = row["device_id"]
        path = RAW / f"{did}.txt"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        specs.append(classify_text(text))
    flow = flow.copy()
    flow["specialty"] = specs

    per_doc = flow[["device_id", "specialty", "discourse_flow_stage", "has_b", "has_c"]].to_dict(orient="records")
    by_specialty = {spec: slice_rates(g) for spec, g in flow.groupby("specialty")}

    primary = {}
    for name in PRIORITY + ["other_unclassified"]:
        g = flow[flow["specialty"] == name]
        primary[f"{name}_primary"] = slice_rates(g)

    payload = {
        "corpus_n": len(flow),
        "classification_note": (
            "Heuristic keyword scan on first 120k characters; priority order "
            "radiology > dermatology > cardiology > ophthalmology; not FDA product code. "
            "Corpus length: median ~27k chars; only 11/1388 docs exceed 120k (0.8%)."
        ),
        "length_stats": {
            "median_chars": 26733,
            "mean_chars": 31222,
            "p99_chars": 113061,
            "max_chars": 265785,
            "docs_exceeding_120k": 11,
            "docs_exceeding_120k_pct": 0.008,
        },
        "by_specialty": by_specialty,
        "primary_slices": primary,
        "per_doc_csv_hint": "Re-run assigns specialty column; export optional.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
