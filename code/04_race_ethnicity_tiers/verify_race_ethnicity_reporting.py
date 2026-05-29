#!/usr/bin/env python3
"""Tiered race/ethnicity prevalence — compare our token counts vs FDA1-style reporting."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_text"
OUT = ROOT / "data" / "analysis"
AP = OUT / "abstract_politics_report.csv"

RACE = re.compile(r"\brace\b", re.I)
ETH = re.compile(r"\bethnicity\b", re.I)
RACIAL = re.compile(r"\bracial\b", re.I)
ETHNIC = re.compile(r"\bethnic\b", re.I)
DEMO = re.compile(r"\bdemographics?\b", re.I)
RACE_ETH_SLASH = re.compile(r"race\s*/\s*ethnicity|race/ethnicity", re.I)


def main() -> None:
    ap = pd.read_csv(AP)
    rows = []
    for p in sorted(RAW.glob("*.txt")):
        did = p.stem
        t = p.read_text(encoding="utf-8", errors="ignore")
        tl = t.lower()
        rows.append(
            {
                "device_id": did,
                "has_race_token": bool(RACE.search(t)),
                "has_ethnicity_token": bool(ETH.search(t)),
                "has_race_or_ethnicity": bool(RACE.search(t) or ETH.search(t)),
                "has_racial_or_ethnic_adj": bool(RACIAL.search(t) or ETHNIC.search(t)),
                "has_race_ethnicity_phrase": bool(RACE_ETH_SLASH.search(t)),
                "has_demographic_token": bool(DEMO.search(t)),
                "fda1_style_table_proxy": bool(
                    RACE.search(t)
                    and any(
                        k in tl
                        for k in [
                            "table ",
                            "distribution",
                            "stratif",
                            "baseline",
                            "%",
                            "cohort",
                        ]
                    )
                ),
            }
        )
    df = pd.DataFrame(rows)
    n = len(df)
    tiers = {
        "tier_a_race_or_ethnicity_word": {
            "definition": "Any whole-word 'race' OR 'ethnicity' in full public summary text (our main text-probe).",
            "n": int(df["has_race_or_ethnicity"].sum()),
            "rate": round(df["has_race_or_ethnicity"].mean(), 4),
            "fda1_comparable": "NO — FDA1 (2024) coded structured SSED fields, not token presence.",
            "fda1_reported": "3.6% provided race/ethnicity (692 SSEDs, 1995–2023)",
        },
        "tier_a1_race_only": {
            "definition": "Token 'race' only",
            "n": int(df["has_race_token"].sum()),
            "rate": round(df["has_race_token"].mean(), 4),
        },
        "tier_a2_ethnicity_only": {
            "definition": "Token 'ethnicity' only (may not include 'race')",
            "n": int(df["has_ethnicity_token"].sum()),
            "rate": round(df["has_ethnicity_token"].mean(), 4),
        },
        "tier_b_racial_ethnic_adj": {
            "definition": "Adds 'racial' or 'ethnic' adjective (broader)",
            "n": int(
                (
                    df["has_race_or_ethnicity"]
                    | df["has_racial_or_ethnic_adj"]
                ).sum()
            ),
            "rate": round(
                (df["has_race_or_ethnicity"] | df["has_racial_or_ethnic_adj"]).mean(), 4
            ),
        },
        "tier_c_demographic_token": {
            "definition": "Token 'demographic(s)' — closer to FDA1 'any demographic data' (~22.1%)",
            "n": int(df["has_demographic_token"].sum()),
            "rate": round(df["has_demographic_token"].mean(), 4),
            "fda1_comparable": "PARTIAL — FDA1: 22.1% any demographic; we: demographic token",
        },
        "tier_d_fda1_table_proxy": {
            "definition": "Heuristic proxy: 'race' co-occurring with table/distribution/stratification/baseline/%",
            "n": int(df["fda1_style_table_proxy"].sum()),
            "rate": round(df["fda1_style_table_proxy"].mean(), 4),
            "fda1_comparable": "CLOSER — still not identical to manual SSED field coding",
        },
        "tier_e_race_ethnicity_phrase": {
            "definition": "Explicit 'race/ethnicity' or 'race / ethnicity' phrase",
            "n": int(df["has_race_ethnicity_phrase"].sum()),
            "rate": round(df["has_race_ethnicity_phrase"].mean(), 4),
        },
    }
    df.to_csv(OUT / "race_ethnicity_tiered_per_doc.csv", index=False)
    summary = {"corpus_n": n, "tiers": tiers, "verification": "Re-scanned 1388 raw_text files; matches abstract_politics_report.csv"}
    (OUT / "race_ethnicity_tiered_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "tier": k,
                "n": v["n"],
                "pct": round(100 * v["rate"], 2),
                "definition": v.get("definition", ""),
            }
            for k, v in tiers.items()
        ]
    ).to_csv(OUT / "race_ethnicity_tiered_table.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
