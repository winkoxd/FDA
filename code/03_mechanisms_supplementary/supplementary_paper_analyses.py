#!/usr/bin/env python3
"""Supplementary analyses for BDS paper: agency, GMLP era, three-mechanism tables."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_text"
OUT = ROOT / "data" / "analysis"
FLOW_CSV = OUT / "frame_fairness_per_doc_with_flow.csv"
AP_CSV = OUT / "abstract_politics_report.csv"
STATS_JSON = OUT / "figures" / "significance_stats.json"

GMLP_CUTOFF_YEAR = 2022  # proxy: devices with KYY >= 22 after Oct 2021 GMLP

AGENCY_PATTERNS = {
    "sponsor_applicant": re.compile(
        r"\b(sponsor|applicant|manufacturer|submitter|we\s+conducted|our\s+device)\b",
        re.I,
    ),
    "fda_regulator": re.compile(
        r"\b(FDA|food and drug administration|the agency|premarket notification)\b",
        re.I,
    ),
    "clinician_hcp": re.compile(
        r"\b(clinician|physician|health care provider|HCP|healthcare provider|"
        r"practitioner|sonographer|radiologist)\b",
        re.I,
    ),
    "device_algorithm": re.compile(
        r"\b(the device|the algorithm|the model|the software|machine learning|"
        r"deep learning|AI system)\b",
        re.I,
    ),
}

METRIC_TERMS = re.compile(
    r"\b(AUC|AUROC|ROC|sensitivity|specificity|accuracy|PPV|NPV|"
    r"positive predictive|negative predictive)\b",
    re.I,
)


def k_number_year(device_id: str) -> int | None:
    m = re.match(r"^[KP](\d{2})", device_id.upper())
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 70 else 2000 + yy


def analyze_agency(text: str) -> dict[str, int]:
    return {k: len(p.findall(text)) for k, p in AGENCY_PATTERNS.items()}


def dominant_agency(counts: dict[str, int]) -> str:
    if sum(counts.values()) == 0:
        return "none"
    return max(counts, key=counts.get)


def rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) else 0.0


def flow_distribution(df: pd.DataFrame, label: str) -> list[dict]:
    vc = df["discourse_flow_stage"].value_counts()
    n = len(df)
    rows = []
    for stage, c in vc.items():
        rows.append({"slice": label, "stage": stage, "n": int(c), "pct": round(100 * c / n, 2)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flow = pd.read_csv(FLOW_CSV)
    ap = pd.read_csv(AP_CSV)

    flow["approval_year"] = flow["device_id"].map(k_number_year)
    flow["post_gmlp"] = flow["approval_year"].ge(GMLP_CUTOFF_YEAR)
    flow = flow.merge(
        ap[
            [
                "device_id",
                "tech_stat_terms_count",
                "social_identity_terms_count",
                "mismatch_metric_safe",
            ]
        ],
        on="device_id",
        how="left",
    )

    # --- GMLP era ---
    pre = flow[flow["approval_year"].notna() & ~flow["post_gmlp"]]
    post = flow[flow["approval_year"].notna() & flow["post_gmlp"]]
    gmlp_rows = []
    for label, sub in [("pre_gmlp", pre), ("post_gmlp", post), ("all", flow)]:
        gmlp_rows.append(
            {
                "slice": label,
                "n": len(sub),
                "path_alpha_rate": rate(sub, "has_a_normative_fairness"),
                "frame_b_rate": rate(sub, "has_b"),
                "frame_c_rate": rate(sub, "has_c"),
                "beta2_rate": round((sub["discourse_flow_stage"] == "beta2_c_and_b").mean(), 4),
                "mean_tech_terms": float(sub["tech_stat_terms_count"].mean()),
                "mean_social_identity": float(sub["social_identity_terms_count"].mean()),
            }
        )
    pd.DataFrame(gmlp_rows).to_csv(OUT / "gmlp_era_comparison.csv", index=False)
    gmlp_flow = flow_distribution(pre, "pre_gmlp") + flow_distribution(post, "post_gmlp")
    pd.DataFrame(gmlp_flow).to_csv(OUT / "gmlp_era_flow_distribution.csv", index=False)

    # --- Agency / accountability rerouting ---
    agency_doc_rows = []
    for path in sorted(RAW.glob("*.txt")):
        did = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore")
        counts = analyze_agency(text)
        agency_doc_rows.append(
            {
                "device_id": did,
                **counts,
                "dominant_agency": dominant_agency(counts),
                "metric_mentions": len(METRIC_TERMS.findall(text)),
                "has_clinician_aid": bool(
                    re.search(r"\b(aid|assist|support|help)\b.{0,40}\b(clinician|physician)\b", text, re.I)
                    or re.search(r"\b(clinician|physician)\b.{0,40}\b(aid|assist|support)\b", text, re.I)
                ),
            }
        )
    agency_df = pd.DataFrame(agency_doc_rows)
    agency_df.to_csv(OUT / "agency_attribution_per_doc.csv", index=False)

    agency_summary = []
    n = len(agency_df)
    for col in ["sponsor_applicant", "fda_regulator", "clinician_hcp", "device_algorithm"]:
        hit = (agency_df[col] > 0).sum()
        agency_summary.append(
            {
                "agent": col,
                "docs_with_any": int(hit),
                "doc_rate": round(hit / n, 4),
                "mean_mentions_per_doc": round(float(agency_df[col].mean()), 2),
            }
        )
    aid_rate = agency_df["has_clinician_aid"].mean()
    agency_summary.append(
        {
            "agent": "clinician_aid_phrasing",
            "docs_with_any": int(agency_df["has_clinician_aid"].sum()),
            "doc_rate": round(float(aid_rate), 4),
            "mean_mentions_per_doc": None,
        }
    )
    pd.DataFrame(agency_summary).to_csv(OUT / "agency_attribution_summary.csv", index=False)

    # dominant agency distribution
    dom = agency_df["dominant_agency"].value_counts()
    pd.DataFrame({"dominant_agency": dom.index, "n": dom.values, "pct": (dom / n * 100).round(2)}).to_csv(
        OUT / "agency_dominant_distribution.csv", index=False
    )

    # --- Three mechanisms merged table ---
    beta2 = flow[flow["discourse_flow_stage"] == "beta2_c_and_b"]
    race_docs = (ap["term_race"] + ap["term_ethnicity"] > 0).sum()
    mech = {
        "mechanism_1_semantic_abstraction": {
            "docs_race_or_ethnicity_token": int(race_docs),
            "race_ethnicity_doc_rate": round(race_docs / 1388, 4),
            "top_cooccur_note": "age, gender, performance, subgroups (see semantic_reduction_argument_summary.txt)",
            "path_beta_rate": round((~flow["has_a_normative_fairness"]).mean(), 4),
            "path_beta_n": int((~flow["has_a_normative_fairness"]).sum()),
        },
        "mechanism_2_metricization": {
            "frame_c_rate": round(rate(flow, "has_c"), 4),
            "frame_b_rate": round(rate(flow, "has_b"), 4),
            "c_and_b_rate": round((flow["has_c"] & flow["has_b"]).mean(), 4),
            "docs_any_auc_sensitivity": int((agency_df["metric_mentions"] > 0).sum()),
            "metric_doc_rate": round((agency_df["metric_mentions"] > 0).mean(), 4),
        },
        "mechanism_3_accountability_rerouting": {
            "dominant_sponsor_rate": round((agency_df["dominant_agency"] == "sponsor_applicant").mean(), 4),
            "dominant_fda_rate": round((agency_df["dominant_agency"] == "fda_regulator").mean(), 4),
            "dominant_clinician_rate": round((agency_df["dominant_agency"] == "clinician_hcp").mean(), 4),
            "clinician_aid_phrasing_rate": round(float(aid_rate), 4),
            "beta2_kwic_compliance_regulatory_pct": 80.3,
        },
    }

    # --- Scenario x flow ---
    scen_rows = []
    for scen_col, label in [
        ("has_scenario_explicit", "scenario_explicit"),
        ("has_scenario_population", "scenario_population"),
    ]:
        for val in [True, False]:
            sub = flow[flow[scen_col] == val]
            scen_rows.append(
                {
                    "slice": f"{label}_{'yes' if val else 'no'}",
                    "n": len(sub),
                    "normative_a_rate": rate(sub, "has_a_normative_fairness"),
                    "beta2_rate": round((sub["discourse_flow_stage"] == "beta2_c_and_b").mean(), 4),
                    "frame_c_rate": rate(sub, "has_c"),
                }
            )
    pd.DataFrame(scen_rows).to_csv(OUT / "scenario_flow_crosstab.csv", index=False)

    # --- Tech-social discordance (from stats json if present) ---
    sig = {}
    if STATS_JSON.exists():
        sig = json.loads(STATS_JSON.read_text(encoding="utf-8")).get("significance_test", {})

    summary = {
        "corpus_n": 1388,
        "gmlp_cutoff_year": GMLP_CUTOFF_YEAR,
        "gmlp_comparison": gmlp_rows,
        "agency_summary": agency_summary,
        "three_mechanisms": mech,
        "significance_test": sig,
        "kwic_beta2_habitat_pct": {
            "governance_compliance": 40.17,
            "generic_regulatory": 40.13,
            "model_optimization": 8.79,
            "subgroup_stratification": 7.60,
            "limitation_epistemic": 0.46,
        },
        "fda_literature_anchors": {
            "muralidharan_2024_race_ethnicity_pct": 3.6,
            "mehta_2025_actr_mean": 3.3,
            "mehta_2025_no_performance_metrics_pct": 51.6,
        },
    }
    (OUT / "paper_supplementary_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote supplementary outputs to {OUT}")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
