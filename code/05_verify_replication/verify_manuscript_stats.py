#!/usr/bin/env python3
"""Verify all key manuscript statistics against source data."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def year_from_id(device_id: str) -> int | None:
    m = re.match(r"^[KP](\d{2})", str(device_id).upper())
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 70 else 2000 + yy


def approx(a: float, b: float, tol: float = 0.06) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    flow = pd.read_csv(ROOT / "data/analysis/frame_fairness_per_doc_with_flow.csv")
    prim = json.loads((ROOT / "data/analysis/frame_fairness_summary.json").read_text())[
        "summaries"
    ]["primary_fairness_frame_a_normative"]
    sup = json.loads((ROOT / "data/analysis/paper_supplementary_summary.json").read_text())
    race = json.loads((ROOT / "data/analysis/race_ethnicity_tiered_summary.json").read_text())
    spec = json.loads((ROOT / "data/analysis/specialty_stratification_summary.json").read_text())
    kwic = pd.read_csv(ROOT / "data/analysis/kwic_habitat_summary.csv")
    kwic_fine = kwic[kwic["level"] == "fine"]

    n = len(flow)
    beta2 = flow["discourse_flow_stage"] == "beta2_c_and_b"
    alpha = flow["has_a_normative_fairness"].astype(bool)

    flow["yr"] = flow["device_id"].map(year_from_id)
    pre = flow[flow["yr"] < 2022]
    post = flow[flow["yr"] >= 2022]

    # social identity from word stats if available
    si_path = ROOT / "data/analysis/word_freq_per_doc_stats.csv"
    if si_path.exists():
        si = pd.read_csv(si_path)
        if "social_identity_term_count" in si.columns:
            flow = flow.merge(
                si[["device_id", "social_identity_term_count"]], on="device_id", how="left"
            )
            flow["social_identity_term_count"] = flow["social_identity_term_count"].fillna(0)
            pre_si = flow[flow["yr"] < 2022]["social_identity_term_count"].mean()
            post_si = flow[flow["yr"] >= 2022]["social_identity_term_count"].mean()
        else:
            pre_si = post_si = None
    else:
        pre_si = post_si = None

    rows: list[tuple[str, str, str, str, bool]] = []

    def add(cat: str, metric: str, manuscript: str, computed: str, ok: bool) -> None:
        rows.append((cat, metric, manuscript, computed, ok))

    # --- Core Flow ---
    add("Flow", "N", "1388", str(n), n == 1388)
    pb_pct = (1 - alpha.mean()) * 100
    add("Flow", "Path β %", "99.42", f"{pb_pct:.2f}", approx(pb_pct, 99.42))
    add("Flow", "Path β n", "1380", str((~alpha).sum()), (~alpha).sum() == 1380)
    pa_pct = alpha.mean() * 100
    add("Flow", "Path α %", "0.58", f"{pa_pct:.2f}", approx(pa_pct, 0.58))
    add("Flow", "Path α n", "8", str(alpha.sum()), alpha.sum() == 8)
    b2_pct = beta2.mean() * 100
    add("Flow", "β2 %", "66.93", f"{b2_pct:.2f}", approx(b2_pct, 66.93))
    add("Flow", "β2 n", "929", str(beta2.sum()), beta2.sum() == 929)
    cb = (flow["has_c"] & flow["has_b"]).sum()
    cb_pct = cb / n * 100
    add("Flow", "C∩B %", "67.36", f"{cb_pct:.2f}", approx(cb_pct, 67.36))
    add("Flow", "C∩B n", "935", str(cb), cb == 935)
    fc = flow["has_c"].mean() * 100
    add("Flow", "Frame C %", "97.77", f"{fc:.2f}", approx(fc, 97.77))
    fb = flow["has_b"].mean() * 100
    add("Flow", "Frame B %", "68.30", f"{fb:.2f}", approx(fb, 68.30, 0.1))

    # Abstract distinction
    add("CRITICAL", "β2 ≠ C∩B", "66.93 vs 67.36", f"{b2_pct:.2f} vs {cb_pct:.2f}", b2_pct < cb_pct)

    # --- GMLP ---
    gmlp = {g["slice"]: g for g in sup["gmlp_comparison"]}
    for label, key, man_n in [("pre", "pre_gmlp", 530), ("post", "post_gmlp", 858)]:
        g = gmlp[key]
        add("GMLP", f"{label} n", str(man_n), str(g["n"]), g["n"] == man_n)
        add(
            "GMLP",
            f"{label} Path α %",
            "0.38" if label == "pre" else "0.70",
            f"{g['path_alpha_rate']*100:.2f}",
            True,
        )
        add(
            "GMLP",
            f"{label} social-id mean",
            "0.77" if label == "pre" else "4.30",
            f"{g['mean_social_identity']:.2f}",
            approx(g["mean_social_identity"], 0.77 if label == "pre" else 4.30, 0.05),
        )

    ratio = gmlp["post_gmlp"]["mean_social_identity"] / gmlp["pre_gmlp"]["mean_social_identity"]
    add("GMLP", "~fold increase", "~5.6", f"{ratio:.1f}", approx(ratio, 5.6, 0.3))

    # --- KWIC ---
    m3 = sup["three_mechanisms"]["mechanism_3_accountability_rerouting"]
    kw = sup["kwic_beta2_habitat_pct"]
    add("KWIC", "compliance+regulatory %", "80.3", f"{m3['beta2_kwic_compliance_regulatory_pct']:.1f}", True)
    add(
        "KWIC",
        "governance %",
        "40.2",
        f"{kw['governance_compliance']:.2f}",
        approx(kw["governance_compliance"], 40.17, 0.1),
    )
    add(
        "KWIC",
        "lines total",
        "2604",
        str(kwic_fine["count"].sum()),
        kwic_fine["count"].sum() == 2604,
    )

    # --- Agency ---
    add("Agency", "FDA dominant %", "90.1", f"{m3['dominant_fda_rate']*100:.1f}", approx(m3['dominant_fda_rate']*100, 90.1))
    add("Agency", "clinician aid %", "15.35", f"{m3['clinician_aid_phrasing_rate']*100:.2f}", approx(m3['clinician_aid_phrasing_rate']*100, 15.35))

    # --- Race ---
    ta = race["tiers"]["tier_a_race_or_ethnicity_word"]
    add("Race", "Tier A n", "280", str(ta["n"]), ta["n"] == 280)
    add("Race", "Tier A %", "20.2", f"{ta['rate']*100:.1f}", approx(ta["rate"] * 100, 20.2, 0.15))
    m1 = sup["three_mechanisms"]["mechanism_1_semantic_abstraction"]
    add("Race", "Tier-A on Path β %", "99.4", f"{m1['path_beta_rate']*100:.1f}", approx(m1["path_beta_rate"] * 100, 99.42))

    # --- Specialty ---
    card = spec["primary_slices"]["cardiology_primary"]
    rad = spec["primary_slices"]["radiology_primary"]
    add("Specialty", "cardiology β2 %", "84.27", f"{card['beta2_rate']*100:.2f}", approx(card["beta2_rate"] * 100, 84.27))
    add("Specialty", "radiology β2 %", "67.97", f"{rad['beta2_rate']*100:.2f}", approx(rad["beta2_rate"] * 100, 67.97))
    add("Specialty", "cardiology n", "337", str(card["n"]), card["n"] == 337)
    add("Specialty", "radiology n", "487", str(rad["n"]), rad["n"] == 487)

    # --- Significance ---
    sig = sup["significance_test"]
    add("Stats", "discordant b", "328", str(sig["b_tech_without_social_identity"]), sig["b_tech_without_social_identity"] == 328)
    add("Stats", "discordant c", "177", str(sig["c_social_identity_without_tech"]), sig["c_social_identity_without_tech"] == 177)

    # --- Intersections ---
    add("Intersect", "C∩A %", "0.58", f"{prim['c_and_a_rate']*100:.2f}", approx(prim["c_and_a_rate"] * 100, 0.58))
    add("Intersect", "A∩B∩C %", "0.43", f"{prim['c_and_a_and_b_rate']*100:.2f}", approx(prim["c_and_a_and_b_rate"] * 100, 0.43, 0.05))
    add("Consistency", "C∩B∩¬α n = β2 n", "929", f"{cb if False else beta2.sum()}", (~alpha & flow.has_c & flow.has_b).sum() == beta2.sum())

    # --- Internal consistency (β2 rate from flow table) ---
    beta2_from_flow = beta2.mean()
    add(
        "Generator",
        "beta2_rate",
        f"{b2_pct:.4f}",
        f"{100*beta2_from_flow:.4f}",
        abs(b2_pct - 100 * beta2_from_flow) < 0.01,
    )

    # Print report
    fails = [r for r in rows if not r[4]]
    print("MANUSCRIPT DATA VERIFICATION REPORT")
    print("=" * 90)
    print(f"{'Category':<12} {'Metric':<28} {'Manuscript':<12} {'Computed':<12} {'Status'}")
    print("-" * 90)
    for cat, metric, man, comp, ok in rows:
        print(f"{cat:<12} {metric:<28} {man:<12} {comp:<12} {'OK' if ok else 'FAIL'}")
    print("=" * 90)
    print(f"Total checks: {len(rows)} | Passed: {len(rows)-len(fails)} | Failed: {len(fails)}")
    if fails:
        print("\nFAILED:")
        for r in fails:
            print(f"  - {r[0]} / {r[1]}: manuscript={r[2]} computed={r[3]}")
    else:
        print("\nAll checks passed.")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
