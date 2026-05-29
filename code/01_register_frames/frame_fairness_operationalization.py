#!/usr/bin/env python3
"""
Fairness-focused Frame A/B/C operationalization and representation audit.

Primary Frame A (fairness): normative lexicon WITHOUT bare "representation".
  Bare representation is polysemous: ML representation learning / statistical sample sizes (depoliticized)
  vs demographic representation (justice/demography). Primary Path α excludes bare token; legacy sensitivity adds it.
Legacy §4 A (sensitivity): includes bare "representation".
Representation rows are audited (technical_ml vs demographic_fairness vs dataset_population vs unclear).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REP = re.compile(r"\brepresentation\b", re.IGNORECASE)
REPRESENTATIVENESS = re.compile(r"\brepresentativeness\b", re.IGNORECASE)

A_NORMATIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fairness", re.compile(r"\bfairness\b", re.I)),
    ("equity", re.compile(r"\bequity\b", re.I)),
    ("basic_rights", re.compile(r"\bbasic\s+rights\b", re.I)),
    ("fundamental_rights", re.compile(r"\bfundamental\s+rights\b", re.I)),
    ("representativeness", re.compile(r"\brepresentativeness\b", re.I)),
]

A_LEGACY_PATTERNS = A_NORMATIVE_PATTERNS + [
    ("representation", REP),
]

B_PATTERNS = [
    re.compile(r"\bbias(?:es|ed)?\b", re.I),
    re.compile(r"\bmitigat(?:e|ion|ing)\b", re.I),
    re.compile(r"\bgovernance\b", re.I),
    re.compile(r"\bpost[\s-]?market\b", re.I),
    re.compile(r"\bmonitor(?:ing|ed)?\b", re.I),
    re.compile(r"\brisk\s+management\b", re.I),
]

C_PATTERNS = [
    re.compile(r"\bsubgroup(?:s)?\b", re.I),
    re.compile(r"\baccuracy\b", re.I),
    re.compile(r"\bsensitivity\b", re.I),
    re.compile(r"\bspecificity\b", re.I),
    re.compile(r"\bauc\b", re.I),
    re.compile(r"\bperformance\b", re.I),
]

DEMO_WINDOW = re.compile(
    r"(?:demographic|race|ethnic|gender|sex|age|bmi|skin\s+tone|"
    r"underrepresent|overrepresent|divers|patient\s+population|cohort)",
    re.I,
)

SCENARIO_EXPLICIT = [
    re.compile(r"\bscenarios?\b", re.I),
    re.compile(r"\buse\s+cases?\b", re.I),
    re.compile(r"\buse\s+scenario", re.I),
]
SCENARIO_SETTING = [
    re.compile(r"\bintended\s+use\b", re.I),
    re.compile(r"\bindications?\s+for\s+use\b", re.I),
    re.compile(r"\bdeployment\b", re.I),
    re.compile(r"\breal[\s-]world\b", re.I),
]
SCENARIO_WORKFLOW = [
    re.compile(r"\bworkflow", re.I),
    re.compile(r"\bclinical\s+site", re.I),
    re.compile(r"\bhealthcare\s+setting", re.I),
]
SCENARIO_POPULATION = [
    re.compile(r"\bintended\s+use\s+population\b", re.I),
    re.compile(r"\bpatient\s+population\b", re.I),
    re.compile(r"\bclinically\s+relevant\b", re.I),
]

FAIRNESS_BROAD_PATTERNS = [
    re.compile(r"\bfairness\b", re.I),
    re.compile(r"\bequity\b", re.I),
    re.compile(r"\brepresentativeness\b", re.I),
    re.compile(r"\bbias(?:es|ed)?\b", re.I),
    re.compile(r"\bdisparit", re.I),
    re.compile(r"\bmitigat", re.I),
]
DEMOGRAPHIC = re.compile(r"\bdemographic", re.I)
SUBGROUP = re.compile(r"\bsubgroup", re.I)

TECH_WINDOW = re.compile(
    r"(?:image|visual|feature|latent|tensor|pixel|3d|2d|model|signal|"
    r"anatomical|neural|fused|index|computer\s+analysis|automatic)",
    re.I,
)

DATASET_WINDOW = re.compile(
    r"(?:dataset|study\s+sample|validation\s+set|training\s+set|"
    r"heterogenous\s+dataset|us\s+population)",
    re.I,
)


def any_hit(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def normative_hits(text: str) -> list[str]:
    return [name for name, pat in A_NORMATIVE_PATTERNS if pat.search(text)]


def classify_representation(text: str) -> tuple[str, str]:
    """Return (category, first_snippet)."""
    m = REP.search(text)
    if not m:
        return ("no_representation", "")

    start = max(0, m.start() - 160)
    end = min(len(text), m.end() + 160)
    window = text[start:end]
    snippet = " ".join(window.split())
    low = window.lower()

    if DEMO_WINDOW.search(low):
        return ("demographic_fairness", snippet)
    if TECH_WINDOW.search(low):
        return ("technical_ml", snippet)
    if DATASET_WINDOW.search(low):
        return ("dataset_population", snippet)
    return ("unclear_manual_review", snippet)


def intersection_flags(
    has_a: bool, has_b: bool, has_c: bool
) -> dict[str, bool]:
    return {
        "a": has_a,
        "b": has_b,
        "c": has_c,
        "c_and_a": has_a and has_c,
        "c_and_b": has_b and has_c,
        "a_and_b_and_c": has_a and has_b and has_c,
        "non_a": not has_a,
    }


def summarize_corpus(rows: list[dict], a_key: str) -> dict[str, int | float]:
    n = len(rows)
    if n == 0:
        return {"documents_total": 0}

    def rate(flag_key: str) -> float:
        return sum(1 for r in rows if r[flag_key]) / n

    a = [r[a_key] for r in rows]
    b = [r["has_b"] for r in rows]
    c = [r["has_c"] for r in rows]

    return {
        "documents_total": n,
        "a_doc_count": sum(a),
        "a_doc_rate": rate(a_key),
        "non_a_doc_count": sum(1 for x in a if not x),
        "non_a_doc_rate": sum(1 for x in a if not x) / n,
        "b_doc_count": sum(b),
        "b_doc_rate": rate("has_b"),
        "c_doc_count": sum(c),
        "c_doc_rate": rate("has_c"),
        "c_and_a": sum(1 for i in range(n) if c[i] and a[i]),
        "c_and_a_rate": sum(1 for i in range(n) if c[i] and a[i]) / n,
        "c_and_b": sum(1 for i in range(n) if c[i] and b[i]),
        "c_and_b_rate": sum(1 for i in range(n) if c[i] and b[i]) / n,
        "c_and_a_and_b": sum(1 for i in range(n) if c[i] and a[i] and b[i]),
        "c_and_a_and_b_rate": sum(1 for i in range(n) if c[i] and a[i] and b[i]) / n,
    }


def run(raw_text_dir: Path, output_dir: Path) -> None:
    paths = sorted(raw_text_dir.glob("*.txt"))
    rows: list[dict] = []
    rep_audit: list[dict] = []

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        norm_terms = normative_hits(text)
        has_rep = bool(REP.search(text))
        rep_cat, rep_snip = classify_representation(text)
        has_a_normative = bool(norm_terms)
        has_a_legacy = has_a_normative or has_rep
        # Fairness-focused: representation counts only with demographic-fairness context
        has_a_rep_demo = has_a_normative or (
            has_rep and rep_cat == "demographic_fairness"
        )

        has_b = any_hit(text, B_PATTERNS)
        has_c = any_hit(text, C_PATTERNS)
        has_scenario_explicit = any_hit(text, SCENARIO_EXPLICIT)
        has_scenario_setting = any_hit(text, SCENARIO_SETTING)
        has_scenario_workflow = any_hit(text, SCENARIO_WORKFLOW)
        has_scenario_population = any_hit(text, SCENARIO_POPULATION)
        has_fairness_broad = any_hit(text, FAIRNESS_BROAD_PATTERNS)
        has_demographic = bool(DEMOGRAPHIC.search(text))
        has_subgroup = bool(SUBGROUP.search(text))

        device_id = path.stem
        row = {
            "device_id": device_id,
            "has_a_normative_fairness": has_a_normative,
            "has_a_legacy_with_representation": has_a_legacy,
            "has_a_representation_demographic": has_a_rep_demo,
            "has_b": has_b,
            "has_c": has_c,
            "has_scenario_explicit": has_scenario_explicit,
            "has_scenario_setting": has_scenario_setting,
            "has_scenario_workflow": has_scenario_workflow,
            "has_scenario_population": has_scenario_population,
            "has_fairness_broad": has_fairness_broad,
            "has_demographic": has_demographic,
            "has_subgroup": has_subgroup,
            "normative_terms_hit": "|".join(norm_terms),
            "has_representation_token": has_rep,
            "representation_category": rep_cat if has_rep else "",
        }
        rows.append(row)

        if has_rep:
            rep_audit.append(
                {
                    "device_id": device_id,
                    "representation_category": rep_cat,
                    "in_a_legacy": int(has_a_legacy),
                    "in_a_normative_fairness": int(has_a_normative),
                    "in_a_representation_demographic": int(has_a_rep_demo),
                    "normative_terms_hit": "|".join(norm_terms),
                    "has_bias": int(any_hit(text, [B_PATTERNS[0]])),
                    "has_demographic_window": int(bool(DEMO_WINDOW.search(text))),
                    "first_representation_snippet": rep_snip,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    per_doc_fields = [
        "device_id",
        "has_a_normative_fairness",
        "has_a_legacy_with_representation",
        "has_a_representation_demographic",
        "has_b",
        "has_c",
        "has_scenario_explicit",
        "has_scenario_setting",
        "has_scenario_workflow",
        "has_scenario_population",
        "has_fairness_broad",
        "has_demographic",
        "has_subgroup",
        "normative_terms_hit",
        "has_representation_token",
        "representation_category",
    ]
    with (output_dir / "frame_fairness_per_doc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_doc_fields)
        writer.writeheader()
        writer.writerows(rows)

    audit_fields = list(rep_audit[0].keys()) if rep_audit else [
        "device_id",
        "representation_category",
        "first_representation_snippet",
    ]
    with (output_dir / "representation_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(rep_audit)

    summaries = {
        "primary_fairness_frame_a_normative": summarize_corpus(
            rows, "has_a_normative_fairness"
        ),
        "sensitivity_legacy_frame_a_with_representation": summarize_corpus(
            rows, "has_a_legacy_with_representation"
        ),
        "alternative_a_representation_demographic_only": summarize_corpus(
            rows, "has_a_representation_demographic"
        ),
    }

    rep_cat_counts: dict[str, int] = {}
    for r in rep_audit:
        cat = r["representation_category"]
        rep_cat_counts[cat] = rep_cat_counts.get(cat, 0) + 1

    meta = {
        "operationalization_notes": {
            "primary_for_fairness_paper": (
                "Frame A = fairness, equity, basic/fundamental rights, representativeness "
                "(excludes bare 'representation')."
            ),
            "legacy_sensitivity": "Frame A includes bare 'representation' (§4 original).",
            "representation_demographic": (
                "Normative terms OR 'representation' with demographic-fairness context window."
            ),
        },
        "representation_token_documents": len(rep_audit),
        "representation_category_counts": rep_cat_counts,
        "summaries": summaries,
    }

    summary_rows: list[dict[str, str]] = []
    for label, stats in summaries.items():
        for metric, value in stats.items():
            summary_rows.append(
                {
                    "operationalization": label,
                    "metric": metric,
                    "value": str(value),
                }
            )
    with (output_dir / "frame_fairness_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["operationalization", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    def subgroup_rate(pred, metric: str) -> tuple[int, float]:
        sub = [r for r in rows if pred(r)]
        if not sub:
            return 0, 0.0
        hits = sum(1 for r in sub if r[metric])
        return len(sub), hits / len(sub)

    crosstab_rows: list[dict[str, str]] = []
    slices = [
        ("all_corpus", lambda r: True),
        ("scenario_explicit_yes", lambda r: r["has_scenario_explicit"]),
        ("scenario_explicit_no", lambda r: not r["has_scenario_explicit"]),
        ("scenario_population_yes", lambda r: r["has_scenario_population"]),
        ("scenario_population_no", lambda r: not r["has_scenario_population"]),
        ("scenario_workflow_yes", lambda r: r["has_scenario_workflow"]),
        ("non_a_normative", lambda r: not r["has_a_normative_fairness"]),
        ("non_a_normative_and_C", lambda r: not r["has_a_normative_fairness"] and r["has_c"]),
        (
            "non_a_normative_and_C_and_B",
            lambda r: not r["has_a_normative_fairness"] and r["has_c"] and r["has_b"],
        ),
    ]
    metrics = [
        ("has_a_normative_fairness", "normative_a"),
        ("has_fairness_broad", "fairness_broad"),
        ("has_b", "frame_b"),
        ("has_demographic", "demographic"),
        ("has_subgroup", "subgroup"),
    ]
    n_total = len(rows)
    for slice_name, pred in slices:
        n_sub, _ = subgroup_rate(pred, "has_c")
        for field, metric_label in metrics:
            _, rate = subgroup_rate(pred, field)
            crosstab_rows.append(
                {
                    "slice": slice_name,
                    "n_docs": str(n_sub),
                    "pct_of_corpus": f"{100.0 * n_sub / n_total:.4f}",
                    "metric": metric_label,
                    "doc_count": str(int(round(rate * n_sub))),
                    "rate": f"{rate:.6f}",
                }
            )

    with (output_dir / "scenario_fairness_crosstab.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice",
                "n_docs",
                "pct_of_corpus",
                "metric",
                "doc_count",
                "rate",
            ],
        )
        writer.writeheader()
        writer.writerows(crosstab_rows)

    meta["scenario_fairness_crosstab"] = {
        "notes": (
            "scenario_explicit = scenario/use case tokens; scenario_population = "
            "intended use population / patient population / clinically relevant; "
            "fairness_broad = fairness|equity|representativeness|bias|disparity|mitigation; "
            "intended use alone is ~99% and omitted as a slice."
        ),
        "slice_names": [name for name, _ in slices],
    }

    (output_dir / "frame_fairness_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    primary = summaries["primary_fairness_frame_a_normative"]
    print("Fairness-focused Frame A (no bare representation):")
    print(f"  A: {primary['a_doc_count']} ({100*primary['a_doc_rate']:.2f}%)")
    print(f"  non-A: {primary['non_a_doc_count']} ({100*primary['non_a_doc_rate']:.2f}%)")
    print(f"  C∩A: {primary['c_and_a']} ({100*primary['c_and_a_rate']:.2f}%)")
    print(f"Wrote {output_dir}/representation_audit.csv ({len(rep_audit)} rows)")
    print(f"Wrote {output_dir}/frame_fairness_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-text-dir", type=Path, default=Path("data/raw_text"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    raw_dir = (base / args.raw_text_dir).resolve()
    out_dir = (base / args.output_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    run(raw_dir, out_dir)


if __name__ == "__main__":
    main()
