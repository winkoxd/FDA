#!/usr/bin/env python3
"""
KWIC samples for normative-absent / auditable-saturated documents (flow Path β).

Reads frame_fairness_per_doc.csv, assigns discourse_flow_stage, exports KWIC rows.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

ANCHOR_TERMS = [
    ("bias", re.compile(r"\bbias(?:es|ed)?\b", re.I)),
    ("subgroup", re.compile(r"\bsubgroup", re.I)),
    ("performance", re.compile(r"\bperformance\b", re.I)),
    ("mitigation", re.compile(r"\bmitigat", re.I)),
    ("monitoring", re.compile(r"\bmonitor", re.I)),
    ("risk_management", re.compile(r"\brisk\s+management", re.I)),
]

LIMITATION_CTX = re.compile(
    r"\b(?:limitation|limitations|missing\s+data|incomplete|underrepresent|"
    r"generaliz|constraint|uncertain|not\s+evaluated|small\s+sample|"
    r"drawback|confounder|not\s+defined\s+for\s+the\s+datasets)\b",
    re.I,
)
OPTIMIZATION_CTX = re.compile(
    r"\b(?:model|training|architecture|objective|optimiz|calibrat|"
    r"reweight|augment|hyperparameter|algorithm|neural\s+network|cnn|"
    r"machine\s+learning|retraining|pipeline)\b",
    re.I,
)
GOVERNANCE_CTX = re.compile(
    r"\b(?:post[\s-]?market|surveillance|risk\s+management|quality\s+system|"
    r"corrective|preventive|iso\s+14971|iec\s+62304|cybersecurity|"
    r"verification\s+and\s+validation|v&?v\b|design\s+control|substantial\s+equivalence)\b",
    re.I,
)
# "Bias" in imaging/statistics — not social-fairness discourse
TECHNICAL_BIAS_CTX = re.compile(
    r"\b(?:bias\s+field|inhomogeneity|bland[\s-]altman|intercept|slope|"
    r"analytic\s+performance\s+metrics|acceleration\s+factor|roi\s+signal|"
    r"bias:\s*[\d.])\b",
    re.I,
)
SUBGROUP_DEMO_CTX = re.compile(
    r"\b(?:subgroup\s+analysis|age\s+and\s+sex|race\s+or\s+ethnicity|"
    r"gender|demographic|diverse\s+subgroups)\b",
    re.I,
)
# Hardware display monitor — not therapeutic monitoring
DEVICE_MONITOR_CTX = re.compile(
    r"\b(?:display\s+monitor|lcd\s+monitor|endoscopic\s+display|"
    r"monitors/screens|video\s+processor|calibrated\s+display|hmd)\b",
    re.I,
)
AI_FAIRNESS_LABEL_CTX = re.compile(
    r"\b(?:bias\s*/\s*limitation|ai\s*/\s*ml\s+model|fairness|equity|"
    r"population\s+ai)\b",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def classify_habitat(sentence: str, anchor_term: str = "") -> str:
    """Rhetorical habitat for §2.9.1 depoliticization coding."""
    sent = sentence
    has_lim = bool(LIMITATION_CTX.search(sent))
    has_opt = bool(OPTIMIZATION_CTX.search(sent))
    has_gov = bool(GOVERNANCE_CTX.search(sent))

    if anchor_term == "bias" and TECHNICAL_BIAS_CTX.search(sent):
        return "technical_statistical_bias"
    if AI_FAIRNESS_LABEL_CTX.search(sent) and anchor_term in ("bias", "mitigation"):
        return "ai_fairness_label_section"
    if anchor_term == "monitoring" and DEVICE_MONITOR_CTX.search(sent):
        return "device_hardware_monitor"
    if anchor_term == "subgroup" and SUBGROUP_DEMO_CTX.search(sent):
        return "subgroup_stratification"
    if has_lim and has_opt:
        return "mixed_limitation_optimization"
    if has_lim and not has_opt:
        return "limitation_epistemic"
    if has_opt and not has_lim:
        return "model_optimization"
    if has_gov and not has_lim and not has_opt:
        return "governance_compliance"
    if has_gov:
        return "governance_mixed"
    if anchor_term == "performance" and re.search(
        r"\b(?:v&?v|verification|validation|bench|non-?clinical)\b", sent, re.I
    ):
        return "governance_compliance"
    if anchor_term == "subgroup":
        return "subgroup_stratification"
    if anchor_term == "mitigation" and re.search(r"\bcyber", sent, re.I):
        return "governance_compliance"
    return "generic_regulatory_other"


def habitat_group(habitat: str) -> str:
    """Collapse to §2.9.1 main buckets for Findings."""
    if habitat in {
        "limitation_epistemic",
        "mixed_limitation_optimization",
        "ai_fairness_label_section",
    }:
        return "A_limitation_epistemic"
    if habitat == "model_optimization":
        return "B_model_optimization"
    if habitat in {
        "governance_compliance",
        "governance_mixed",
        "governance_postmarket",
    }:
        return "C_governance_compliance"
    if habitat in {"technical_statistical_bias", "device_hardware_monitor"}:
        return "D_technical_depoliticized"
    if habitat == "subgroup_stratification":
        return "E_subgroup_performance"
    return "F_other_regulatory"


def flow_stage(row: dict[str, str]) -> str:
    if row["has_a_normative_fairness"] == "True":
        return "alpha_normative_anchor"
    if row["has_fairness_broad"] == "False" and row["has_c"] == "False":
        return "beta0_no_fairness_lexicon"
    if row["has_c"] == "True" and row["has_b"] == "True":
        return "beta2_c_and_b"
    if row["has_c"] == "True":
        return "beta1_c_only"
    if row["has_b"] == "True":
        return "beta1_b_without_c"
    return "beta_other"


def kwic_rows_for_doc(
    device_id: str, text: str, flow: str, max_per_term: int = 2
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sentences = split_sentences(text)
    for term_name, pat in ANCHOR_TERMS:
        count = 0
        for sent in sentences:
            if not pat.search(sent):
                continue
            habitat = classify_habitat(sent, term_name)
            rows.append(
                {
                    "device_id": device_id,
                    "discourse_flow_stage": flow,
                    "anchor_term": term_name,
                    "rhetorical_habitat": habitat,
                    "habitat_group": habitat_group(habitat),
                    "sentence": " ".join(sent.split())[:500],
                }
            )
            count += 1
            if count >= max_per_term:
                break
    return rows


def assign_flow_table(per_doc_csv: Path, out_csv: Path) -> None:
    rows = list(csv.DictReader(per_doc_csv.open(encoding="utf-8")))
    out_rows = []
    stage_counts: dict[str, int] = {}
    for r in rows:
        stage = flow_stage(r)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        out_rows.append({**r, "discourse_flow_stage": stage})
    fields = list(rows[0].keys()) + ["discourse_flow_stage"] if rows else ["discourse_flow_stage"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print("Flow stage counts:")
    for k in sorted(stage_counts, key=lambda x: -stage_counts[x]):
        n = stage_counts[k]
        print(f"  {k}: {n} ({100*n/len(rows):.2f}%)")


KWIC_FIELDS = [
    "device_id",
    "discourse_flow_stage",
    "anchor_term",
    "rhetorical_habitat",
    "habitat_group",
    "sentence",
]


def write_kwic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KWIC_FIELDS)
        w.writeheader()
        w.writerows(rows)


def collect_kwic(
    doc_rows: list[dict[str, str]],
    raw_dir: Path,
    max_per_term: int = 2,
) -> list[dict[str, str]]:
    kwic_all: list[dict[str, str]] = []
    for r in doc_rows:
        path = raw_dir / f"{r['device_id']}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore")
        flow = flow_stage(r)
        kwic_all.extend(
            kwic_rows_for_doc(r["device_id"], text, flow, max_per_term=max_per_term)
        )
    return kwic_all


def summarize_habitats(kwic_rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    fine: dict[str, int] = {}
    grouped: dict[str, int] = {}
    for row in kwic_rows:
        fine[row["rhetorical_habitat"]] = fine.get(row["rhetorical_habitat"], 0) + 1
        g = row["habitat_group"]
        grouped[g] = grouped.get(g, 0) + 1
    return fine, grouped


def write_habitat_summary_csv(
    path: Path,
    fine: dict[str, int],
    grouped: dict[str, int],
    meta: dict[str, str],
) -> None:
    rows: list[dict[str, str]] = []
    n = int(meta.get("n_lines", sum(fine.values()) or 1))
    for level, counts in [("fine", fine), ("group", grouped)]:
        for habitat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            rows.append(
                {
                    "level": level,
                    "habitat": habitat,
                    "count": str(cnt),
                    "pct": f"{100.0 * cnt / n:.2f}",
                    "n_docs": meta.get("n_docs", ""),
                    "slice": meta.get("slice", ""),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["level", "habitat", "count", "pct", "n_docs", "slice"],
        )
        w.writeheader()
        w.writerows(rows)


def write_findings_panels(path: Path, kwic_rows: list[dict[str, str]]) -> None:
    """One exemplar sentence per fine habitat (first hit)."""
    seen: set[str] = set()
    panels: list[dict[str, str]] = []
    for row in kwic_rows:
        h = row["rhetorical_habitat"]
        if h in seen:
            continue
        seen.add(h)
        panels.append(
            {
                "rhetorical_habitat": h,
                "habitat_group": row["habitat_group"],
                "device_id": row["device_id"],
                "anchor_term": row["anchor_term"],
                "sentence": row["sentence"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(panels[0].keys()) if panels else [])
        w.writeheader()
        w.writerows(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-doc-csv", type=Path, default=Path("data/analysis/frame_fairness_per_doc.csv"))
    parser.add_argument("--raw-text-dir", type=Path, default=Path("data/raw_text"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--slice", default="beta2_c_and_b", help="discourse_flow_stage to sample")
    parser.add_argument("--sample-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    per_doc = (base / args.per_doc_csv).resolve()
    raw_dir = (base / args.raw_text_dir).resolve()
    out_dir = (base / args.output_dir).resolve()

    flow_csv = out_dir / "frame_fairness_per_doc_with_flow.csv"
    assign_flow_table(per_doc, flow_csv)

    all_rows = list(csv.DictReader(per_doc.open(encoding="utf-8")))
    pool = [r for r in all_rows if flow_stage(r) == args.slice]

    random.seed(args.seed)
    sample_docs = random.sample(pool, min(args.sample_n, len(pool)))
    sample_kwic = collect_kwic(sample_docs, raw_dir, max_per_term=2)
    sample_path = out_dir / "kwic_flow_beta2_c_b_sample.csv"
    write_kwic_csv(sample_path, sample_kwic)

    corpus_kwic = collect_kwic(pool, raw_dir, max_per_term=1)
    corpus_path = out_dir / "kwic_flow_beta2_c_b_corpus.csv"
    write_kwic_csv(corpus_path, corpus_kwic)

    fine_c, group_c = summarize_habitats(corpus_kwic)
    write_habitat_summary_csv(
        out_dir / "kwic_habitat_summary.csv",
        fine_c,
        group_c,
        {
            "n_lines": str(len(corpus_kwic)),
            "n_docs": str(len(pool)),
            "slice": args.slice,
        },
    )
    write_findings_panels(out_dir / "kwic_findings_panels.csv", corpus_kwic)

    fine_s, group_s = summarize_habitats(sample_kwic)
    lines = [
        f"KWIC sample: {args.slice}, docs={len(sample_docs)}, lines={len(sample_kwic)}",
        f"KWIC full slice: docs={len(pool)}, lines={len(corpus_kwic)}",
        "",
        "Sample — habitat groups:",
    ]
    for k, v in sorted(group_s.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v} ({100*v/max(len(sample_kwic),1):.1f}%)")
    lines.append("")
    lines.append("Full β2 — habitat groups:")
    for k, v in sorted(group_c.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v} ({100*v/max(len(corpus_kwic),1):.1f}%)")
    (out_dir / "kwic_flow_habitat_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {sample_path} ({len(sample_kwic)} lines)")
    print(f"Wrote {corpus_path} ({len(corpus_kwic)} lines from {len(pool)} docs)")
    print("Full β2 habitat groups:")
    for k, v in sorted(group_c.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
