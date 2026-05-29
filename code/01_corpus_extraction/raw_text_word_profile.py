#!/usr/bin/env python3
"""
Exploratory word-frequency and TF-IDF profile for FDA device summary raw text.

Outputs corpus terrain maps (filtered counts, document-level TF-IDF) and compares
high-signal terms to the paper's Frame A/B/C lexicons (§4). Not a substitute for
dictionary-based frame analysis — see analyze_abstract_politics.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"\b[a-z][a-z0-9'-]{2,}\b", re.IGNORECASE)

# Regulatory / genre boilerplate (extend as needed).
REGULATORY_STOPWORDS: frozenset[str] = frozenset(
    w.lower()
    for w in """
    a about above across after again against all also am an and any are as at be
    been before being below between both but by can could did do does doing done
    down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just ll me more
    most my myself no nor not of off on once only or other our ours ourselves
    out over own re s same she should so some such t than that the their theirs
    them themselves then there these they this those through to too under until
    up us very was we were what when where which while who whom why will with
    within would you your yours yourself yourselves
    act addition additionally administration adverse after also amendment
    applicable application appropriate approval approximately assessment
    associated available based behalf bloomington cdrh center change changes
    class classification clinical code combination commercial commission
    committee company compliance comply conducted contact contains content
    control controls cosmetic data database dated dear decision deemed
    demonstrate demonstrated department described description design determined
    device devices diagnostic documentation documents drug effective
    electronically enactment enclosed entity equivalent established evaluation
    evidence examination exceed existing external federal food following
    general guidance health healthcare however https http including indicated
    indication indications information intended interstate issued january july
    june labeling legally letter listed listing manufacture manufactured
    manufacturer manufacturers marketing material may medical minnesota
    modification name new nh newhampshire notification notified november
    number october office part particular patient patients please pmn premarket
    president prior product products proposed provisions public pursuant
    received reference referenced regarding regulation regulatory requirements
    review reviewed rights risk safety section silver spring software special
    specific spring state stated states subject submission substantially
    summary supplement system table thereof therefore     thereof title trade
    under united unless updated us usa use used using validation verification
    version view website whether within www year years your
    cfr fda gov page pages see yes image images www accessdata scripts cfdocs
    """.split()
)

FRAME_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    # Align with analyze_abstract_politics.py policy_discourse_terms + §4.
    "frame_a_normative": [
        re.compile(r"\bfairness\b", re.I),
        re.compile(r"\bequity\b", re.I),
        re.compile(r"\bbasic\s+rights\b", re.I),
        re.compile(r"\bfundamental\s+rights\b", re.I),
        re.compile(r"\brepresentativeness\b", re.I),
        re.compile(r"\brepresentation\b", re.I),
    ],
    "frame_b_governance": [
        re.compile(r"\bbias(?:es|ed)?\b", re.I),
        re.compile(r"\bmitigat(?:e|ion|ing)\b", re.I),
        re.compile(r"\bgovernance\b", re.I),
        re.compile(r"\bpost[\s-]?market\b", re.I),
        re.compile(r"\bmonitor(?:ing|ed)?\b", re.I),
        re.compile(r"\brisk\s+management\b", re.I),
    ],
    "frame_c_technical": [
        re.compile(r"\bsubgroup(?:s)?\b", re.I),
        re.compile(r"\baccuracy\b", re.I),
        re.compile(r"\bsensitivity\b", re.I),
        re.compile(r"\bspecificity\b", re.I),
        re.compile(r"\bauc\b", re.I),
        re.compile(r"\bperformance\b", re.I),
    ],
}

FRAME_LEXICON_TOKENS: dict[str, frozenset[str]] = {
    "frame_a_normative": frozenset(
        "fairness equity representativeness representation".split()
    ),
    "frame_b_governance": frozenset(
        "bias mitigation governance postmarket monitoring risk".split()
    ),
    "frame_c_technical": frozenset(
        "subgroup accuracy sensitivity specificity auc performance".split()
    ),
}

DOMAIN_BUCKETS: dict[str, frozenset[str]] = {
    "technical_performance": frozenset(
        """
        performance accuracy sensitivity specificity auc subgroup validation
        verified verification algorithm model training testing test tests
        sensitivity specificity false positive negative predictive value
        """.split()
    ),
    "governance_compliance": frozenset(
        """
        mitigation governance audit monitoring surveillance postmarket risk
        management quality cybersecurity calibration threshold reweighting
        """.split()
    ),
    "normative_fairness": frozenset(FRAME_LEXICON_TOKENS["frame_a_normative"]),
    "demographics_social": frozenset(
        """
        race ethnicity gender age demographic demographics cohort diverse
        diversity underrepresented hispanic latino african asian white
        """.split()
    ),
    "clinical_evidence": frozenset(
        """
        clinical study studies patient patients prospective retrospective
        validation endpoint endpoints trial trials cohort imaging mri ct
        """.split()
    ),
    "regulatory_process": frozenset(
        """
        predicate substantially equivalent clearance cleared labeling
        manufacturing design controls corrective preventive action
        """.split()
    ),
}


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def filter_tokens(tokens: list[str], extra_stop: frozenset[str]) -> list[str]:
    stop = REGULATORY_STOPWORDS | extra_stop
    return [t for t in tokens if t not in stop and not t.isdigit()]


def doc_has_frame(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def compute_frame_coverage(docs: list[tuple[str, str]]) -> dict[str, object]:
    n = len(docs)
    flags: dict[str, list[bool]] = {k: [] for k in FRAME_PATTERNS}
    for _did, text in docs:
        for frame, pats in FRAME_PATTERNS.items():
            flags[frame].append(doc_has_frame(text, pats))

    def rate(key: str) -> float:
        return sum(flags[key]) / n if n else 0.0

    a, b, c = flags["frame_a_normative"], flags["frame_b_governance"], flags["frame_c_technical"]
    cap = sum(1 for i in range(n) if a[i] and b[i] and c[i])
    ca = sum(1 for i in range(n) if c[i] and a[i])
    cb = sum(1 for i in range(n) if c[i] and b[i])

    return {
        "documents_total": n,
        "frame_a_doc_count": sum(a),
        "frame_a_doc_rate": rate("frame_a_normative"),
        "frame_b_doc_count": sum(b),
        "frame_b_doc_rate": rate("frame_b_governance"),
        "frame_c_doc_count": sum(c),
        "frame_c_doc_rate": rate("frame_c_technical"),
        "non_a_doc_count": sum(1 for x in a if not x),
        "non_a_doc_rate": sum(1 for x in a if not x) / n if n else 0.0,
        "intersection_c_and_a": ca,
        "intersection_c_and_a_rate": ca / n if n else 0.0,
        "intersection_c_and_b": cb,
        "intersection_c_and_b_rate": cb / n if n else 0.0,
        "intersection_c_and_a_and_b": cap,
        "intersection_c_and_a_and_b_rate": cap / n if n else 0.0,
    }


def compute_tfidf_means(
    doc_tokens: list[list[str]],
) -> dict[str, float]:
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return {}

    df: Counter[str] = Counter()
    tf_per_doc: list[Counter[str]] = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        tf_per_doc.append(tf)
        for term in tf:
            df[term] += 1

    idf: dict[str, float] = {
        term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0 for term, count in df.items()
    }

    tfidf_sum: Counter[str] = Counter()
    for tf in tf_per_doc:
        doc_len = sum(tf.values()) or 1
        for term, cnt in tf.items():
            tfidf_sum[term] += (cnt / doc_len) * idf[term]

    return {term: tfidf_sum[term] / n_docs for term, _ in tfidf_sum.items()}


def classify_domain(term: str) -> str:
    hits = [name for name, lex in DOMAIN_BUCKETS.items() if term in lex]
    return "|".join(hits) if hits else ""


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(raw_text_dir: Path, output_dir: Path, top_n: int) -> None:
    paths = sorted(raw_text_dir.glob("*.txt"))
    docs: list[tuple[str, str]] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append((p.stem, text))

    all_tokens_raw: list[str] = []
    all_tokens_filtered: list[str] = []
    doc_tokens_filtered: list[list[str]] = []
    per_doc_word_counts: list[dict[str, str | int]] = []

    for device_id, text in docs:
        raw = tokenize(text)
        filt = filter_tokens(raw, frozenset())
        all_tokens_raw.extend(raw)
        all_tokens_filtered.extend(filt)
        doc_tokens_filtered.append(filt)
        per_doc_word_counts.append(
            {
                "device_id": device_id,
                "raw_token_count": len(raw),
                "filtered_token_count": len(filt),
                "unique_filtered": len(set(filt)),
            }
        )

    raw_counter = Counter(all_tokens_raw)
    filt_counter = Counter(all_tokens_filtered)
    tfidf_means = compute_tfidf_means(doc_tokens_filtered)
    frame_coverage = compute_frame_coverage(docs)

    total_filt = sum(filt_counter.values()) or 1
    raw_top = raw_counter.most_common(top_n)
    filt_top = filt_counter.most_common(top_n)
    tfidf_top = sorted(tfidf_means.items(), key=lambda x: x[1], reverse=True)[:top_n]

    raw_rows = [
        {
            "rank": i + 1,
            "term": term,
            "count": cnt,
            "pct_of_corpus": f"{100.0 * cnt / sum(raw_counter.values()):.4f}",
        }
        for i, (term, cnt) in enumerate(raw_top)
    ]
    filt_rows = [
        {
            "rank": i + 1,
            "term": term,
            "count": cnt,
            "pct_of_filtered": f"{100.0 * cnt / total_filt:.4f}",
            "domain_tags": classify_domain(term),
        }
        for i, (term, cnt) in enumerate(filt_top)
    ]
    tfidf_rows = [
        {
            "rank": i + 1,
            "term": term,
            "mean_tfidf": f"{score:.6f}",
            "filtered_count": filt_counter.get(term, 0),
            "domain_tags": classify_domain(term),
        }
        for i, (term, score) in enumerate(tfidf_top)
    ]

    lexicon_rows: list[dict[str, str | int | float]] = []
    for frame, tokens in FRAME_LEXICON_TOKENS.items():
        for term in sorted(tokens):
            lexicon_rows.append(
                {
                    "frame": frame,
                    "term": term,
                    "token_count": filt_counter.get(term, 0),
                    "pct_of_filtered": f"{100.0 * filt_counter.get(term, 0) / total_filt:.4f}",
                    "mean_tfidf": f"{tfidf_means.get(term, 0.0):.6f}",
                    "doc_rate_any_token": "",
                }
            )

    bucket_totals: dict[str, int] = defaultdict(int)
    for term, cnt in filt_counter.items():
        tags = classify_domain(term)
        if not tags:
            bucket_totals["other_untagged"] += cnt
        else:
            for tag in tags.split("|"):
                bucket_totals[tag] += cnt

    bucket_rows = [
        {
            "bucket": name,
            "token_occurrences": bucket_totals[name],
            "pct_of_filtered": f"{100.0 * bucket_totals[name] / total_filt:.4f}",
        }
        for name in sorted(bucket_totals)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "word_freq_raw_top.csv",
        ["rank", "term", "count", "pct_of_corpus"],
        raw_rows,
    )
    write_csv_rows(
        output_dir / "word_freq_filtered_top.csv",
        ["rank", "term", "count", "pct_of_filtered", "domain_tags"],
        filt_rows,
    )
    write_csv_rows(
        output_dir / "word_freq_tfidf_top.csv",
        ["rank", "term", "mean_tfidf", "filtered_count", "domain_tags"],
        tfidf_rows,
    )
    write_csv_rows(
        output_dir / "word_freq_frame_lexicon.csv",
        ["frame", "term", "token_count", "pct_of_filtered", "mean_tfidf", "doc_rate_any_token"],
        lexicon_rows,
    )
    write_csv_rows(
        output_dir / "word_freq_domain_buckets.csv",
        ["bucket", "token_occurrences", "pct_of_filtered"],
        bucket_rows,
    )
    write_csv_rows(
        output_dir / "word_freq_per_doc_stats.csv",
        ["device_id", "raw_token_count", "filtered_token_count", "unique_filtered"],
        per_doc_word_counts,
    )

    summary_lines = [
        "Word-frequency profile for data/raw_text (exploratory; not Frame A/B/C primary analysis).",
        f"Documents: {len(docs)}",
        f"Raw tokens: {sum(raw_counter.values()):,} | Filtered tokens: {total_filt:,}",
        f"Unique filtered types: {len(filt_counter):,}",
        "",
        "Frame coverage (§4 lexicon, document-level any-hit):",
        f"  Frame A (normative): {frame_coverage['frame_a_doc_count']} "
        f"({100 * frame_coverage['frame_a_doc_rate']:.2f}%)",
        f"  Frame B (governance): {frame_coverage['frame_b_doc_count']} "
        f"({100 * frame_coverage['frame_b_doc_rate']:.2f}%)",
        f"  Frame C (technical): {frame_coverage['frame_c_doc_count']} "
        f"({100 * frame_coverage['frame_c_doc_rate']:.2f}%)",
        f"  non-A: {frame_coverage['non_a_doc_count']} "
        f"({100 * frame_coverage['non_a_doc_rate']:.2f}%)",
        f"  C∩A: {frame_coverage['intersection_c_and_a']} "
        f"({100 * frame_coverage['intersection_c_and_a_rate']:.2f}%)",
        f"  C∩B: {frame_coverage['intersection_c_and_b']} "
        f"({100 * frame_coverage['intersection_c_and_b_rate']:.2f}%)",
        f"  C∩A∩B: {frame_coverage['intersection_c_and_a_and_b']} "
        f"({100 * frame_coverage['intersection_c_and_a_and_b_rate']:.2f}%)",
        "",
        "Top 15 filtered terms:",
    ]
    for i, (term, cnt) in enumerate(filt_top[:15], 1):
        tags = classify_domain(term) or "(untagged)"
        summary_lines.append(f"  {i:2d}. {term}: {cnt:,} [{tags}]")

    summary_lines.extend(["", "Top 15 TF-IDF terms (mean across documents):"])
    for i, (term, score) in enumerate(tfidf_top[:15], 1):
        tags = classify_domain(term) or "(untagged)"
        summary_lines.append(f"  {i:2d}. {term}: {score:.4f} [{tags}]")

    summary_path = output_dir / "word_freq_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "documents": len(docs),
        "raw_token_total": sum(raw_counter.values()),
        "filtered_token_total": total_filt,
        "unique_filtered_types": len(filt_counter),
        "frame_coverage": frame_coverage,
        "top_filtered_terms": [t for t, _ in filt_top[:20]],
        "top_tfidf_terms": [t for t, _ in tfidf_top[:20]],
    }
    (output_dir / "word_freq_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote outputs under {output_dir}")
    print(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-text-dir",
        type=Path,
        default=Path("data/raw_text"),
        help="Directory of per-device .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Where to write CSV/JSON summary outputs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top terms to export per list",
    )
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    raw_dir = (base / args.raw_text_dir).resolve() if not args.raw_text_dir.is_absolute() else args.raw_text_dir
    out_dir = (base / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw text directory not found: {raw_dir}")
    run(raw_dir, out_dir, args.top_n)


if __name__ == "__main__":
    main()
