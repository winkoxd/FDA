# FDA AI Device Public Summaries

[![Repository](https://img.shields.io/badge/GitHub-winkoxd%2FFDA-blue)](https://github.com/winkoxd/FDA.git)

Lexical analysis of **N = 1,388** U.S. FDA public summaries for AI/ML-enabled medical devices. The study documents a **saturation–gap** pattern: dense technical–procedural disclosure can coexist with scarce normative fairness grammar on the public regulatory spine.

**Data & code availability:** see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

---

## Quick start

```bash
git clone https://github.com/winkoxd/FDA.git
cd FDA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce.sh
python3 code/05_verify_replication/verify_manuscript_stats.py
```

---

## Repository layout

```
FDA/
├── README.md
├── DATA_AVAILABILITY.md
├── requirements.txt
├── pdf_text_extract.py          # optional PDF → text
├── data/
│   ├── raw_text/                # N=1388 corpus (UTF-8)
│   ├── analysis/                # tables, JSON, figures/
│   └── device_list/             # FDA AI device list CSV
├── code/                        # categorized scripts (see code/README.md)
│   ├── 01_register_frames/
│   ├── 02_discourse_flow_kwic/
│   ├── 03_mechanisms_supplementary/
│   ├── 04_race_ethnicity_tiers/
│   ├── 05_verify_replication/
│   ├── 06_figures/
│   └── 01_corpus_extraction/
└── scripts/
    └── reproduce.sh
```

---

## Headline statistics (primary coding)

| Metric | Value | File |
|--------|-------|------|
| Corpus | N = 1388 | `data/raw_text/` |
| Path α (normative anchor) | 0.58% (n=8) | `data/analysis/frame_fairness_summary.json` |
| Path β | 99.42% | same |
| Frame C (performance register) | 97.77% | same |
| β2 band (C+B) | 66.93% | `data/analysis/frame_fairness_per_doc_with_flow.csv` |
| Race/ethnicity Tier A (token probe) | 20.17% | `data/analysis/race_ethnicity_tiered_summary.json` |
| Explicit race/ethnicity phrase (Tier E) | 3.03% | same |

**Primary Frame A** excludes bare `representation` (ML vs demographic polysemy). Definitions: `code/01_register_frames/frame_fairness_operationalization.py` and § Coding scheme below.

---

## Coding scheme

| Register | Role | Lexicon (whole-word, case-insensitive) |
|----------|------|----------------------------------------|
| **Frame A** | Normative fairness | `fairness`, `equity`, `basic rights`, `fundamental rights`, `representativeness` |
| **Frame B** | Governance / process | `bias`, `mitigat*`, `governance`, `post-market`, `monitor*`, `risk management` |
| **Frame C** | Performance / metrics | `subgroup`, `accuracy`, `sensitivity`, `specificity`, `AUC`, `performance` |

- **Path α** — any Frame A hit (primary spec: **no** bare `representation`).  
- **Path β** — no Path α; **β2** = both B and C (66.93%, n=929).  
- **Race tiers** — text-probe only; Tier A (20.17%) is **not** comparable to Muralidharan et al. (2024) structured field coding (3.6%).

---

## Citation

When citing this repository, include the GitHub URL and commit hash. Add the peer-reviewed paper DOI when available.

Repository: https://github.com/winkoxd/FDA.git
