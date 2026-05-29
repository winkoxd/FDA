# Analysis code index

Run all core steps from the repository root:

```bash
bash scripts/reproduce.sh
```

## Pipeline (in order)

| Step | Script | Output |
|------|--------|--------|
| 1 | `01_register_frames/frame_fairness_operationalization.py` | Frame A/B/C flags, `representation_audit.csv`, `frame_fairness_summary.json` |
| 2 | `02_discourse_flow_kwic/kwic_non_a_flow.py` | `frame_fairness_per_doc_with_flow.csv`, KWIC CSVs |
| 3 | `03_mechanisms_supplementary/supplementary_paper_analyses.py` | Agency attribution, GMLP era, `paper_supplementary_summary.json` |
| 3b | `03_mechanisms_supplementary/specialty_stratification_analysis.py` | `specialty_stratification_summary.json` |
| 4 | `04_race_ethnicity_tiers/verify_race_ethnicity_reporting.py` | `race_ethnicity_tiered_*.csv/json` |
| 5 | `05_verify_replication/verify_manuscript_stats.py` | Console check vs headline stats |

## Optional

| Script | Purpose |
|--------|---------|
| `01_corpus_extraction/reextract_pdfs_to_raw_text.py` | Rebuild `data/raw_text` from local PDFs |
| `01_corpus_extraction/raw_text_word_profile.py` | Word-frequency / TF-IDF exploration |
| `06_figures/render_perspective_figure1.py` | Saturation–gap schematic (PNG/PDF) |

Coding definitions: see root `README.md` § Coding scheme.
