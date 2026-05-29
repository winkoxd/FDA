#!/usr/bin/env bash
# Reproduce headline statistics (N=1388).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

n=$(find data/raw_text -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
[[ "$n" == "1388" ]] || echo "WARNING: expected 1388 raw_text files, found $n" >&2

python3 code/01_register_frames/frame_fairness_operationalization.py
python3 code/02_discourse_flow_kwic/kwic_non_a_flow.py
python3 code/03_mechanisms_supplementary/supplementary_paper_analyses.py
python3 code/03_mechanisms_supplementary/specialty_stratification_analysis.py
python3 code/04_race_ethnicity_tiers/verify_race_ethnicity_reporting.py
python3 code/05_verify_replication/verify_manuscript_stats.py

echo "OK — see data/analysis/frame_fairness_summary.json"
