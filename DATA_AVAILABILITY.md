# Data and Code Availability

The original contributions presented in the study are publicly available.

The quantitative text-analysis corpus (**N = 1,388** FDA public AI/ML device summary texts) and the complete replication code developed for this study are available at:

**https://github.com/winkoxd/FDA.git**

## Contents

| Component | Location |
|-----------|----------|
| UTF-8 corpus (one file per 510(k) ID) | `data/raw_text/` |
| Regenerated analysis tables & JSON | `data/analysis/` (also produced by `scripts/reproduce.sh`) |
| Categorized Python pipeline | `code/` |
| Figures (optional regenerate) | `data/analysis/figures/` · `code/06_figures/` |

## Reproduce

```bash
git clone https://github.com/winkoxd/FDA.git
cd FDA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce.sh
```

## External data

FDA source PDFs can be retrieved from FDA AccessData; this repository ships extracted text to ensure stable replication without re-downloading ~1,400 PDFs.
