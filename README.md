
# Hate Speech Detection System with Bias Analysis

Multi-label toxicity detection using DistilBERT. Achieves 13.3% improvement over baseline with comprehensive bias testing.

**Results:** Macro F1: 0.70 (baseline) → 0.79 (DistilBERT)


## Quick Start
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data from Kaggle
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Place train.csv, test.csv, test_labels.csv in data/

# Run notebooks in order
jupyter notebook

# Launch web interface
streamlit run app.py
```

## Key Results

| Label | Baseline F1 | DistilBERT F1 | Improvement |
|-------|-------------|---------------|-------------|
| Toxic | 0.716 | 0.798 | +11.5% |
| Threat | 0.339 | 0.519 | +53.1% |

**Bias Findings:**
- Significant religious identity bias (p=0.024)
- 60% false positive on positive profanity
- Marginal AAVE dialect bias (+3.8%)

