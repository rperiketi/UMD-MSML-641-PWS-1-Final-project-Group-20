
# Hate Speech Detection System with Bias Analysis

Multi-label toxicity detection using DistilBERT. Achieves 25% improvement over baseline with comprehensive bias testing.

**Results:** Macro F1: 0.52 (baseline) → 0.65 (DistilBERT)


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

**Bias Findings:**
- Significant religious identity bias
- False positive on positive profanity
- Marginal AAVE dialect bias

**Per-Label Improvements:**
- Best gains on rare labels:
- Threat: +31% (0.348 → 0.456)
- Identity Hate: +71.8% (0.341 → 0.587)
- Strong on frequent labels:
- Obscene: 0.724
- Insult: 0.643


