```markdown
# Osteoporosis Gene Analysis with SHAP Visualization

This Streamlit application provides interpretable SHAP analysis for osteoporosis risk prediction based on 20 key genes related to bone metabolism regulation.

```

## Setup and Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

## Features

- Interactive adjustment of 20 osteoporosis-related gene expressions
- SHAP force plots visualizing directional gene impacts
- Waterfall plots showing stepwise feature contributions
- Decision plots with cumulative effect visualization
- Mechanistic insights combining SHAP values with biological pathways

## Key Gene Indicators

### Core Osteoporosis-related Genes:
```
VAMP1       ATP10B      ABCC4       LCT        NTNG1
GLRA2      ELSPBP1     IL13        MLN        ZNF280A
CA7        ELAVL4      SPDEF       SRPK3      FSCN3
CLDN18     IL19        CD80        ZFYVE9     CARTPT
```

## Interpretation Guidance
Results should be evaluated in conjunction with:
1. SHAP value magnitude (absolute importance)
2. Directionality (positive/negative impact)
3. Known biological pathways (Mechanistic Insights tab)
4. Clinical risk thresholds (≥0.5 = High Risk)
