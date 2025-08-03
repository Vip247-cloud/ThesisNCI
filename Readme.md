
# Edge IoT NIDS — Thesis Reproducibility (NCI 2025)

**Author:** Vipin Poswal  
**Thesis:** *The NIDS Framework for Identifying Anomalous Traffic in Resource Constrained Networks Using Stacking Ensembles*  
**Dataset:** IoTID20 (5 classes: DoS, MITM ARP Spoofing, Mirai, Normal, Scan)

## Overview
This repository contains the reproducible pipeline and artifacts for a lightweight, flow-based **network intrusion detection system (NIDS)** targeting **resource-constrained IoT** deployments. The pipeline follows a **KDD** process—preprocessing, feature reduction, training, and evaluation—with a focus on **accuracy–efficiency** trade-offs suitable for edge gateways.

## Key Results (Test Set)
- **Best model:** Random Forest (`n_estimators=50`, `max_depth=None`, `min_samples_split=4`, `bootstrap=False`)
- **Overall accuracy:** ~0.98  
- **Macro-F1:** ~0.98  
- **CV mean AUC:** ~0.9993  
- See: `docs/NIDS_results_sheet.pdf` and `figures/rf_test_metrics_table.png`

| Class               | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| DoS                 | 1.00      | 1.00   | 1.00     | 25087   |
| MITM ARP Spoofing   | 0.96      | 0.98   | 0.97     | 24904   |
| Mirai               | 0.98      | 0.97   | 0.97     | 24847   |
| Normal              | 1.00      | 0.99   | 0.99     | 25037   |
| Scan                | 0.98      | 0.98   | 0.98     | 24828   |
| **Overall Accuracy**| **0.98**  |        |          | 124703  |
| **Macro Average**   | **0.98**  | **0.98** | **0.98** | 124703 |
| **Weighted Average**| **0.98**  | **0.98** | **0.98** | 124703 |

> Figures to include (export from the notebook):  
> - `figures/confusion_matrix_test.png` (Confusion matrix for the test set)  
> - `figures/roc_multiclass_test.png` (One-vs-rest multiclass ROC)

## Pipeline
1. **Preprocessing:** cleaning & imputation; label consolidation to 5 classes; down-sampling + **SMOTE**; standardization.  
2. **Feature reduction:** Pearson correlation filtering (>|0.90|), resulting in ~55 features; optional PCA (reported in thesis).  
3. **Split & CV:** 70/30 stratified train/test; **5-fold stratified CV** for model selection.  
4. **Models:** Random Forest, KNN (k=5, distance, Manhattan), shallow ANN (MLP).  
5. **Evaluation:** Accuracy, per-class precision/recall/F1, confusion matrix, multiclass ROC/AUC.

## Quickstart
```bash
# (1) Create environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (2) Run the notebook end-to-end (generates metrics and figures)
# open notebooks/thesis_pipeline.ipynb

# (3) Export figures for the README (from within the notebook)
# Save as: figures/confusion_matrix_test.png, figures/roc_multiclass_test.png
```

### Minimal code snippet to export the figures
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import numpy as np

# Assuming you already have y_test, y_pred, y_score (probabilities) and class_names list
# 1) Confusion matrix (test)
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, xticks_rotation=45)
plt.tight_layout()
plt.savefig("figures/confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.close()

# 2) ROC (one-vs-rest)
Y_test = label_binarize(y_test, classes=np.arange(len(class_names)))
for i, name in enumerate(class_names):
    RocCurveDisplay.from_predictions(Y_test[:, i], y_score[:, i], name=f"Class {name}")
plt.plot([0,1],[0,1], linestyle="--")
plt.tight_layout()
plt.savefig("figures/roc_multiclass_test.png", dpi=300, bbox_inches="tight")
plt.close()
```

## Repo Structure (suggested)
```
ThesisNCI/
├─ notebooks/
│  └─ thesis_pipeline.ipynb
├─ figures/
│  ├─ rf_test_metrics_table.png
│  ├─ confusion_matrix_test.png     # <- export from notebook
│  └─ roc_multiclass_test.png       # <- export from notebook
├─ docs/
│  └─ NIDS_results_sheet.pdf
├─ data/                            # (optional, or downloaded on demand)
├─ src/                             # (optional helpers)
├─ requirements.txt
└─ README.md
```

## Environment
- Python 3.8+; Pandas, NumPy, scikit-learn (0.24+), imbalanced-learn (SMOTE), TensorFlow/Keras (2.x), Matplotlib.

## Notes & Limitations
- Trained and evaluated on **IoTID20** only; recommend time-disjoint splits and cross-dataset validation (e.g., UNSW-NB15, TON-IoT).
- Add runtime profiling (latency, memory, CPU) on an edge-class CPU.
- Consider model compression (pruning/quantization) and SHAP-based explainability.

## License
MIT (or your preferred license).
