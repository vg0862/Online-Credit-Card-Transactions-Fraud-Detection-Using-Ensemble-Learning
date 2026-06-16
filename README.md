# Fraudulent E-Commerce Transaction Detection (Group 7)

## Project Overview

This repository contains a Jupyter notebook implementing a machine learning pipeline to detect fraudulent e-commerce transactions. The notebook defines data loading, preprocessing, feature engineering, model training (Logistic Regression, Random Forest, XGBoost, Feed-Forward Neural Network), resampling (SMOTE, ADASYN, SVMSMOTE), an ensemble meta-model, and evaluation utilities.

**Important:** The notebook expects CSV datasets; update the dataset file paths in the notebook before running.

**Notebook:** `fraud_detection.ipynb`

## Files

- **fraud_detection.ipynb**: Main notebook with all implementation (data classes, preprocessors, models, training pipeline, evaluation/plots).
- **README.md**: This file.
- **requirements.txt**: Python dependencies used by the notebook.

## Key Components

- `TxnDataSet` — loads dataset and filters to credit-card payments.
- `TxnDataPreProcessor` — fills missing values, extracts new features (hour bins, log transforms, z-scores, account age groups, device-location composite, flags for high amounts/quantities, etc.), encodes categorical features.
- `TxnDataSampler` — applies resampling methods to balance classes.
- `BaseModels` — trains classical models and evaluates them.
- `FNN` — feed-forward neural network built with Keras/TensorFlow.
- `EnsembleModel` — stacking meta-classifier that combines base models and FNN.
- `ModelsEvaluator` — plotting utilities for ROC, metrics and confusion matrices.
- `MLPipeline` — orchestrates loading, preprocessing, resampling, training and evaluation.

## Setup & Installation

1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
# Install dependencies
pip install -r requirements.txt
```

3. If you have a GPU and want TensorFlow GPU support, install the appropriate TensorFlow build for your CUDA/cuDNN versions.

## How to Run

1. Open `fraud_detection.ipynb` in Jupyter Lab/Notebook or VS Code.
2. Update the dataset paths passed to `MLPipeline` (currently the notebook uses Google Drive paths). Point them to your local CSV files.
3. Run the cells sequentially. The main entry to run training + evaluation is:

- Initialize pipeline and load data:

```python
pipeline = MLPipeline('<train_csv_path>', '<test_csv_path>')
pipeline.loadData()
```

- Run training and evaluation:

```python
pipeline.trainAndEvaluate()
```

Notes:
- Training may be slow for large datasets. Consider sampling or using a machine with GPU for the FNN and XGBoost.
- The code uses large model sizes, high tree counts, and long epochs; adjust hyperparameters for your environment.

## Output

- The notebook prints model metrics and shows:
  - ROC curves
  - Metric bar charts (Accuracy, F1, Precision, Recall, ROC AUC)
  - Confusion matrices for each model

## Troubleshooting

- Missing packages: re-run `pip install -r requirements.txt`.
- Label encoding errors on unseen categories: the preprocessing code attempts to map unknowns; ensure `label_encoders` are passed correctly when using the test dataset.
- If GPU is not detected, the notebook falls back to CPU.

## Next steps / Suggestions

- Save trained models and encoders to disk (pickle / joblib) to avoid re-training.
- Add command-line interface or a small script to run experiments reproducibly.
- Add automated tests for data preprocessing steps.
