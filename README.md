# Cancer Death Rate Analysis

## Overview
This repository contains an end-to-end exploratory data analysis (EDA) and baseline modeling pipeline for county-level cancer data. The project investigates predictors of `TARGET_deathRate` and provides reproducible code, visualizations, and model interpretation using SHAP.

## Dataset
- File: `cancer_reg.csv` (3,047 rows × 34 columns)
- Primary target: `TARGET_deathRate`
- Notes: The dataset contains demographic, socioeconomic, and health-related features at the county level. One column (`PctSomeCol18_24`) has high missingness and may be removed or imputed during preprocessing.

## Project Overview
- Performed data loading and cleaning, missing-value profiling, and type conversion.
- Conducted feature analysis using descriptive statistics and Pearson correlation with the target to identify top predictors.
- Built a reproducible Jupyter Notebook (`G10_Code.ipynb`) that includes baseline regression experiments (Linear Regression, SVR, Random Forest) and cross-validation.
- Interpreted models using SHAP to produce global and local feature importance explanations.
- Saved deliverables: cleaned dataset, correlation matrix, and a `requirements.txt` for reproducibility.

## Repository structure (recommended)
```
README.md
G10_Code.ipynb            # main analysis & modeling notebook
cancer_reg.csv           # dataset (place in repo root or data/ folder)
requirements.txt
results/                  # saved figures, model artifacts, metrics
data/                     # optional directory for datasets
```

## Quick setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# venv\Scripts\Activate.ps1

pip install -r requirements.txt
```
Example minimal `requirements.txt` contents:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
shap
```

## How to run
1. Place `cancer_reg.csv` in the repository root (or `data/` and update the notebook path).
2. Open the notebook:
```bash
jupyter notebook G10_Code.ipynb
```
3. Run cells from top to bottom. The notebook contains EDA, preprocessing, baseline model training, and SHAP interpretation snippets.

## Example: Train baseline models (snippet)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('G10_dataset.csv', encoding='latin1')
# Drop high-missing column or impute as needed
if 'PctSomeCol18_24' in df.columns:
    df = df.drop(columns=['PctSomeCol18_24'])

df = df.dropna(subset=['TARGET_deathRate'])
X = df.select_dtypes(include=['number']).drop(columns=['TARGET_deathRate'])
y = df['TARGET_deathRate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name, 'R2:', r2_score(y_test, preds), 'RMSE:', mean_squared_error(y_test, preds, squared=False))
```

## Model interpretation (SHAP)
For tree-based models like RandomForest, compute SHAP values and generate a summary plot:
```python
import shap
explainer = shap.TreeExplainer(models['RandomForest'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## Key findings (from EDA)
- Dataset size: 3,047 rows × 34 features.
- Top correlated features with `TARGET_deathRate` include `PctBachDeg25_Over`, `incidenceRate`, `povertyPercent`, and `medIncome`.


