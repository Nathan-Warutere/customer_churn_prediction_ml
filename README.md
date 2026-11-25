# Customer Churn Prediction — Machine Learning

A standalone capstone project applying Kaggle's **Intro/Intermediate Machine Learning**
concepts to a telecom-style customer churn dataset, end to end: EDA → leakage-free
preprocessing pipeline → baseline vs. improved model → evaluation → feature importance →
saved, reusable model artifact.

## The problem

Predict which customers are likely to churn so a retention team can target offers, using
features like tenure, contract type, internet service, support-call history, and payment
method.

## What this project does

`notebooks/customer_churn_ml.ipynb` covers:

1. Exploratory data analysis (churn rate by contract type, tenure)
2. A proper train/test split (stratified, done *before* fitting any preprocessing)
3. A `ColumnTransformer` + `Pipeline` for scaling + one-hot encoding (no manual leakage risk)
4. A Logistic Regression baseline, evaluated with 5-fold cross-validated ROC-AUC
5. A Random Forest model compared fairly against the same baseline
6. Full evaluation beyond accuracy: precision/recall/F1, confusion matrix, ROC curve
7. Feature importance — the actionable output a retention team would use
8. Saving the entire fitted pipeline (preprocessing + model) as a single `.joblib` file

## Repo structure

```
.
├── data/
│   ├── generate_data.py
│   └── customer_churn.csv
├── images/                     # EDA chart, confusion matrix + ROC, feature importance
├── models/
│   └── churn_model_pipeline.joblib   # full fitted sklearn Pipeline, ready to reuse
├── notebooks/
│   └── customer_churn_ml.ipynb
├── requirements.txt
└── README.md
```

## Run it yourself

```bash
pip install -r requirements.txt
python data/generate_data.py          # regenerate the dataset (optional, already included)
jupyter notebook notebooks/customer_churn_ml.ipynb
```

## Load the saved model elsewhere

```python
import joblib
pipeline = joblib.load("models/churn_model_pipeline.joblib")
pipeline.predict(new_customers_df)   # same preprocessing is applied automatically
```

## Why synthetic data?

`data/generate_data.py` generates churn labels from a logistic combination of realistic
risk factors (month-to-month contracts and fiber internet increase churn risk; tenure,
tech support, and online security reduce it), so the model's learned feature importances
can be checked against ground-truth logic — a useful sanity check that the pipeline is
learning real signal.
