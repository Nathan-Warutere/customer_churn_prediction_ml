"""Generates a synthetic telecom-style customer churn dataset with realistic signal."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(123)
n = 2000

tenure_months = rng.integers(0, 73, n)
monthly_charges = np.round(rng.uniform(18, 120, n), 2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
tech_support = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
online_security = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
num_support_calls = rng.poisson(1.5, n)
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n,
    p=[0.35, 0.2, 0.225, 0.225],
)
senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) * rng.uniform(0.9, 1.0, n), 2)

# Build churn probability from a logistic combination of realistic risk factors
logit = (
    -1.3
    + 1.6 * (contract == "Month-to-month")
    - 1.1 * (contract == "Two year")
    + 0.9 * (internet_service == "Fiber optic")
    - 0.7 * (tech_support == "Yes")
    - 0.6 * (online_security == "Yes")
    + 0.22 * num_support_calls
    + 0.5 * (payment_method == "Electronic check")
    - 0.03 * tenure_months
    + 0.012 * monthly_charges
    + 0.35 * senior_citizen
    + rng.normal(0, 0.6, n)
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < prob_churn).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST-{10000+i}" for i in range(n)],
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract": contract,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "online_security": online_security,
    "num_support_calls": num_support_calls,
    "payment_method": payment_method,
    "senior_citizen": senior_citizen,
    "churn": churn,
})

df.to_csv("/home/claude/kaggle-portfolio/05-machine-learning-customer-churn/data/customer_churn.csv", index=False)
print(df.shape)
print("Churn rate:", round(df["churn"].mean(), 3))
print(df.head())
