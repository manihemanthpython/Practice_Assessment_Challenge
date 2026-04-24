Problem Statement
You are building and validating regularised regression models to predict car prices, investigating how alpha tuning and train-validation splits affect generalisation.

Tasks
Task 1 — Baseline and Diagnosis Split 80/20 with random_state=42, scale with StandardScaler on training only, and train LinearRegression. Print train R² and test R². In a comment, state whether the gap indicates overfitting, underfitting, or a good fit.

Task 2 — Ridge Tuning Loop through alphas [0.01, 1, 10, 100, 500], train a Ridge model for each, and print train R² and test R². Print the best alpha. In a comment, explain what happens to bias and variance as alpha increases.

Task 3 — Lasso Feature Selection Train a Lasso model with alpha=1.0 and max_iter=10000. Print each feature's coefficient. In a comment, identify which features were zeroed out and what that means.

Task 4 — Validation Stability Re-run your best Ridge model using three different random seeds [42, 7, 123] for the train-test split. Print test R² for each seed. In a comment, state whether the small or large variation confirms model stability.

Dataset (paste directly into your script)
import pandas as pd
import numpy as np

np.random.seed(0)
n = 120

data = {
    'age_years':   np.random.randint(1, 15, n),
    'km_driven':   np.random.randint(5000, 150000, n),
    'engine_cc':   np.random.randint(800, 3500, n),
    'fuel_type':   np.random.randint(0, 2, n),
    'noise_1':     np.random.randn(n),
    'noise_2':     np.random.randn(n),
}
df = pd.DataFrame(data)
df['price_lakhs'] = (
    -0.5  * df['age_years']
    - 0.00003 * df['km_driven']
    + 0.003 * df['engine_cc']
    + 2.0  * df['fuel_type']
    + np.random.randn(n) * 2
)