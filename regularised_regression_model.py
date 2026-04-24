from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
import pandas as pd
import numpy as np

np.random.seed(0)
n = 200

data = {
    "age_years": np.random.randint(1, 15, n),
    "km_driven": np.random.randint(5000, 150000, n),
    "engine_cc": np.random.randint(800, 3500, n),
    "fuel_type": np.random.randint(0, 2, n),
    "noise_1": np.random.randn(n),
    "noise_2": np.random.randn(n),
}
df = pd.DataFrame(data)
df["price_lakhs"] = (
    -0.5 * df["age_years"]
    - 0.00003 * df["km_driven"]
    + 0.003 * df["engine_cc"]
    + 2.0 * df["fuel_type"]
    + np.random.randn(n) * (2)
)

# Checking Null values in Dataset.
N_valu = df.isnull().sum() * 100 / len(df)

print(N_valu)

"==============================================================================================================================================="
# Task 1 —
# Baseline and Diagnosis Split 80/20 with random_state=42,
# scale with StandardScaler on training only, and train LinearRegression.
# Print train R² and test R². In a comment, state whether the gap indicates overfitting, underfitting, or a good fit.
print("\nTest 1: Baseline and Diagnosis")
# Splitting The Data.
x = df.drop(columns=["price_lakhs"])
y = df["price_lakhs"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.80, random_state=42
)

# Standardscaler

num_col = ["age_years", "km_driven", "engine_cc", "fuel_type", "noise_1", "noise_2"]
scaler = StandardScaler()

# Store column names before transformation
x_train_columns = x_train.columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# LinearRegression
Linear_model = LinearRegression()

Linear_model.fit(x_train, y_train)
y_train_pred = Linear_model.predict(x_train)
y_test_pred = Linear_model.predict(x_test)

MAE = mean_absolute_error(y_train, y_train_pred)
Train_R2 = r2_score(y_train, y_train_pred)
Test_R2 = r2_score(y_test, y_test_pred)

gap = Train_R2 - Test_R2

print(f"Training R2 score : {Train_R2.__round__(4)}")
print(f"Testing R2 score  : {Test_R2.__round__(4)}")
print(f"The Gap Between Tr vs Te : {gap.__round__(4)}\n")

# The gap indicates
# If the train r2 >> test r2, it's Overfitting. like (0.92 and 0.68)
# If Both are Low, it's a underfitting.         like (0.50 and 0.40)
# If they are close and high, it's a Good Fit.  like (0.92 and 0.90)


# Task 2 —
# Ridge Tuning Loop through alphas [0.01, 1, 10, 100, 500],
# train a Ridge model for each, and print train R² and test R².
# Print the best alpha. In a comment,
# explain what happens to bias and variance as alpha increases.
print("Task 2: Ridge Tuning")
print("Finding Best alpha")
alphas = [0.01, 1, 10, 100, 500]

result = []
for i in alphas:
    Ridge_model = Ridge(alpha=i)
    Ridge_model.fit(x_train, y_train)
    y_train_pr = Ridge_model.predict(x_train)
    y_test_pr = Ridge_model.predict(x_test)

    tr_r2 = r2_score(y_train, y_train_pr)
    te_r2 = r2_score(y_test, y_test_pr)

    result.append((tr_r2, te_r2))

    print("-" * 50)
    print(f"Best alpha : {i}")
    print(f"{"Ridge Model":<15} {"Train":>15} {"Test":>15}")
    print(f"{"R2 score Error":<15} {tr_r2:>15.4f} {te_r2:>15.4f}")
tr_r2, te_r2 = zip(*result)
best_idx = np.argmax(te_r2)
best_alpha = alphas[best_idx]
print("-" * 50)
print()
print(f"Best alpha   : {best_alpha}")
# As alpha increases, Bias increases and Variance decreases.
# Higher alpha penalizes complexity, making the model "stiffer" (less likely to overfit
# noise, but potentially missing the true underlying signal if alpha is too high).


# Task 3 —
# Lasso Feature Selection Train a Lasso model with alpha=1.0 and max_iter=10000.
# Print each feature's coefficient.
# In a comment, identify which features were zeroed out and what that means.

ols_model = Lasso(alpha=1.0, max_iter=10000)
ols_model.fit(x_train, y_train)

y_train_pre = ols_model.predict(x_train)
y_test_pre = ols_model.predict(x_test)

ols_tr_r2 = r2_score(y_train, y_train_pre)
ols_te_r2 = r2_score(y_test, y_test_pre)

coef_comp = pd.DataFrame(
    {"Feature": x_train_columns, "OLS": ols_model.coef_, "Ridge": Ridge_model.coef_}
)
print("\nTask 3: Lasso Feature Selection")
print(coef_comp)
# Features where the Coefficient is 0.0 were "zeroed out" by the model.
# Meaning: This indicates that these specific features were deemed redundant or
# non-informative for predicting the target variable. Lasso effectively performed
# automatic feature selection, simplifying the model and reducing overfitting
# by excluding these variables entirely.


# Task 4 —
# Validation Stability Re-run your best Ridge model using three different random seeds [42, 7, 123] for the train-test split.
# Print test R² for each seed.
# In a comment, state whether the small or large variation confirms model stability.

print("\nTask 4: Validation Stability")
print("=" * 50)

seeds = [42, 7, 123]
test_r2_scores = []

for seed in seeds:
    # Split data with different random seed
    x_train_seed, x_test_seed, y_train_seed, y_test_seed = train_test_split(
        x, y, train_size=0.80, random_state=seed
    )

    # Scale the data
    scaler_seed = StandardScaler()
    x_train_seed = scaler_seed.fit_transform(x_train_seed)
    x_test_seed = scaler_seed.transform(x_test_seed)

    # Train Ridge model with best alpha
    ridge_stable = Ridge(alpha=best_alpha)
    ridge_stable.fit(x_train_seed, y_train_seed)

    # Predict and calculate test R²
    y_test_pred_seed = ridge_stable.predict(x_test_seed)
    test_r2_seed = r2_score(y_test_seed, y_test_pred_seed)
    test_r2_scores.append(test_r2_seed)

    print(f"Random Seed {seed:<3} - Test R2: {test_r2_seed:.4f}")

# Calculate variation
mean_r2 = np.mean(test_r2_scores)
std_r2 = np.std(test_r2_scores)
print("-" * 50)
print(f"Mean Test R2: {mean_r2:.4f}")
print(f"Std Dev of Test R2: {std_r2:.4f}")

# Model Stability Assessment:
# If the standard deviation is small (< 0.05), it indicates HIGH stability - the model
# produces consistent results across different train-test splits, suggesting the model
# generalizes well and is robust.
# If the standard deviation is large (> 0.05), it indicates LOW stability - the model's
# performance varies significantly depending on the data split, suggesting potential
# overfitting or high sensitivity to specific data samples.
