import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

data = {
    "area_sqft": np.random.randint(500, 4000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "age_years": np.random.randint(1, 40, n),
    "distance_km": np.round(np.random.uniform(1, 30, n), 1),
}
df = pd.DataFrame(data)
df["price_lakhs"] = (
    0.010 * df["area_sqft"]
    + 4.0 * df["bedrooms"]
    - 0.6 * df["age_years"]
    - 1.0 * df["distance_km"]
    + np.random.randn(n) * 4
)
# Task 1 —
# Train and Compute Metrics Split 80/20 with random_state=42,
# scale with StandardScaler on training data only,
# train LinearRegression, and print MAE, MSE, RMSE, and R² on the test set.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Data Spliting 
x = df.drop(columns=["price_lakhs"])
y = df["price_lakhs"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42)

# Standardscaler
scaler = StandardScaler()

numeric_col = ['area_sqft', 'bedrooms', 'age_years', 'distance_km']
x_train[numeric_col] = scaler.fit_transform(x_train)
x_test[numeric_col]  = scaler.transform(x_test)

# ML Model bulding.
# Linear Regression

my_model = LinearRegression()
my_model.fit(x_train,y_train)
y_train_pred = my_model.predict(x_train)
y_test_pred  = my_model.predict(x_test)

# Errors Mean Absoluat Error
tr_mae = mean_absolute_error(y_train,y_train_pred)
te_mae = mean_absolute_error(y_test, y_test_pred)

#  Mean Square Error
tr_mse = mean_squared_error(y_train,y_train_pred)
te_mse = mean_squared_error(y_test,y_test_pred)

# Root mean squared Error
tr_rmse = root_mean_squared_error(y_train,y_train_pred)
te_rmse = root_mean_squared_error(y_test,y_test_pred)

# R2 Error
tr_r2 = r2_score(y_train,y_train_pred)
te_r2 = r2_score(y_test,y_test_pred)
print("Summary metrics")
print("="*50)
print("Task:-1")
print("="*50)
print(f"{"Linear Regression":<15} {'Train Error':>15}      {'Test Error':.15}")
print('='*50)
print(f"{'MAE':<15} {"₹{:,.0f}".format(tr_mae):>15} {"₹{:,.0f}".format(te_mae):>15}")
print(f"{'MSE':<15} {"₹{:,.0f}".format(tr_mse):>15} {"₹{:,.0f}".format(te_mse):>15}")
print(f"{'RMsE':<15} {"₹{:,.0f}".format(tr_rmse):>15} {"₹{:,.0f}".format(te_rmse):>15}")
print(f"{'R2':<15} {"₹{:,.0f}".format(tr_r2):>15} {"₹{:,.0f}".format(te_r2):>15}")
print('='*50)
gap = tr_r2 - te_r2
print(f"Train-Test Gap(R²): {gap:.2f}")
print()

# Task 2 — 
# Residual Analysis Compute residuals as actual − predicted and print the minimum, maximum, and mean. 
# In a comment, explain what a mean residual near zero does and does not confirm.
actual    = y_train
predicted = y_train_pred

residuals = (actual) - (predicted)
res_Min = np.min(residuals)
res_Max = np.max(residuals)
res_mean = np.mean(residuals)
print("\nTask:- 2")
print(f"Minimum Residual: {res_Min:.2f}")
print("Maxmum Residual : {res_Max:.2f}")
print("Average of Residual : {res_mean:.2f}")


# Task 3 — Adjusted R² Using n = number of test samples and k = number of features, 
# compute and print Adjusted R² as 1 − ((1 − R²) × (n − 1) / (n − k − 1)). 
# In a comment, state in one line why it is more reliable than R².

r2 = r2_score(y_test,y_test_pred)
n  = x_test.shape[0]
k  = x_test.shape[1]

adj_r2 = 1 - ((1 - r2)*(n - 1) / (n - k - 1))

print("\nTask:-3")
print(f"R2_squared : {r2:.4f}")
print(f"Adjusted R2_squared : {adj_r2:.4f}")