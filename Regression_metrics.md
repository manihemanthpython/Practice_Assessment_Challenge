Tasks
Task 1 — Train and Compute Metrics Split 80/20 with random_state=42, scale with StandardScaler on training data only, train LinearRegression, and print MAE, MSE, RMSE, and R² on the test set.

Task 2 — Residual Analysis Compute residuals as actual − predicted and print the minimum, maximum, and mean. In a comment, explain what a mean residual near zero does and does not confirm.

Task 3 — Adjusted R² Using n = number of test samples and k = number of features, compute and print Adjusted R² as 1 − ((1 − R²) × (n − 1) / (n − k − 1)). In a comment, state in one line why it is more reliable than R².

Task 4 — Summary Print all four metrics together and write one comment per metric explaining what it tells you about model quality.

#  Dataset (paste directly into your script)

import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

data = {
    'area_sqft':   np.random.randint(500, 4000, n),
    'bedrooms':    np.random.randint(1, 6, n),
    'age_years':   np.random.randint(1, 40, n),
    'distance_km': np.round(np.random.uniform(1, 30, n), 1),
}
df = pd.DataFrame(data)
df['price_lakhs'] = (
    0.010 * df['area_sqft']
    + 4.0  * df['bedrooms']
    - 0.6  * df['age_years']
    - 1.0  * df['distance_km']
    + np.random.randn(n) * 4
)