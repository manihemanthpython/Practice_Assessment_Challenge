Problem Statement
You are given a messy used car dataset. Your goal is to explore the raw data, clean it systematically, and evaluate a simple baseline model using MAE before any machine learning model is applied.

Task 1 — Explore and Identify Issues Load the dataset and use df.info(), df.describe(), and df.shape to report at least three data quality problems you observe (e.g., wrong dtypes, nulls, impossible values).

Task 2 — Clean the Data Fix the issues identified: drop null target rows, impute missing input features, strip and lowercase the brand column, extract numeric values from the mileage column, and remove duplicates.

Task 3 — Compute Baseline MAE Build a baseline model that predicts the mean selling_price for every record. Calculate and print the MAE of this baseline on the full cleaned dataset.