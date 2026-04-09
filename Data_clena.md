# Practice_Assessment_Challenge

Problem Statement
You have been given a raw customer dataset that contains missing values, duplicate records, categorical text columns, and numerical features on vastly different scales. Before any model can be trained, this data needs to be cleaned and prepared.

Task 1 — Clean the Data
Using the synthetic dataset below, write Python code to:

Impute missing values in age with the median and in city with the mode
Remove any duplicate rows
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 2, 5],
    'age':         [25, np.nan, 35, np.nan, np.nan, 40],
    'city':        ['Mumbai', 'Delhi', np.nan, 'Mumbai', 'Delhi', np.nan],
    'gender':      ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'annual_income': [40000, 80000, 60000, 120000, 80000, 95000]
})
Task 2 — Encode Categorical Columns
After cleaning, encode the categorical columns:

Apply One-Hot Encoding to city
Apply Label Encoding to gender
Print the resulting dataframe.

Task 3 — Scale Numerical Features
Scale age and annual_income using both methods below on the cleaned dataframe, and print the results of each:

Min-Max Scaling (using MinMaxScaler)
Standardisation (using StandardScaler)
In 2–3 sentences, explain when you would prefer one over the other.

