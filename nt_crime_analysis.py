#=================== STEP-1 Import Libraries ===========================
# pandas is a library that lets you work with tables of data
# numpy is a library for doing maths on lists of numbers
# LinearRegression → draws the best straight line through data points
# PolynomialFeatures → allows the line to CURVE, not just be straight
# make_pipeline → connects PolynomialFeatures + LinearRegression


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")
print("All libraries imported successfully")




#===================== STEP-2 Load Data =========================
df = pd.read_csv("nt_crime_dataset_feb_2026.csv")
print("Column names in the files:")
print(df.columns.tolist())
df.columns = df.columns.str.strip()
print("\nColumn names after stripping spaces:")
print(df.columns.tolist())

# print(f"\nTable size: {df.shape[0]:,} rows and {df.shape[1]} columns")
print("\nTable size: \n rows = " + format(df.shape[0]) + " columns = " +format(df.shape[1]))

# .head(5) shows you the first 5 rows of the table
print("\nFirst 5 rows of data:")
print(df.head(5))

# dtypes shows the DATA TYPE of each column
print("\nColumn data types:")
print(df.dtypes)