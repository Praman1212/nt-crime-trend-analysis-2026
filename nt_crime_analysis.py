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


#======================== STEP-3 CLEAN THE DATA =======================
print(f"Rows Before removing Unknown region: {len(df):,}")

# df[condition] filters the DataFrame to only keep rows where condition is True
# df["Reporting Region"] != "Unknown" means "keep rows where region is NOT Unknown"
df = df[df["Reporting Region"] != "Unknown"]
print(f"Rows After removing unknown region: {len(df):,}")


#------------------------------Rename crime categories----------------------------

# A dictionary maps old name → new name like key value pair
cat_map = {
    "01 Homicide"                          : "Homicide",
    "02 Assault"                           : "Assault",
    "03 Sexual offences"                   : "Sexual Offences",
    "04 Harm or endanger persons"          : "Harm/Endanger",
    "05 Robbery, blackmail, and extortion" : "Robbery/Extortion",
    "061 Burglary - dwelling"              : "Dwelling Burglary",
    "062 Burglary - non-residential"       : "Commercial Burglary",
    "07 Theft"                             : "Theft",
    "11 Property damage offences"          : "Property Damage",
}


# .map(cat_map) looks up each value in the dictionary and replaces it
# We store the result in a NEW column called "Crime Type"

df["Crime Type"] = df["Offence category"].map(cat_map)
print("\nCrime Type values after renaming:")
print(df["Crime Type"].unique())

#---------------------- Create a proper Date column ------------------

# 1. df["Year"].astype(str)              → converts 2024 to "2024"
# 2. df["Month number"].astype(str)      → converts 3 to "3"
# 3. .str.zfill(2)                       → pads to 2 digits: "3" → "03"
# 4. We join them: "2024" + "-" + "03" + "-01" = "2024-03-01"
# 5. pd.to_datetime() converts that string into a real Python date

df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" +
    df["Month number"].astype(str).str.zfill(2) + "-01"
)

print("\nDate column created")
print(df["Date"].head(5))

#-------------- In DV and Alcohol Inv. columns there is - words-------------------------
# Original data uses "-" for rows where DV/alcohol is not recorded
# "-" is confusing — replace it with the word "Unknown"
# This makes Power BI slicers show "Unknown" instead of "-"

df["DV involvement"] = df["DV involvement"].replace("-","Unknown")
df["Alcohol involvement"] = df["Alcohol involvement"].replace("-","Unknown")

print("\nDV involvement values:")
print(df["DV involvement"].unique())

print("\nAlcohol involvement values:")
print(df["Alcohol involvement"].unique())

print("\nData cleaning complete!")