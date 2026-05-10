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


#================= STEP-4 EXPLORE THE DATA ==================

print("\n" + "="*55)
print("EXPLORATION RESULTS")
print("="*55)

total = df["Number of offences"].sum()
print(f"\nTotal number of offences recorded: {total:,}")

#----------- Group by Date (monthly totals) ----------------
monthly_totals = df.groupby("Date")["Number of offences"].sum().reset_index()
print(type(monthly_totals))
print(f"\nMonthly totals table ({len(monthly_totals)} months):")
print(monthly_totals.head(5))

#------ MEAN AVERAGE ------------
# WHAT DOES THIS MEAN AS AN INSIGHT?
# If mean = 2,900 → on average, NT records ~2,900 crimes per month

mean_val = monthly_totals["Number of offences"].mean()
print(f"\nMean average offenece per month recorded: {mean_val:.0f}")
print(f"\nEvery month on average, NT records about {mean_val:.0f}")

#-------------------- MEDIAN CALCULATION ---------------------
#.median() finds the MIDDLE value when all months sorted low to high
# Example with 5 months: 1800, 2600, 2900, 3100, 5200
# Sorted: 1800  2600  [2900]  3100  5200 median = 2900

median_val = monthly_totals["Number of offences"].median()

# WHY MEDIAN MATTERS:
# Mean is pulled up by extreme months (e.g. a riot month with 6000 crimes)
# Median ignores those extremes and shows the "typical" month
# If mean (3200) >> median (2700) → a few months had extreme crime spikes
# If mean ≈ median → crime is fairly stable across all months

print(f"\nMedian offeneces per months: {median_val:.0f}")

difference = mean_val - median_val 
if difference > 200:
    print(f"\nMean is {difference:.0f} higher than median.")
    print(f"\nSome months had unusually HIGH crime recorded.")
else:
    print(f"\nMean and median are close which mean crime is fairly consistent")


#-------------------- MAX AND MIN --------------------------
#.max() and .min() finds the highest and lowest value in the columns

max_offeneces = monthly_totals["Number of offences"].max()
min_offeneces = monthly_totals["Number of offences"].min()

#.idxmax() and .idxmin() finds the highest and lowest index in the rows

max_idx = monthly_totals["Number of offences"].idxmax()
min_idx = monthly_totals["Number of offences"].idxmin()

# Get the max and min value as DATE from monthly_totals location
max_date = monthly_totals.loc[max_idx,"Date"]
min_date = monthly_totals.loc[min_idx, "Date"]

#INSIGHT:
# Max month -> when was crime at its absolute worst? why?
# Min month -> when was crime lowest? What was happening then ?

print(f"\nMaximun number of offeneces is recorded as: {max_offeneces} in the month of: {max_date.strftime("%B %Y")}")
print(f"\nMinimun number of the offences is recorded as: {min_offeneces} in the month of: {min_date.strftime("%B %Y")}")

#------------------- BY REGION -----------------------------

region_totals = df.groupby("Reporting Region")["Number of offences"].sum().sort_values(ascending=False).reset_index()
#print(f"\nThe number of offences according to the region are: \n{region_totals}")

region_totals["Percentage"] = (region_totals["Number of offences"] / total * 100).round(1)
print(f"\nThe number of offences according to the REGION are: \n{region_totals}")


#------------------BY CRIME TYPE-------------------------
crime_totals = df.groupby("Crime Type")["Number of offences"].sum().sort_values(ascending=False).reset_index()
crime_totals["Precentage"] = (crime_totals["Number of offences"] / total * 100).round(1)
print(f"\nThe number of offences according to the CRIME TYPE are: \n{region_totals}")

#--------------- DV AND ALCOHOL------------------------------
dv_df = df[df["DV involvement"] == "Yes"]
alc_df = df[df["Alcohol involvement"] == 'Yes']

dv_count = dv_df["Number of offences"].sum()
alc_count = alc_df["Number of offences"].sum()

print(f"\nDV offences recorded as: {dv_count:,} and the DV_percentage is: {dv_count/total * 100:.1f}% of all crime.")
print(f"\nAlcohol offences recorded as: {alc_count:,} and the Alc_percantage is: {alc_count/total *100:.1f}% of all crime.")

#----------------------- Year over year -------------------
yearly = df[df["Year"].isin([2024,2025])].groupby("Year")["Number of offences"].sum()
yearly_crime_rate = ((yearly[2025] - yearly[2024])/ yearly[2024]) * 100

print(f"\nTotal 2024 offences recorded as : {yearly[2024]}")
print(f"\nTotal 2025 offences recorded as : {yearly[2025]}")
print(f"\nYearly crime rate = {yearly_crime_rate:.1f}%")

if yearly_crime_rate < 0:
    print("Good: Crime rate has decreased from 2024 to 2025")
else:
    print("Concerning: Crime increased from 2024 to 2025")


#========================== STEP 5 - FUTURE PREDICTION =======================
#GOAL : Use past monthly crime data to predict future months

print("\n" + "="*55)
print("PREDICTION")
print("="*55)

#------------------- Prepare training data ------------------------
train_df = df[df["Year"].isin([2024, 2025])].groupby("Date")["Number of offences"].sum().reset_index().sort_values("Date")
print(f"Trainig data: {len(train_df)} months")
print(train_df)

train_df["t"] = np.arange(len(train_df))
print("\nTime column (t) added: ")
print(train_df[["Date", "t", "Number of offences"]].head(5))

#------------------ INPUT AND OUTPUT -----------------------------
# X = what we feed INTO the model (the time steps). 2D practice [[]]
# y = what we want the model to LEARN to predict (offence counts) 1D pratice []
# The model learns: "when t=0, offences=3291, when t=1, offences=2683..."
# Then we ask: "when t=24 (March 2026), what do you predict?"

X = train_df[["t"]].values
y = train_df["Number of offences"].values

print(f"\nX shape: {X.shape} (24 motths, 1 feature)")
print(f"\n shape: {y.shape} (24 months crime counts)")

# ======= Build and train the model =============================

# make_pipeline chains two steps into one model:
# Step 1: PolynomialFeatures(degree=2)
#         Takes t and creates [t, t²]
#         This allows the model to fit curves not just straight lines
#         degree=2 means we go up to t-squared (parabola shape)
# Step 2: LinearRegression
#         Finds the best curve through all the data points
#simple say okay haha 

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
print("\nModel trained!")

r2 = model.score(X, y)
print(f"R2 score: {r2:.2f}")
print(f" Model explains {r2*100:.0f}% of past monthly crime variation")

if r2 >= 0.7:
    print("STRONG model -> prediction are reliable")
elif r2 >= 0.4: 
    print("MODERATE model -> prediction are resonable estimates")
else:
    print("WEAK model -> treat prediction as rough estimates")


