import pandas as pd
"""
data_register.py
------------------
Registers the tourism dataset for the "Visit with Us" MLOps pipeline.

This script reads the dataset straight from the repository `data` folder,
validates that every expected column is present, and prints a short summary
so the pipeline (and any reviewer) can confirm the data is correct before the
downstream jobs run. Keeping the CSV inside the GitHub repo means no external
dataset store is required.
"""

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# Some exports carry an unnamed index column ("Unnamed: 0"); drop it.
unnamed = [c for c in df.columns if c.startswith("Unnamed")]
if unnamed:
    df = df.drop(columns=unnamed)

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Target distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())
