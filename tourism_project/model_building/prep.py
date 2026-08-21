# for data manipulation
import pandas as pd
# for splitting the data into train and test sets
from sklearn.model_selection import train_test_split

# Load the registered dataset from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# Drop the serialization index column ("Unnamed: 0") if present
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Drop the unique identifier column (no predictive value)
df.drop(columns=["CustomerID"], inplace=True)

# Fix inconsistent category labels found during the Data Check
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Treat missing values if any (median for numerical, mode for categorical)
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Define the target variable
target = "ProdTaken"

# Split into predictors (X) and target (y)
X = df.drop(columns=[target])
y = df[target]

# stratify=y keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Xtrain:", Xtrain.shape, " Xtest:", Xtest.shape)
