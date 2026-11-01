from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dry_bean_dataset.csv"
TARGET = "Class"

df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Display the dataset dimensions.
print("Dataset shape:", df.shape)

# Display the first five rows in the terminal.
print("\nFirst five rows:")
print(df.head())

# Display the input feature names.
print("\nInput features:")
print(X.columns.tolist())

# Display the number of records in each class.
print("\nClass distribution:")
print(y.value_counts())

# Check for missing values in each column.
print("\nMissing values per column:")
print(df.isnull().sum())

# Save the dataset as a CSV file.
df.to_csv(DATASET_PATH, index=False)

print("\nDataset saved as dry_bean_dataset.csv")