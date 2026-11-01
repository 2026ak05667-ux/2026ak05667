# Import the required libraries.
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download the Dry Bean dataset from the UCI repository.
dry_bean = fetch_ucirepo(id=602)

# Separate input features (X) and target labels (y).
X = dry_bean.data.features
y = dry_bean.data.targets

# Combine features and target into one DataFrame.
df = X.copy()
df["Class"] = y.iloc[:, 0]

# Display the dataset dimensions.
print("Dataset shape:", df.shape)

# Display the first five rows.
print("\nFirst five rows:")
print(df.head())
# Display feature names.
print("\nInput features:")
print(X.columns.tolist())

# Count the examples in each class.
print("\nClass distribution:")
print(y.iloc[:, 0].value_counts())

# Check missing values.
print("\nMissing values per column:")
print(df.isnull().sum())

# Save the dataset for future experiments.
df.to_csv("dry_bean_dataset.csv", index=False)

print("\nDataset saved as dry_bean_dataset.csv")