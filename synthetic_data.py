import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# Step 1: Load your real dataset
df = pd.read_csv("students.csv")  # or use your DataFrame if already loaded

# Step 2: Select only the relevant columns
columns_to_use = ["score", "age", "x", "y", "z", "accepted_school_id"]
df = df[columns_to_use]  # optional: filter if needed

# Step 2: Create metadata from the DataFrame
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Step 3: Train the SDV model
model = CTGANSynthesizer(metadata)
model.fit(df)

# Step 4: Generate synthetic data
synthetic_df = model.sample(10000)

# Step 5: Save synthetic data to CSV
synthetic_df.to_csv("synthetic_students.csv", index=False)

print("✅ Synthetic data saved to 'synthetic_students.csv'")

# Load real data
real_df = pd.read_csv("students.csv")
real_df = real_df[["score", "age", "x", "y", "z", "accepted_school_id"]]

# Load or use generated synthetic data
synthetic_df = pd.read_csv("synthetic_students.csv")  # or use directly if it's still in memory

# Optionally add a label to distinguish them
real_df["source"] = "real"
synthetic_df["source"] = "synthetic"

# Combine the two datasets
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# Save to CSV
combined_df.to_csv("combined_students.csv", index=False)