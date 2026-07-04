# ----------------------------------------------
# SMART DOCTOR PROJECT - PHASE 1
# STEP 3: DATA PREPROCESSING & CLEANING
# ----------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
print("📂 Loading dataset...")
df = pd.read_csv("Disease and symptoms dataset.csv")

print("\n✅ Dataset loaded successfully!")
print("🔹 Shape of dataset:", df.shape)
print("🔹 Columns:", list(df.columns))
print("\nSample records:")
print(df.head())

# Step 2: Check for missing values
print("\n🔍 Checking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# Step 3: Handle missing values (replace NaN with empty string)
df.fillna("", inplace=True)

# Step 4: Remove duplicate records
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\n🧹 Removed {before - after} duplicate rows.")

# Step 5: Normalize column names (optional)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Step 6: Encode categorical data
print("\n🎯 Encoding disease labels...")
if "diseases" in df.columns:
    le = LabelEncoder()
    df["disease_encoded"] = le.fit_transform(df["diseases"])
else:
    print("⚠️ 'disease' column not found! Please check your dataset column names.")
    exit()

# Step 7: Display cleaned dataset info
print("\n✅ Data after cleaning:")
print(df.head())

print("\n📊 Dataset Summary:")
print(df.describe(include='all'))

# Step 8: Save the cleaned dataset
df.to_csv("cleaned_disease_dataset.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_disease_dataset.csv'.")

# Step 9: Verify unique diseases
print("\n🩺 Unique Diseases in Dataset:", df['diseases'].nunique())

print("\n✅ Phase 1 Data Preprocessing Completed Successfully!")

