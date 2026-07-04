# train_and_save_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load your cleaned dataset
df = pd.read_csv("Disease and symptoms dataset.csv")

# Step 2: Prepare X and y
X = df.drop(columns=["diseases"])
y = df["diseases"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the trained model
joblib.dump(model, "smart_doctor_model.pkl")

print("✅ Model trained and saved as smart_doctor_model.pkl")
