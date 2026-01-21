import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# 🔥 Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Drop ID column
df = df.drop(columns=["loan_id"])

# Drop missing values
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])
df["self_employed"] = le.fit_transform(df["self_employed"])

# Features
X = df[
    [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value"
    ]
]

# Target
y = df["loan_status"].str.strip().str.lower().map({"approved": 1, "rejected": 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

print("Model trained successfully and saved!")
