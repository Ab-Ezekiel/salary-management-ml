# model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("payroll_dataset_fixed.csv")

# ================================
# PREPROCESSING FUNCTION
# ================================
def preprocess_data(df):

    # Target
    y = df["anomaly_label"]

    # Features
    X = df.drop(columns=["employee_id", "anomaly_label"])

    # Remove leaky features
    leaky_features = [
        "expected_salary",
        "salary_deviation",
        "salary_deviation_ratio",
        "bonus_ratio",
        "deduction_ratio"
    ]

    X = X.drop(columns=leaky_features)

    # Encode categorical variables
    X = pd.get_dummies(
        X,
        columns=["gender", "department", "employment_type"],
        drop_first=True
    )

    return X, y


# ================================
# TRAIN MODEL
# ================================
X, y = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save column structure
model_columns = X.columns


# ================================
# PREDICTION FUNCTION
# ================================
def predict(input_df):

    # Copy input
    data = input_df.copy()

    # Drop unnecessary columns if present
    drop_cols = ["employee_id", "anomaly_label"]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # Remove leaky features if present
    leaky_features = [
        "expected_salary",
        "salary_deviation",
        "salary_deviation_ratio",
        "bonus_ratio",
        "deduction_ratio"
    ]

    data = data.drop(columns=[col for col in leaky_features if col in data.columns])

    # Encode categorical variables
    data = pd.get_dummies(
        data,
        columns=["gender", "department", "employment_type"],
        drop_first=True
    )

    # Align columns with training data
    data = data.reindex(columns=model_columns, fill_value=0)

    # Predict probabilities
    probs = model.predict_proba(data)[:, 1]

    # Apply threshold (IMPORTANT)
    preds = (probs > 0.2).astype(int)

    return preds, probs