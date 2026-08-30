"""
data_processing.py

Handles cleaning, feature engineering, and encoding of raw Telco Customer
Churn data so that it matches the feature format the trained model expects.

IMPORTANT: engineer_features() must mirror exactly what was done in the
training notebook (avg_monthly_spend, tenure_group, service_count). If this
drifts from the notebook, predictions will be wrong even though no error
is thrown - reindex() silently zero-fills any column it doesn't recognize.
"""

import pandas as pd

SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning steps for the raw Telco Customer Churn dataset."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreates the exact engineered features used in training:
    avg_monthly_spend, tenure_group, service_count.
    """
    df = df.copy()

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    )

    present_service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
    df["service_count"] = (df[present_service_cols] == "Yes").sum(axis=1)

    return df


def encode_for_model(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """
    One-hot encodes a cleaned + feature-engineered dataframe and aligns it
    to the exact column structure the model was trained on.
    """
    encoded = pd.get_dummies(df)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded


def compute_revenue_at_risk(monthly_charges, churn_probability):
    """Revenue at Risk = Monthly Charges x Probability of Churn."""
    return monthly_charges * churn_probability


def score_dataframe(raw_df: pd.DataFrame, model, model_columns: list) -> pd.DataFrame:
    """
    Full pipeline: raw customer rows -> cleaned -> engineered -> encoded ->
    scored with churn probability, revenue at risk, and risk level.
    Returns the ORIGINAL raw_df with new columns appended (for readability
    in the dashboard), not the one-hot encoded version.
    """
    cleaned = clean_raw_dataframe(raw_df)
    engineered = engineer_features(cleaned)
    encoded = encode_for_model(engineered, model_columns)

    probs = model.predict_proba(encoded)[:, 1]

    result = raw_df.copy()
    result["Churn_Probability"] = probs
    result["RevenueAtRisk"] = result["MonthlyCharges"] * probs
    result["RiskLevel"] = pd.cut(
        probs, bins=[-0.01, 0.35, 0.65, 1.0], labels=["Low", "Medium", "High"]
    )
    return result
