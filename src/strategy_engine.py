"""
strategy_engine.py

A transparent, rule-based retention prescription engine.

This intentionally does NOT use a black-box model to decide the retention
action - the whole point is that a business user can read these rules and
understand exactly why a customer got a given recommendation. This is what
turns a churn prediction into a business decision.

Tiers:
  Tier 1 - Month-to-month contract risk -> push toward annual lock-in
  Tier 2 - Missing security/tech support, or a generic risk fallback
  Tier 3 - Electronic check payment risk -> push toward auto-pay
  Tier 4 - Low risk, no action needed
"""

import pandas as pd

RISK_THRESHOLD = 0.35


def _get(record: dict, key: str, default=""):
    """Case-insensitive lookup on a customer record dict."""
    for k, v in record.items():
        if k.lower() == key.lower():
            return v
    return default


def recommend_action(record: dict, churn_probability: float) -> dict:
    """
    Given a single customer's attributes (as a dict) and their predicted
    churn probability, return a dict with:
      - tier (1-4)
      - risk_level ("Low", "Medium", "High")
      - action (human-readable recommendation)
    """
    contract = str(_get(record, "contract")).strip().lower()
    security = str(_get(record, "onlinesecurity")).strip().lower()
    tech_support = str(_get(record, "techsupport")).strip().lower()
    payment = str(_get(record, "paymentmethod")).strip().lower()

    if churn_probability < RISK_THRESHOLD:
        return {
            "tier": 4,
            "risk_level": "Low",
            "action": "No action needed - customer profile is healthy and retained.",
        }

    risk_level = "High" if churn_probability >= 0.65 else "Medium"

    if contract == "month-to-month":
        return {
            "tier": 1,
            "risk_level": risk_level,
            "action": "Offer 1-Year Contract Upgrade with 15% monthly bill discount.",
        }

    if security in ("no",) or tech_support in ("no",):
        return {
            "tier": 2,
            "risk_level": risk_level,
            "action": "Offer free Online Security & Tech Support bundle for 6 months.",
        }

    if payment == "electronic check":
        return {
            "tier": 3,
            "risk_level": risk_level,
            "action": "Offer $5/month credit for switching to automatic payment (card/bank transfer).",
        }

    return {
        "tier": 2,
        "risk_level": risk_level,
        "action": "Flag for priority customer care review / account manager outreach.",
    }


def recommend_batch(df, probability_col: str = "Churn_Probability"):
    """
    Applies recommend_action() across a dataframe of customers.
    Returns the same dataframe with tier, risk_level, and action columns added.
    """
    results = df.apply(
        lambda row: recommend_action(row.to_dict(), row[probability_col]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df.reset_index(drop=True), results], axis=1)

