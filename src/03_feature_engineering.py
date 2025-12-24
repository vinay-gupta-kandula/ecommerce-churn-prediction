import pandas as pd
import json
from pathlib import Path


# =========================
# Paths
# =========================
CLEAN_DATA_PATH = Path("data/processed/cleaned_transactions.csv")
FEATURE_DATA_PATH = Path("data/processed/customer_features.csv")
FEATURE_INFO_PATH = Path("data/processed/feature_info.json")


def main():
    print("Starting Phase 4: Feature Engineering...")

    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["invoicedate"])

    # -------------------------
    # Reference dates
    # -------------------------
    snapshot_date = df["invoicedate"].max()
    churn_window_days = 90

    # -------------------------
    # Aggregate to customer level
    # -------------------------
    customer_df = df.groupby("customerid").agg(
        first_purchase=("invoicedate", "min"),
        last_purchase=("invoicedate", "max"),
        frequency=("invoice", "nunique"),
        total_transactions=("invoice", "count"),
        monetary_value=("price", "sum"),
        avg_order_value=("price", "mean"),
        total_quantity=("quantity", "sum"),
        avg_quantity_per_txn=("quantity", "mean"),
        unique_products=("stockcode", "nunique"),
        unique_invoices=("invoice", "nunique"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        avg_price=("price", "mean"),
        price_std=("price", "std"),
        country_count=("country", "nunique")
    ).reset_index()

    # -------------------------
    # Time-based features
    # -------------------------
    customer_df["recency_days"] = (snapshot_date - customer_df["last_purchase"]).dt.days
    customer_df["customer_tenure_days"] = (
        customer_df["last_purchase"] - customer_df["first_purchase"]
    ).dt.days
    customer_df["days_since_first_purchase"] = (
        snapshot_date - customer_df["first_purchase"]
    ).dt.days
    customer_df["days_since_last_purchase"] = (
        snapshot_date - customer_df["last_purchase"]
    ).dt.days

    # -------------------------
    # Average days between purchases
    # -------------------------
    customer_df["avg_days_between_purchases"] = (
        customer_df["customer_tenure_days"] / customer_df["frequency"].clip(lower=1)
    )

    # -------------------------
    # Churn label
    # -------------------------
    customer_df["churn"] = (
        customer_df["recency_days"] > churn_window_days
    ).astype(int)

    # -------------------------
    # Handle missing values
    # -------------------------
    customer_df.fillna(0, inplace=True)

    # -------------------------
    # Save feature data
    # -------------------------
    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer_df.to_csv(FEATURE_DATA_PATH, index=False)

    # -------------------------
    # Feature metadata
    # -------------------------
    churn_rate = round(customer_df["churn"].mean() * 100, 2)

    feature_info = {
        "total_customers": int(len(customer_df)),
        "total_features": int(customer_df.shape[1] - 2),  # excluding customerid & churn
        "churn_rate_percent": churn_rate,
        "churn_definition": f"No purchase in last {churn_window_days} days",
        "feature_categories": {
            "rfm": [
                "recency_days",
                "frequency",
                "monetary_value",
                "avg_order_value"
            ],
            "transaction_behavior": [
                "total_transactions",
                "total_quantity",
                "avg_quantity_per_txn",
                "unique_products",
                "unique_invoices"
            ],
            "time_based": [
                "customer_tenure_days",
                "days_since_first_purchase",
                "days_since_last_purchase",
                "avg_days_between_purchases"
            ],
            "price_metrics": [
                "min_price",
                "max_price",
                "avg_price",
                "price_std"
            ],
            "geography": [
                "country_count"
            ]
        }
    }

    with open(FEATURE_INFO_PATH, "w") as f:
        json.dump(feature_info, f, indent=4)

    print("Phase 4 completed successfully ✅")
    print(f"Total customers: {len(customer_df)}")
    print(f"Churn rate: {churn_rate}%")


if __name__ == "__main__":
    main()
