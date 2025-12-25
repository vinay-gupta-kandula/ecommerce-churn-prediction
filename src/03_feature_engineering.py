import pandas as pd
import numpy as np
from datetime import timedelta
import json
import os


class FeatureEngineer:
    def __init__(self, data_path):
        print("Loading cleaned transactions...")

        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.lower()

        required_cols = {
            "invoice",
            "stockcode",
            "quantity",
            "invoicedate",
            "price",
            "customerid",
            "country"
        }

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df["invoicedate"] = pd.to_datetime(self.df["invoicedate"])
        self.df["total_price"] = self.df["quantity"] * self.df["price"]

        print(f"Loaded transactions: {len(self.df)}")

    def find_valid_temporal_split(self):
        print("\n=== PHASE 4: FEATURE ENGINEERING ===")

        self.observation_end = self.df["invoicedate"].max()

        candidate_windows = [30, 45, 60, 75, 90, 120]

        for days in candidate_windows:
            cutoff = self.observation_end - timedelta(days=days)

            train_df = self.df[self.df["invoicedate"] <= cutoff]
            obs_df = self.df[
                (self.df["invoicedate"] > cutoff) &
                (self.df["invoicedate"] <= self.observation_end)
            ]

            train_customers = set(train_df["customerid"].unique())
            obs_customers = set(obs_df["customerid"].unique())

            churn_rate = len(train_customers - obs_customers) / len(train_customers)

            print(f"Trying {days} days → churn rate: {round(churn_rate,3)}")

            if 0.20 <= churn_rate <= 0.40:
                self.training_cutoff = cutoff
                self.churn_window_days = days
                self.train_df = train_df
                self.obs_df = obs_df

                print(f"✅ Selected churn window: {days} days")
                print(f"Final churn rate: {round(churn_rate,3)}")
                return

        raise ValueError(
            "❌ No temporal window produced churn between 20–40%. "
            "Dataset is too imbalanced near the end."
        )

    def define_churn(self):
        train_customers = set(self.train_df["customerid"].unique())
        obs_customers = set(self.obs_df["customerid"].unique())

        churned = train_customers - obs_customers

        self.customer_labels = pd.DataFrame({
            "customerid": list(train_customers),
            "churn": [1 if c in churned else 0 for c in train_customers]
        })

    def create_features(self):
        df = self.train_df.copy()

        features = df.groupby("customerid").agg(
            frequency=("invoice", "nunique"),
            total_transactions=("invoice", "count"),
            monetary_value=("total_price", "sum"),
            avg_order_value=("total_price", "mean"),
            total_quantity=("quantity", "sum"),
            avg_quantity_per_txn=("quantity", "mean"),
            unique_products=("stockcode", "nunique"),
            unique_invoices=("invoice", "nunique"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            avg_price=("price", "mean"),
            price_std=("price", "std"),
            country_count=("country", "nunique"),
            first_purchase=("invoicedate", "min"),
            last_purchase=("invoicedate", "max"),
        ).reset_index()

        features["recency_days"] = (
            self.training_cutoff - features["last_purchase"]
        ).dt.days

        features["customer_tenure_days"] = (
            features["last_purchase"] - features["first_purchase"]
        ).dt.days

        features["days_since_first_purchase"] = (
            self.training_cutoff - features["first_purchase"]
        ).dt.days

        features["days_since_last_purchase"] = (
            self.training_cutoff - features["last_purchase"]
        ).dt.days

        features["avg_days_between_purchases"] = (
            features["customer_tenure_days"] /
            features["frequency"].replace(0, 1)
        )

        self.final_df = features.merge(
            self.customer_labels, on="customerid", how="inner"
        )

    def save_outputs(self):
        os.makedirs("data/processed", exist_ok=True)

        output_path = "data/processed/customer_features.csv"
        self.final_df.to_csv(output_path, index=False)

        info = {
            "total_customers": int(len(self.final_df)),
            "total_features": int(len(self.final_df.columns) - 2),
            "churn_rate": float(self.final_df["churn"].mean()),
            "training_cutoff": str(self.training_cutoff),
            "observation_end": str(self.observation_end),
            "churn_window_days": self.churn_window_days
        }

        with open("data/processed/feature_info.json", "w") as f:
            json.dump(info, f, indent=4)

        print(f"Saved customer features → {output_path}")
        print("Phase 4 COMPLETED SUCCESSFULLY")

    def run(self):
        self.find_valid_temporal_split()
        self.define_churn()
        self.create_features()
        self.save_outputs()


if __name__ == "__main__":
    fe = FeatureEngineer("data/processed/cleaned_transactions.csv")
    fe.run()
