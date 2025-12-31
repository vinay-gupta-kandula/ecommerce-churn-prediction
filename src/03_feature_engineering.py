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

        required_cols = {"invoice", "stockcode", "quantity", "invoicedate", "price", "customerid", "country"}
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
            obs_df = self.df[(self.df["invoicedate"] > cutoff) & (self.df["invoicedate"] <= self.observation_end)]

            train_customers = set(train_df["customerid"].unique())
            obs_customers = set(obs_df["customerid"].unique())
            
            if not train_customers: continue
            
            churn_rate = len(train_customers - obs_customers) / len(train_customers)

            print(f"Trying {days} days → churn rate: {round(churn_rate,3)}")
            if 0.20 <= churn_rate <= 0.40:
                self.training_cutoff = cutoff
                self.churn_window_days = days
                self.train_df = train_df
                self.obs_df = obs_df
                print(f"✅ Selected churn window: {days} days. Final churn rate: {round(churn_rate,3)}")
                return
        raise ValueError("❌ No temporal window produced churn between 20–40%.")

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

        # 1. Base RFM and Product Features
        features = df.groupby("customerid").agg(
            frequency=("invoice", "nunique"),
            monetary_value=("total_price", "sum"),
            avg_order_value=("total_price", "mean"),
            total_quantity=("quantity", "sum"),
            unique_products=("stockcode", "nunique"),
            min_unit_price=("price", "min"),
            max_unit_price=("price", "max"),
            avg_unit_price=("price", "mean"),
            std_unit_price=("price", "std"),
            country_count=("country", "nunique"),
            first_purchase=("invoicedate", "min"),
            last_purchase=("invoicedate", "max"),
        ).reset_index()

        # 2. Tenure and Recency
        features["recency_days"] = (self.training_cutoff - features["last_purchase"]).dt.days
        features["customer_tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
        
        # 3. Basket Size Statistics
        basket_stats = df.groupby(["customerid", "invoice"])["quantity"].sum().reset_index()
        basket_features = basket_stats.groupby("customerid")["quantity"].agg(
            avg_basket_size="mean",
            std_basket_size="std",
            max_basket_size="max"
        ).reset_index()
        features = features.merge(basket_features, on="customerid", how="left")

        # 4. Activity Windows
        for days in [30, 60, 90]:
            window_cutoff = self.training_cutoff - timedelta(days=days)
            window_counts = df[df["invoicedate"] > window_cutoff].groupby("customerid")["invoice"].nunique().reset_index()
            window_counts.columns = ["customerid", f"purchases_last_{days}_days"]
            features = features.merge(window_counts, on="customerid", how="left")
        
       # 5. Segment Scoring (RFM Quartiles)
        features = features.fillna(0)
        
        # Helper function to handle duplicate bins dynamically
        def safe_qcut(series, labels):
            # Calculate the unique quantiles first
            quantiles = series.quantile([0, 0.25, 0.5, 0.75, 1.0]).unique()
            # Determine how many labels we actually need
            num_bins = len(quantiles) - 1
            if num_bins <= 0: return [1] * len(series) # Fallback if data is too uniform
            
            actual_labels = labels[:num_bins] if labels[0] == 1 else labels[-(num_bins):]
            return pd.qcut(series, q=[0, 0.25, 0.5, 0.75, 1.0], labels=actual_labels, duplicates='drop').astype(int)

        features['recency_score'] = safe_qcut(features['recency_days'], [4, 3, 2, 1])
        features['freq_score'] = safe_qcut(features['frequency'], [1, 2, 3, 4])
        features['monetary_score'] = safe_qcut(features['monetary_value'], [1, 2, 3, 4])
        
        features['rfm_total'] = features['recency_score'] + features['freq_score'] + features['monetary_score']
        # 6. ADVANCED BEHAVIORAL FEATURES (To hit 25+ count)
        # Ratio/Velocity features provide high signal for churn
        features['monetary_per_txn'] = features['monetary_value'] / (features['frequency'] + 1)
        features['quantity_per_txn'] = features['total_quantity'] / (features['frequency'] + 1)
        features['tenure_velocity'] = features['frequency'] / (features['customer_tenure_days'] + 1)
        features['variety_ratio'] = features['unique_products'] / (features['total_quantity'] + 1)
        features['price_stability'] = features['std_unit_price'] / (features['avg_unit_price'] + 0.001)
        features['basket_growth'] = features['max_basket_size'] / (features['avg_basket_size'] + 1)
        
        # Log transforms for skewed data
        features['log_monetary'] = np.log1p(features['monetary_value'])
        features['log_frequency'] = np.log1p(features['frequency'])

        # Fill any new NaNs
        features = features.fillna(0)
        
        self.final_df = features.merge(self.customer_labels, on="customerid", how="inner")

    def save_outputs(self):
        os.makedirs("data/processed", exist_ok=True)
        # Drop datetime columns before saving for modeling
        model_ready_df = self.final_df.drop(columns=['first_purchase', 'last_purchase'], errors='ignore')
        model_ready_df.to_csv("data/processed/customer_features.csv", index=False)

        info = {
            "total_customers": int(len(self.final_df)),
            "total_features": int(len(model_ready_df.columns) - 2), # Exclude customerid and churn
            "churn_rate": float(self.final_df["churn"].mean()),
            "training_cutoff": str(self.training_cutoff),
            "observation_end": str(self.observation_end),
            "churn_window_days": self.churn_window_days,
            "feature_list": list(model_ready_df.columns)
        }

        with open("data/processed/feature_info.json", "w") as f:
            json.dump(info, f, indent=4)

        print(f"Saved {info['total_features']} features for {info['total_customers']} customers.")
        print("Phase 4 COMPLETED SUCCESSFULLY")

    def run(self):
        self.find_valid_temporal_split()
        self.define_churn()
        self.create_features()
        self.save_outputs()

if __name__ == "__main__":
    fe = FeatureEngineer("data/processed/cleaned_transactions.csv")
    fe.run()
    