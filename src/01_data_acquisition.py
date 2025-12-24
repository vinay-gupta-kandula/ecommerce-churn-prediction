import pandas as pd
import json
from pathlib import Path


# =========================
# Paths
# =========================
RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
OUTPUT_PATH = Path("data/raw/data_quality_summary.json")


def main():
    print("Starting data acquisition...")

    # -------------------------
    # Load dataset
    # -------------------------
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH)

    # -------------------------
    # Standardize column names
    # -------------------------
    # Example:
    # "Customer ID" -> "customerid"
    # "InvoiceDate" -> "invoicedate"
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
    )

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # -------------------------
    # Data quality summary
    # -------------------------
    data_quality_summary = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": {
            "start": str(df["invoicedate"].min()),
            "end": str(df["invoicedate"].max())
        },
        "negative_quantities": int((df["quantity"] < 0).sum()),
        "zero_or_negative_prices": int((df["price"] <= 0).sum()),
        "unique_customers": int(df["customerid"].nunique()),
        "unique_products": int(df["stockcode"].nunique()),
        "unique_countries": int(df["country"].nunique())
    }

    # -------------------------
    # Save JSON artifact
    # -------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data_quality_summary, f, indent=4, default=str)

    print(f"Data quality summary saved to: {OUTPUT_PATH}")
    print("Phase 2 (Data Acquisition) completed successfully ✅")


if __name__ == "__main__":
    main()
