import pandas as pd
import json
from pathlib import Path


# =========================
# Paths
# =========================
RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
CLEAN_DATA_PATH = Path("data/processed/cleaned_transactions.csv")
CLEANING_STATS_PATH = Path("data/processed/cleaning_statistics.json")
VALIDATION_REPORT_PATH = Path("data/processed/validation_report.json")


def main():
    print("Starting Phase 3: Data Cleaning...")

    # -------------------------
    # Load raw data
    # -------------------------
    df = pd.read_excel(RAW_DATA_PATH)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
    )

    initial_rows = len(df)

    stats = {
        "initial_rows": int(initial_rows)
    }

    # -------------------------
    # Convert invoice date
    # -------------------------
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # -------------------------
    # Remove missing customer IDs
    # -------------------------
    before = len(df)
    df = df[df["customerid"].notna()]
    stats["removed_missing_customerid"] = int(before - len(df))

    # -------------------------
    # Remove cancelled invoices (Invoice starts with 'C')
    # -------------------------
    before = len(df)
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    stats["removed_cancelled_invoices"] = int(before - len(df))

    # -------------------------
    # Remove negative quantities
    # -------------------------
    before = len(df)
    df = df[df["quantity"] > 0]
    stats["removed_negative_quantities"] = int(before - len(df))

    # -------------------------
    # Remove zero or negative prices
    # -------------------------
    before = len(df)
    df = df[df["price"] > 0]
    stats["removed_zero_or_negative_prices"] = int(before - len(df))

    # -------------------------
    # Remove duplicates
    # -------------------------
    before = len(df)
    df = df.drop_duplicates()
    stats["removed_duplicates"] = int(before - len(df))

    final_rows = len(df)
    retention_rate = round((final_rows / initial_rows) * 100, 2)

    stats["final_rows"] = int(final_rows)
    stats["retention_rate_percent"] = retention_rate

    # -------------------------
    # Save cleaned data
    # -------------------------
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # -------------------------
    # Save cleaning statistics
    # -------------------------
    with open(CLEANING_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    # -------------------------
    # Validation report
    # -------------------------
    validation_report = {
        "initial_rows": int(initial_rows),
        "final_rows": int(final_rows),
        "retention_rate_percent": retention_rate,
        "date_range": {
            "start": str(df["invoicedate"].min()),
            "end": str(df["invoicedate"].max())
        },
        "unique_customers": int(df["customerid"].nunique()),
        "unique_products": int(df["stockcode"].nunique()),
        "validation_status": "PASS" if 50 <= retention_rate <= 80 else "REVIEW"
    }

    with open(VALIDATION_REPORT_PATH, "w") as f:
        json.dump(validation_report, f, indent=4)

    print("Phase 3 completed successfully ✅")
    print(f"Retention rate: {retention_rate}%")
    print("Cleaned data saved to:", CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()
