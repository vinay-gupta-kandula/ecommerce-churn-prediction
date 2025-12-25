"""
Phase 2: Data Acquisition

- Loads UCI Online Retail II dataset (Excel)
- Converts to CSV
- Normalizes schema differences
- Generates strict data quality summary JSON
"""

import pandas as pd
import os
import json

RAW_DIR = "data/raw"
EXCEL_FILE = "online_retail_II.xlsx"
CSV_FILE = "online_retail.csv"


def download_dataset():
    print("Starting Phase 2: Data Acquisition")

    os.makedirs(RAW_DIR, exist_ok=True)

    excel_path = os.path.join(RAW_DIR, EXCEL_FILE)
    csv_path = os.path.join(RAW_DIR, CSV_FILE)

    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Dataset not found. Place '{EXCEL_FILE}' inside data/raw/"
        )

    print(f"Found Excel file: {EXCEL_FILE}")

    xls = pd.ExcelFile(excel_path)
    print(f"Available sheets: {xls.sheet_names}")

    # Prefer 2010–2011 if present
    preferred_sheet = "Year 2010-2011" if "Year 2010-2011" in xls.sheet_names else xls.sheet_names[0]
    print(f"Using sheet: {preferred_sheet}")

    df = pd.read_excel(xls, sheet_name=preferred_sheet)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Normalize schema differences across sheets
    df.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Customer ID": "CustomerID",
            "Price": "UnitPrice"   # 🔑 FIX
        },
        inplace=True
    )

    required_cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("Converting to CSV...")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    return df


def generate_data_profile(df: pd.DataFrame):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    summary = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": {
            "start": str(df["InvoiceDate"].min()),
            "end": str(df["InvoiceDate"].max())
        },
        "negative_quantities": int((df["Quantity"] < 0).sum()),
        "cancelled_invoices": int(
            df["InvoiceNo"].astype(str).str.startswith("C").sum()
        ),
        "missing_customer_ids": int(df["CustomerID"].isnull().sum()),
        "missing_customer_ids_percentage": round(
            df["CustomerID"].isnull().mean() * 100, 2
        )
    }

    output_path = os.path.join(RAW_DIR, "data_quality_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Data quality summary saved to {output_path}")


if __name__ == "__main__":
    df = download_dataset()
    generate_data_profile(df)
    print("Phase 2 completed successfully.")
