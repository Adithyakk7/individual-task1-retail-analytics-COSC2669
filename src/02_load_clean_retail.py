# load_clean_retail.py
# loads the online retail II transaction data and rolls it up into
# per-customer RFM features (recency, frequency, monetary) which is
# the standard way to do customer segmentation on this kind of data

import pandas as pd
import numpy as np

DATA_DIR = "../data"
OUT_DIR = "../outputs"


def load_raw():
    # the file has 2 sheets, one for each year of data, just stack them together
    xl = pd.ExcelFile(f"{DATA_DIR}/online_retail_II.xlsx")
    sheets = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
    df = pd.concat(sheets, ignore_index=True)
    return df


def clean_transactions(df):
    before = len(df)

    # invoice numbers starting with 'C' are cancellations/returns, not actual
    # purchases - keeping these in would throw off the monetary totals
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # rows with no Customer ID are basically anonymous/guest checkouts, there's
    # no way to attach them to a customer so they can't be used for RFM anyway
    df = df.dropna(subset=["Customer ID"])

    # some rows have 0 or negative quantity/price, these look like data entry
    # issues or adjustment entries rather than real purchases
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]

    after = len(df)
    print(f"Online Retail II: {before} rows -> {after} rows after cleaning ({before - after} dropped)")
    return df


def build_rfm(df):
    # "today" for recency purposes = 1 day after the last transaction in the dataset
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        DistinctProducts=("StockCode", "nunique"),
        Country=("Country", "first"),
    ).reset_index()

    # target: top 25% of customers by total spend = "high value"
    # (same top-25% logic as the walmart target, keeps things consistent)
    monetary_q75 = rfm["Monetary"].quantile(0.75)
    rfm["HighValue"] = (rfm["Monetary"] >= monetary_q75).astype(int)

    return rfm


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean_transactions(raw)
    rfm = build_rfm(cleaned)

    print("\nCustomer-level shape:", rfm.shape)
    print("\nClass balance (HighValue):")
    print(rfm["HighValue"].value_counts(normalize=True).round(3))
    print("\nRFM summary:")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe())

    rfm.to_csv(f"{OUT_DIR}/retail_rfm_processed.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/retail_rfm_processed.csv")
