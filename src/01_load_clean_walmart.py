# load_clean_walmart.py
# cleans up the walmart sales data and merges the 3 files into one table
# also builds the target column (HighSales) that the models will predict

import pandas as pd
import numpy as np

DATA_DIR = "../data"
OUT_DIR = "../outputs"


def load_raw():
    # dates need parsing or everything comes in as strings and messes up the merge later
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["Date"])
    features = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
    return train, features, stores


def merge_walmart(train, features, stores):
    # features.csv already has an IsHoliday column and so does train.csv
    # if I don't drop one first pandas renames both to IsHoliday_x / IsHoliday_y which is annoying
    features = features.drop(columns=["IsHoliday"])

    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date"], how="left")
    return df


def clean_walmart(df):
    before = len(df)

    # a few rows have negative weekly sales, these are returns/adjustments not real sales
    # doesn't make sense to keep them for a "high sales vs low sales" classifier
    df = df[df["Weekly_Sales"] > 0].copy()

    # the markdown columns are mostly empty - turns out walmart only started
    # tracking promo markdowns from nov 2011, so it's not really "missing data"
    # in the usual sense, it just means no promo was running yet. filling with 0
    # makes more sense than dropping rows or averaging
    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # CPI and Unemployment are missing for a handful of rows near the end of
    # the date range (features.csv runs a bit past train.csv). these barely
    # change week to week so just carrying the last known value per store forward
    df = df.sort_values(["Store", "Date"])
    df[["CPI", "Unemployment"]] = (
        df.groupby("Store")[["CPI", "Unemployment"]].ffill()
    )
    df = df.dropna(subset=["CPI", "Unemployment"])  # drops the few rows with nothing to ffill from

    after = len(df)
    print(f"Walmart: {before} rows -> {after} rows after cleaning ({before - after} dropped)")
    return df


def engineer_features(df):
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["TotalMarkDown"] = df[[c for c in df.columns if c.startswith("MarkDown")]].sum(axis=1)
    df["HasPromotion"] = (df["TotalMarkDown"] > 0).astype(int)

    # target column: was this store/dept/week in the top 25% of sales for THAT dept
    # doing it per department instead of overall because some departments just sell
    # way more than others in general, so a flat threshold across the whole dataset
    # would basically just be picking out big departments rather than genuinely
    # strong performing weeks
    dept_q75 = df.groupby("Dept")["Weekly_Sales"].transform(lambda x: x.quantile(0.75))
    df["HighSales"] = (df["Weekly_Sales"] >= dept_q75).astype(int)

    return df


if __name__ == "__main__":
    train, features, stores = load_raw()
    merged = merge_walmart(train, features, stores)
    cleaned = clean_walmart(merged)
    final = engineer_features(cleaned)

    print("\nFinal shape:", final.shape)
    print("\nClass balance (HighSales):")
    print(final["HighSales"].value_counts(normalize=True).round(3))

    final.to_csv(f"{OUT_DIR}/walmart_processed.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/walmart_processed.csv")
