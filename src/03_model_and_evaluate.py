# model_and_evaluate.py
# trains random forest + neural network on both datasets and prints/saves
# the metrics. using the same 2 models on both datasets on purpose so the
# results can actually be compared side by side (this was also mentioned
# in the course Q&A - keep the models consistent if telling one story)

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report
)

OUT_DIR = "../outputs"
RANDOM_STATE = 42


def evaluate(y_true, y_pred, y_proba, model_name, dataset_name):
    metrics = {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {dataset_name} | {model_name} ---")
    for k, v in metrics.items():
        if k not in ("dataset", "model"):
            print(f"  {k:>10}: {v}")
    print(f"  confusion matrix:\n{cm}")
    return metrics  # returning this so it can get collected into one summary table later


def run_walmart():
    df = pd.read_csv("../outputs/walmart_processed.csv")

    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    feature_cols = [
        "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "TotalMarkDown", "HasPromotion", "Month", "WeekOfYear", "IsHoliday",
        "Type_B", "Type_C",
    ]
    X = df[feature_cols].copy()
    X["IsHoliday"] = X["IsHoliday"].astype(int)
    y = df["HighSales"]

    return train_evaluate(X, y, feature_cols, "Walmart (store sales)")


def run_retail():
    df = pd.read_csv("../outputs/retail_rfm_processed.csv")
    df["IsUK"] = (df["Country"] == "United Kingdom").astype(int)  # UK is ~91% of customers so just a flag not full one-hot

    # important: Monetary is NOT in the feature list. it's literally what
    # was used to build the HighValue label in the first place, so if it
    # went in as a feature the model would basically just be cheating off
    # the answer (this is target leakage, caught it while writing this)
    feature_cols = ["Recency", "Frequency", "DistinctProducts", "IsUK"]
    X = df[feature_cols].copy()
    y = df["HighValue"]

    return train_evaluate(X, y, feature_cols, "Online Retail II (customers)")


def train_evaluate(X, y, feature_cols, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # random forest doesn't care about feature scale so using the raw
    # (unscaled) X_train/X_test here is fine
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate(y_test, rf_pred, rf_proba,
                             "Random Forest", dataset_name))

    importances = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Feature importances ({dataset_name}):")
    print(importances.round(3).to_string())

    # neural nets are sensitive to feature scale so this one gets the
    # scaled version, otherwise training doesn't converge properly
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
        early_stopping=True, random_state=RANDOM_STATE
    )
    nn.fit(X_train_scaled, y_train)
    nn_pred = nn.predict(X_test_scaled)
    nn_proba = nn.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, nn_pred, nn_proba,
                             "Neural Network", dataset_name))

    return results, importances


if __name__ == "__main__":
    all_results = []

    walmart_results, walmart_importance = run_walmart()
    all_results.extend(walmart_results)

    retail_results, retail_importance = run_retail()
    all_results.extend(retail_results)

    results_df = pd.DataFrame(all_results)
    print("\n\n=== SUMMARY: ALL RESULTS ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(f"{OUT_DIR}/model_results_summary.csv", index=False)
    walmart_importance.to_csv(f"{OUT_DIR}/walmart_feature_importance.csv")
    retail_importance.to_csv(f"{OUT_DIR}/retail_feature_importance.csv")
    print(f"\nSaved results to {OUT_DIR}/")
