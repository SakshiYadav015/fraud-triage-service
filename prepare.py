
import pandas as pd


# ============================================================
# SECTION 1: LOAD THE RAW DATA
# ============================================================

def load_data(file_path="transactions_raw.csv"):
    """Load raw transaction data without modifying the source file."""
    df = pd.read_csv(file_path)
    return df


# ============================================================
# SECTION 2: AUDIT THE RAW DATA
# ============================================================

def audit_data(raw_df):
    """Inspect data quality before cleaning."""
    audit = {
        "rows": len(raw_df),
        "columns": len(raw_df.columns),
        "duplicate_rows": int(raw_df.duplicated().sum()),
        "missing_values": raw_df.isna().sum().to_dict(),
    } 

    
    # Audit missing values after removing exact duplicates
    audit_df = raw_df.drop_duplicates().copy()

    audit["rows_after_deduplication"] = len(audit_df)
    audit["missing_amount_after_deduplication"] = int(
        audit_df["amount_inr"].isna().sum()
    )
    audit["missing_distance_after_deduplication"] = int(
        audit_df["distance_from_home_km"].isna().sum()
    )


    amount = raw_df["amount_inr"]

    audit["missing_amount"] = int(amount.isna().sum())
    audit["invalid_amount"] = int(
        ((amount < 1) | (amount > 1_000_000)).sum()
    )

    for column in ("merchant_category", "city"):
        audit[f"{column}_unique_raw"] = int(raw_df[column].nunique())

    return audit


# ============================================================
# SECTION 3: CLEAN THE TRANSACTION DATA
# ============================================================

def clean_transactions(raw_df):
    """Return cleaned data while preserving the input DataFrame."""
    df = raw_df.copy()

    # Remove exact duplicate records.
    df = df.drop_duplicates()

    # Standardize category and city names.
    for column in ("merchant_category", "city"):
        df[column] = df[column].astype("string").str.strip().str.title()

    # Remove records with missing or impossible transaction amounts.
    df = df.dropna(subset=["amount_inr"])
    df = df[df["amount_inr"].between(1, 1_000_000, inclusive="both")]

    return df.reset_index(drop=True)


# ============================================================
# SECTION 4: ENGINEER FRAUD-RELATED FEATURES
# ============================================================

def add_features(df):
    """Add model features without modifying the input DataFrame."""
    result = df.copy()

    result["amount_vs_avg_ratio"] = (
        result["amount_inr"] / result["avg_amount_30d_inr"]
    )

    result["is_night"] = (
        result["hour_of_day"] <= 5
    ).astype(int)

    result["high_value"] = (
        result["amount_inr"] > 50_000
    ).astype(int)

    return result


# ============================================================
# SECTION 5: RUN THE ANALYSIS IN SMALL STEPS
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SECTION 1: LOADING DATA")
    print("=" * 60)

    raw_df = load_data()

    print("Dataset loaded successfully.")
    print("Shape (rows, columns):", raw_df.shape)
    print("\nFirst five transactions:")
    print(raw_df.head().to_string(index=False))

    print("\nColumn names:")
    print(raw_df.columns.tolist())

    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SECTION 2: RAW DATA AUDIT")
    print("=" * 60)

    audit = audit_data(raw_df)

    print("Raw rows:", audit["rows"])
    print("Raw columns:", audit["columns"])
    print("Exact duplicate rows:", audit["duplicate_rows"])
    print(
        "Rows after duplicate removal:",
        audit["rows_after_deduplication"]
    )
    print(
        "Missing amount after duplicate removal:",
        audit["missing_amount_after_deduplication"]
    )
    print(
        "Missing distance after duplicate removal:",
        audit["missing_distance_after_deduplication"]
    )
    print("Missing amount values:", audit["missing_amount"])
    print("Invalid amount values:", audit["invalid_amount"])

    print("\nMissing values by column:")
    print(pd.Series(audit["missing_values"]).to_string())

    print("\nUnique raw merchant categories:",
          audit["merchant_category_unique_raw"])
    print("Unique raw city spellings:",
          audit["city_unique_raw"])

    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SECTION 3: CLEANING")
    print("=" * 60)

    cleaned_df = clean_transactions(raw_df)

    print("Rows before cleaning:", len(raw_df))
    print("Rows after cleaning:", len(cleaned_df))
    print("Rows removed:", len(raw_df) - len(cleaned_df))
    print("Duplicates remaining:", cleaned_df.duplicated().sum())

    print("\nMerchant category counts:")
    print(cleaned_df["merchant_category"].value_counts().to_string())

    print("\nCity counts:")
    print(cleaned_df["city"].value_counts().to_string())

    print("\nPayment mode counts:")
    print(cleaned_df["payment_mode"].value_counts().to_string())

    print("\nAmount summary:")
    print(cleaned_df["amount_inr"].describe().to_string())

    print("\nMissing distance values after cleaning:",
          cleaned_df["distance_from_home_km"].isna().sum())

    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SECTION 4: FEATURE ENGINEERING")
    print("=" * 60)

    featured_df = add_features(cleaned_df)

    print("New feature columns:")
    print([
        "amount_vs_avg_ratio",
        "is_night",
        "high_value",
    ])

    print("\nNight transactions:", int(featured_df["is_night"].sum()))
    print("High-value transactions:", int(featured_df["high_value"].sum()))

    print(
        "Maximum amount-to-average ratio:",
        round(featured_df["amount_vs_avg_ratio"].max(), 2),
    )

    print("\nExample engineered features:")
    print(
        featured_df[
            [
                "amount_inr",
                "avg_amount_30d_inr",
                "hour_of_day",
                "amount_vs_avg_ratio",
                "is_night",
                "high_value",
            ]
        ].head().to_string(index=False)
    )

    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SECTION 5: FRAUD LABEL AUDIT")
    print("=" * 60)

    if "is_fraud" in featured_df.columns:
        fraud_counts = featured_df["is_fraud"].value_counts().sort_index()

        print("Fraud label counts:")
        print(fraud_counts.to_string())

        fraud_rate = featured_df["is_fraud"].mean() * 100
        print(f"\nFraud rate: {fraud_rate:.2f}%")

        print(
            "Always predict non-fraud accuracy:",
            f"{(1 - featured_df['is_fraud'].mean()) * 100:.2f}%",
        )

    print("\n" + "=" * 60)
    print("SECTION 6: LEAKAGE AUDIT")
    print("=" * 60)

    leakage_table = (
        cleaned_df.groupby("chargeback_filed")
        .agg(
            transaction_count=("is_fraud", "size"),
            fraud_count=("is_fraud", "sum"),
        )
    )
    print(leakage_table)

    print("\nLeakage explanation:")
    print(
        "chargeback_filed is a post-transaction outcome and can reveal "
        "information unavailable when the model makes a prediction."
    )
    print(
        "txn_id and customer_id are identifiers rather than generalizable "
        "risk signals, so they are excluded along with chargeback_filed "
        "and the target label is_fraud."
    )

    excluded_columns = [
        "txn_id",
        "customer_id",
        "chargeback_filed",
        "is_fraud",
    ]

    model_input_columns = [
        column for column in featured_df.columns
        if column not in excluded_columns
    ]

    print("\nModel input columns before one-hot encoding:")
    print(model_input_columns)
    print("Number of model input columns:", len(model_input_columns))
    print("\nAnalysis complete. No output CSV was written.")
     
    print("\n" + "=" * 60)
    print("SECTION 7: FUNCTION TESTS")
    print("=" * 60)

    # Test that cleaning does not modify the original DataFrame
    raw_copy = raw_df.copy(deep=True)
    cleaned_test = clean_transactions(raw_df)

    print(
        "clean_transactions preserves input:",
        raw_df.equals(raw_copy)
    )

    # Test that feature engineering does not modify its input
    cleaned_copy = cleaned_df.copy(deep=True)
    featured_test = add_features(cleaned_df)

    print(
        "add_features preserves input:",
        cleaned_df.equals(cleaned_copy)
    )

    # Test one-row inputs
    one_raw_row = raw_df.iloc[[0]].copy()
    one_cleaned_row = clean_transactions(one_raw_row)

    one_feature_row = add_features(one_cleaned_row)

    print("One-row cleaning works:", len(one_cleaned_row) == 1)
    print("One-row feature engineering works:", len(one_feature_row) == 1)