
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "transactions_raw.csv"


# --------------------------------------------------
# Load and prepare cleaned data
# --------------------------------------------------

def load_clean_data():
    """Load raw transactions and apply the Part A cleaning pipeline."""
    from prepare import clean_transactions, add_features

    raw_df = pd.read_csv(DATA_PATH)

    cleaned_df = clean_transactions(raw_df)
    featured_df = add_features(cleaned_df)

    return featured_df


# --------------------------------------------------
# Split data before learning preprocessing values
# --------------------------------------------------

def split_data(df):
    """Create stratified train, validation, and test sets."""

    target = "is_fraud"

    excluded_columns = [
        "txn_id",
        "customer_id",
        "chargeback_filed",
        target,
    ]

    X = df.drop(columns=excluded_columns).copy()
    y = df[target].astype(int).copy()

    # First: 60% train, 40% temporary data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Then split the temporary data equally:
    # 20% validation and 20% test overall
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_data(X_train, X_val, X_test):
    """
    Fit imputation and category encoding on training data only.
    Apply the learned transformations to validation and test data.
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    categorical_columns = [
        "merchant_category",
        "city",
        "payment_mode",
    ]


    # Learn the distance median from training data only.
    distance_imputer = SimpleImputer(strategy="median")

    X_train[["distance_from_home_km"]] = (
        distance_imputer.fit_transform(
            X_train[["distance_from_home_km"]]
        )
    )

    X_val[["distance_from_home_km"]] = (
        distance_imputer.transform(
            X_val[["distance_from_home_km"]]
        )
    )

    X_test[["distance_from_home_km"]] = (
        distance_imputer.transform(
            X_test[["distance_from_home_km"]]
        )
    )

    training_distance_median = float(
        distance_imputer.statistics_[0]
    )

    # One-hot encode categorical columns.
    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns,
        dtype=int,
    )

    X_val = pd.get_dummies(
        X_val,
        columns=categorical_columns,
        dtype=int,
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_columns,
        dtype=int,
    )

    # Align validation and test columns with training columns.
    X_val = X_val.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # Make the feature order deterministic.
    sorted_columns = sorted(X_train.columns)

    X_train = X_train.reindex(columns=sorted_columns)
    X_val = X_val.reindex(columns=sorted_columns)
    X_test = X_test.reindex(columns=sorted_columns)

    return (
        X_train,
        X_val,
        X_test,
        training_distance_median,
    )



#---------------------------------------------------
# Model evaluation and comparison
# --------------------------------------------------



 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate a model on validation data and print fraud metrics."""

    predictions = model.predict(X_val)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_val, predictions)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def make_logistic_regression(class_weight=None):
    """Create a reproducible scaled logistic regression pipeline."""

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def run_cross_validation_and_tuning(X_train, y_train, X_val, y_val):
    """Compare models using CV and tune a Random Forest."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    balanced_model = make_logistic_regression(
        class_weight="balanced"
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\n" + "=" * 55)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 55)

    for name, model in [
        ("Balanced Logistic Regression", balanced_model),
        ("Random Forest", random_forest),
    ]:
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        print(f"\n{name}")
        for metric in scoring:
            values = scores[f"test_{metric}"]
            print(
                f"{metric.capitalize()}: "
                f"{values.mean():.4f} "
                f"+/- {values.std():.4f}"
            )

    print("\n" + "=" * 55)
    print("RANDOM FOREST GRID SEARCH")
    print("=" * 55)

    parameter_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    }

    grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print(f"Best mean CV F1: {grid_search.best_score_:.4f}")

    # Evaluate the tuned model on validation data.
    
    best_model = grid_search.best_estimator_

    evaluate_model(
        best_model,
        X_val,
        y_val,
        "Tuned Random Forest",
    )

    print("\n" + "=" * 55)
    print("VALIDATION THRESHOLD TUNING")
    print("=" * 55)

    from sklearn.metrics import precision_score, recall_score, f1_score

    probabilities = best_model.predict_proba(X_val)[:, 1]
    threshold_results = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        threshold_results.append({
            "threshold": round(float(threshold), 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        print(
            f"Threshold={threshold:.1f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    eligible = [
        result for result in threshold_results
        if result["recall"] >= 0.65
    ]

    if not eligible:
        raise ValueError(
            "No threshold achieved validation recall >= 0.65"
        )

    selected = max(
        eligible,
        key=lambda result: result["threshold"],
    )

    selected_threshold = selected["threshold"]

    print("\nSelected threshold:", selected_threshold)
    print("Selected threshold metrics:", selected)

    return best_model, grid_search.best_params_, selected_threshold



def main():
    df = load_clean_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        distance_median,
    ) = preprocess_data(X_train, X_val, X_test)

    print("=" * 55)
    print("PART B: BASELINE MODEL COMPARISON")
    print("=" * 55)
    print("Training rows:", len(X_train_processed))
    print("Validation rows:", len(X_val_processed))
    print("Test rows:", len(X_test_processed))
    print("Training-only distance median:", distance_median)
    print("Number of features:", X_train_processed.shape[1])

    tuned_rf, best_rf_params, selected_threshold = (
        run_cross_validation_and_tuning(
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
        )
    )

    results = {}

    # Model 1: Plain logistic regression
    plain_model = make_logistic_regression()
    plain_model.fit(X_train_processed, y_train)

    results["Plain Logistic Regression"] = evaluate_model(
        plain_model,
        X_val_processed,
        y_val,
        "Plain Logistic Regression",
    )

    # Model 2: Class-weight-balanced logistic regression
    balanced_model = make_logistic_regression(
        class_weight="balanced"
    )
    balanced_model.fit(X_train_processed, y_train)

    results["Balanced Logistic Regression"] = evaluate_model(
        balanced_model,
        X_val_processed,
        y_val,
        "Balanced Logistic Regression",
    )

    # Model 3: Random oversampling on training data only
    training_data = X_train_processed.copy()
    training_data["is_fraud"] = y_train.to_numpy()

    majority = training_data[
        training_data["is_fraud"] == 0
    ]
    minority = training_data[
        training_data["is_fraud"] == 1
    ]

    minority_oversampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE,
    )

    oversampled_data = pd.concat(
        [majority, minority_oversampled],
        ignore_index=True,
    )

    X_train_oversampled = oversampled_data.drop(
        columns=["is_fraud"]
    )
    y_train_oversampled = oversampled_data["is_fraud"]

    oversampled_model = make_logistic_regression()
    oversampled_model.fit(
        X_train_oversampled,
        y_train_oversampled,
    )

    print(
        "\nOversampled training rows:",
        len(X_train_oversampled),
    )
    print(
        "Oversampled fraud cases:",
        int(y_train_oversampled.sum()),
    )

    results["Oversampled Logistic Regression"] = evaluate_model(
        oversampled_model,
        X_val_processed,
        y_val,
        "Oversampled Logistic Regression",
    )

    # Compare results using fraud F1-score.
    comparison = pd.DataFrame([
        {
            "model": name,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "accuracy": result["accuracy"],
        }
        for name, result in results.items()
    ])

    print("\n" + "=" * 55)
    print("VALIDATION RESULTS COMPARISON")
    print("=" * 55)
    print(
        comparison.sort_values(
            "f1",
            ascending=False,
        ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nNote: The test set has not been used for model selection.")
    print("Baseline model comparison complete.")

    
def build_final_model():
    """Train the selected fraud model and return API-ready artifacts."""

    df = load_clean_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        distance_median,
    ) = preprocess_data(X_train, X_val, X_test)

    model, best_params, threshold = run_cross_validation_and_tuning(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
    )

    # Combine train and validation data for final fitting.
    X_final = pd.concat(
        [X_train_processed, X_val_processed],
        axis=0,
    )
    y_final = pd.concat([y_train, y_val], axis=0)

    final_model = clone(model)
    final_model.fit(X_final, y_final)

    return {
        "model": final_model,
        "columns": list(X_final.columns),
        "median": distance_median,
        "threshold": threshold,
        "best_params": best_params,
    }





if __name__ == "__main__":
    main()

