
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "network_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "nids_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


# --------------------------------------------------
# FIND THE TARGET LABEL COLUMN
# --------------------------------------------------

def find_target_column(df):
    possible_names = [
        "label",
        "class",
        "attack",
        "category",
        "target",
        " Label",
    ]

    normalized = {
        str(col).strip().lower(): col for col in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    raise ValueError(
        "Could not find a label column. "
        "Expected a column such as Label, Class or Attack."
    )


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def load_and_clean_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place your CSV at dataset/network_data.csv"
        )

    print("\n[1/5] Loading dataset...")

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = find_target_column(df)

    # Remove rows without a target label.
    df = df.dropna(subset=[target_col]).copy()

    # Clean label text.
    df[target_col] = (
        df[target_col].astype(str).str.strip()
    )

    # Remove blank labels.
    df = df[df[target_col] != ""]

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Remove identifiers and columns that could leak identity.
    excluded_words = [
        "flow id",
        "source ip",
        "src ip",
        "destination ip",
        "dst ip",
        "timestamp",
        "source port",
        "destination ip",
    ]

    feature_df = df.drop(columns=[target_col])

    excluded_columns = [
        col for col in feature_df.columns
        if any(word in col.lower() for word in excluded_words)
    ]

    feature_df = feature_df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    # Convert feature columns to numeric.
    # Non-numeric columns are ignored in this Phase 1 model.
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(
            feature_df[col], errors="coerce"
        )

    # Replace infinite values with missing values.
    feature_df = feature_df.replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove columns with no usable values.
    feature_df = feature_df.dropna(axis=1, how="all")

    # Remove constant columns.
    feature_df = feature_df.loc[
        :, feature_df.nunique(dropna=True) > 1
    ]

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No usable numeric feature columns found."
        )

    # Remove rows where every feature is missing.
    valid_rows = feature_df.notna().any(axis=1)

    feature_df = feature_df.loc[valid_rows].copy()
    labels = df.loc[valid_rows, target_col].copy()

    # Normalize common benign label variations.
    labels = labels.replace({
        "BENIGN": "BENIGN",
        "Benign": "BENIGN",
        "benign": "BENIGN",
    })

    # Encode labels into integer values.
    class_names = sorted(labels.unique().tolist())
    class_to_id = {
        name: index for index, name in enumerate(class_names)
    }

    y = labels.map(class_to_id).astype(int)

    if len(class_names) < 2:
        raise ValueError(
            "At least two different classes are required."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Every attack class needs at least two examples. "
            "Your dataset is too small or too imbalanced."
        )

    print(f"Dataset rows: {len(feature_df):,}")
    print(f"Features: {feature_df.shape[1]}")
    print(f"Classes: {class_names}")

    return feature_df, y, class_names


# --------------------------------------------------
# MODEL CREATION
# --------------------------------------------------

def create_models():
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    xgboost = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])

    return {
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_and_evaluate():
    print("\nStarting NIDS training process...")

    X, y, class_names = load_and_clean_data()

    print("\n[2/5] Splitting dataset...")

    number_of_classes = len(class_names)

    # Make sure the test set can contain each class.
    test_fraction = max(
        0.20,
        number_of_classes / len(y)
    )
    test_fraction = min(test_fraction, 0.40)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")

    models = create_models()
    results = {}
    trained_models = {}

    print("\n[3/5] Training models...")

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        try:
            model.fit(X_train, y_train)

            predictions = model.predict(X_test).astype(int)

            results[model_name] = {
                "accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "f1_score": float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    predictions,
                    labels=list(range(number_of_classes)),
                ).tolist(),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    labels=list(range(number_of_classes)),
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0,
                ),
            }

            trained_models[model_name] = model

            print(
                f"{model_name} weighted F1: "
                f"{results[model_name]['f1_score']:.4f}"
            )

        except Exception as error:
            print(f"{model_name} could not train: {error}")

    if not trained_models:
        raise RuntimeError(
            "Neither model could be trained successfully."
        )

    print("\n[4/5] Selecting the best model...")

    best_model_name = max(
        trained_models,
        key=lambda name: results[name]["f1_score"]
    )

    best_model = trained_models[best_model_name]

    print(f"Best model: {best_model_name}")

    # Extract feature importance from the selected model.
    classifier = best_model.named_steps["classifier"]

    feature_importance = {}

    if hasattr(classifier, "feature_importances_"):
        feature_importance = {
            str(name): float(value)
            for name, value in zip(
                X.columns,
                classifier.feature_importances_,
            )
        }

        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        )

    # Save reusable model bundle.
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "class_names": class_names,
        "class_to_id": {
            name: index for index, name in enumerate(class_names)
        },
        "feature_importance": feature_importance,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    metrics = {
        "dataset": DATASET_PATH.name,
        "dataset_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "class_names": class_names,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "best_model": best_model_name,
        "models": results,
        "feature_importance": feature_importance,
        "warning": (
            "Metrics from a small sample dataset are only a "
            "workflow demonstration, not evidence of real-world "
            "detection performance."
            if len(X) < 1000 else ""
        ),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n[5/5] Training completed!")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    print("\nModel comparison:")
    for name, score in results.items():
        print(f"{name}: F1 = {score['f1_score']:.4f}")

    print("\nNIDS training engine is ready!")


if __name__ == "__main__":
    train_and_evaluate()
