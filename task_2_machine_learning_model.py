from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
OUTPUT_DIR = Path("task_2_outputs")


def load_and_clean_data() -> pd.DataFrame:
    """Load the Titanic dataset and perform basic cleaning."""
    df = pd.read_csv(DATA_URL)

    df = df.drop_duplicates().copy()
    df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = pd.NA
    df.loc[df["Fare"] < 0, "Fare"] = pd.NA
    df.loc[~df["Pclass"].isin([1, 2, 3]), "Pclass"] = pd.NA

    # Fill values used by the model; other columns are intentionally excluded.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    return df


def build_model() -> Pipeline:
    """Build a preprocessing + logistic regression classification pipeline."""
    numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
    categorical_features = ["Sex", "Embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def save_confusion_matrix(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a confusion-matrix visualization for the test predictions."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Did Not Survive", "Survived"],
    )
    display.plot()
    plt.title("Titanic Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    print("Loading Titanic dataset...")
    df = load_and_clean_data()

    target = "Survived"
    features = ["Age", "Fare", "Pclass", "SibSp", "Parch", "Sex", "Embarked"]

    x = df[features]
    y = df[target]

    # 80% training data and 20% testing data.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n=== TASK 2: MACHINE LEARNING MODEL ===")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples:  {len(x_test)}")
    print("Model: Logistic Regression")
    print("Problem: Binary classification (predict passenger survival)")
    print("\n=== EVALUATION METRICS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Did Not Survive", "Survived"],
            zero_division=0,
        )
    )

    save_confusion_matrix(model, x_test, y_test)
    print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print("Task 2 completed successfully.")


if __name__ == "__main__":
    main()
