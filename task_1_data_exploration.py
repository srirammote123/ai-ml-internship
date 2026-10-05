import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
OUTPUT_DIR = Path("plots")


def load_data() -> pd.DataFrame:
    """Load the real-world Titanic dataset from a public CSV source."""
    return pd.read_csv(DATA_URL)


def inspect_data(df: pd.DataFrame) -> None:
    """Print basic dataset information required for exploration."""
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe(include="all").transpose())

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing, duplicate, and clearly invalid values."""
    cleaned = df.copy()

    # Remove completely duplicated records.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Treat impossible ages/fare values as missing before imputation.
    cleaned.loc[(cleaned["Age"] < 0) | (cleaned["Age"] > 100), "Age"] = pd.NA
    cleaned.loc[cleaned["Fare"] < 0, "Fare"] = pd.NA

    # Titanic class is categorical and bounded to 1, 2, or 3.
    cleaned.loc[~cleaned["Pclass"].isin([1, 2, 3]), "Pclass"] = pd.NA

    # Fill numerical missing values with medians.
    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    cleaned["Fare"] = cleaned["Fare"].fillna(cleaned["Fare"].median())
    cleaned["Pclass"] = cleaned["Pclass"].fillna(cleaned["Pclass"].mode()[0])

    # Fill categorical missing values with the mode.
    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])
    cleaned["Sex"] = cleaned["Sex"].fillna(cleaned["Sex"].mode()[0])

    # Keep text fields consistent.
    cleaned["Sex"] = cleaned["Sex"].str.strip().str.lower()
    cleaned["Embarked"] = cleaned["Embarked"].str.strip().str.upper()

    return cleaned


def save_plots(df: pd.DataFrame) -> None:
    """Create basic visualizations and save them as PNG files."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Survival count.
    df["Survived"].value_counts().sort_index().plot(kind="bar")
    plt.title("Titanic Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Passengers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_count.png", dpi=150)
    plt.close()

    # 2. Age distribution.
    df["Age"].plot(kind="hist", bins=20)
    plt.title("Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
    plt.close()

    # 3. Survival by passenger class.
    survival_by_class = pd.crosstab(df["Pclass"], df["Survived"])
    survival_by_class.plot(kind="bar")
    plt.title("Survival by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Passengers")
    plt.xticks(rotation=0)
    plt.legend(["Did Not Survive", "Survived"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_by_class.png", dpi=150)
    plt.close()

    # 4. Gender vs survival.
    gender_survival = pd.crosstab(df["Sex"], df["Survived"])
    gender_survival.plot(kind="bar")
    plt.title("Gender vs Survival")
    plt.xlabel("Gender")
    plt.ylabel("Passengers")
    plt.xticks(rotation=0)
    plt.legend(["Did Not Survive", "Survived"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_vs_survival.png", dpi=150)
    plt.close()

    # 5. Fare distribution.
    df["Fare"].plot(kind="hist", bins=30)
    plt.title("Passenger Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fare_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    print("Loading Titanic dataset...")
    df = load_data()

    print("\n--- BEFORE CLEANING ---")
    inspect_data(df)

    duplicates_before = int(df.duplicated().sum())
    missing_before = int(df.isna().sum().sum())

    cleaned_df = clean_data(df)

    duplicates_after = int(cleaned_df.duplicated().sum())
    missing_after = int(cleaned_df.isna().sum().sum())

    print("\n--- AFTER CLEANING ---")
    inspect_data(cleaned_df)

    print("\n=== CLEANING SUMMARY ===")
    print(f"Missing values before: {missing_before}")
    print(f"Missing values after:  {missing_after}")
    print(f"Duplicate rows before: {duplicates_before}")
    print(f"Duplicate rows after:  {duplicates_after}")

    save_plots(cleaned_df)

    print(f"\nSaved visualizations to: {OUTPUT_DIR.resolve()}")
    print("Task 1 completed successfully.")


if __name__ == "__main__":
    main()
