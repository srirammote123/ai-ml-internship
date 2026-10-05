from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def load_data() -> pd.DataFrame:
    """Load and clean the Titanic dataset used for model training."""
    df = pd.read_csv(DATA_URL).drop_duplicates().copy()

    df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = pd.NA
    df.loc[df["Fare"] < 0, "Fare"] = pd.NA
    df.loc[~df["Pclass"].isin([1, 2, 3]), "Pclass"] = pd.NA

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    return df


def train_model(df: pd.DataFrame) -> Pipeline:
    """Train the same Logistic Regression pipeline used in Task 2."""
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    features = numeric_features + categorical_features
    model.fit(df[features], df["Survived"])
    return model


def get_integer(prompt: str, minimum: int, maximum: int) -> int:
    """Read an integer and reject non-numeric or out-of-range input."""
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("Invalid input. Please enter a whole number.")
            continue

        if not minimum <= number <= maximum:
            print(f"Invalid value. Please enter a number from {minimum} to {maximum}.")
            continue

        return number


def get_float(prompt: str, minimum: float, maximum: float) -> float:
    """Read a decimal number and reject non-numeric or out-of-range input."""
    while True:
        value = input(prompt).strip()
        try:
            number = float(value)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            continue

        if not minimum <= number <= maximum:
            print(f"Invalid value. Please enter a value from {minimum} to {maximum}.")
            continue

        return number


def get_choice(prompt: str, choices: tuple[str, ...]) -> str:
    """Read a case-insensitive choice from an allowed list."""
    allowed = {choice.lower(): choice.lower() for choice in choices}

    while True:
        value = input(prompt).strip().lower()
        if value in allowed:
            return allowed[value]

        print(f"Invalid choice. Please choose one of: {', '.join(choices)}.")


def collect_input() -> pd.DataFrame:
    """Collect and validate one passenger's details from the user."""
    print("\nEnter passenger details:")
    age = get_float("Age (0-100): ", 0, 100)
    fare = get_float("Fare (0-10000): ", 0, 10000)
    pclass = get_integer("Passenger class (1, 2, or 3): ", 1, 3)
    sib_sp = get_integer("Siblings/spouses aboard (0-20): ", 0, 20)
    parch = get_integer("Parents/children aboard (0-20): ", 0, 20)
    sex = get_choice("Sex (male/female): ", ("male", "female"))
    embarked = get_choice("Embarked (C/Q/S): ", ("C", "Q", "S"))

    return pd.DataFrame(
        [
            {
                "Age": age,
                "Fare": fare,
                "Pclass": pclass,
                "SibSp": sib_sp,
                "Parch": parch,
                "Sex": sex,
                "Embarked": embarked.upper(),
            }
        ]
    )


def predict_survival(model: Pipeline, passenger: pd.DataFrame) -> tuple[int, float]:
    """Generate a prediction and survival probability."""
    prediction = int(model.predict(passenger)[0])
    probability = float(model.predict_proba(passenger)[0][1])
    return prediction, probability


def main() -> None:
    print("=" * 55)
    print("TITANIC SURVIVAL PREDICTION APPLICATION")
    print("=" * 55)

    try:
        df = load_data()
        model = train_model(df)
        print("Model loaded successfully.")

        while True:
            passenger = collect_input()
            prediction, probability = predict_survival(model, passenger)

            print("\n" + "-" * 55)
            if prediction == 1:
                print("PREDICTION: The passenger is likely to SURVIVE.")
            else:
                print("PREDICTION: The passenger is likely NOT TO SURVIVE.")

            print(f"Estimated survival probability: {probability:.1%}")
            print("-" * 55)

            again = get_choice("Make another prediction? (yes/no): ", ("yes", "no"))
            if again == "no":
                print("Thank you for using the prediction application.")
                break

    except (KeyboardInterrupt, EOFError):
        print("\nApplication stopped safely.")
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        print("Please verify your inputs and try again.")


if __name__ == "__main__":
    main()
