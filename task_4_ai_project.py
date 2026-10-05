"""Task 4: Practical AI Project - Titanic Survival Assistant.

Run with:
    streamlit run task_4_ai_project.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and clean the real-world Titanic dataset."""
    df = pd.read_csv(DATA_URL).drop_duplicates().copy()
    df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = pd.NA
    df.loc[df["Fare"] < 0, "Fare"] = pd.NA
    df.loc[~df["Pclass"].isin([1, 2, 3]), "Pclass"] = pd.NA

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
    return df


def build_pipeline() -> Pipeline:
    """Create the classification pipeline used by the application."""
    numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
    categorical_features = ["Sex", "Embarked"]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


@st.cache_resource
def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Train the model and calculate held-out test metrics."""
    features = ["Age", "Fare", "Pclass", "SibSp", "Parch", "Sex", "Embarked"]
    x = df[features]
    y = df["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
    }
    return model, metrics


def main() -> None:
    st.set_page_config(
        page_title="Titanic AI Survival Assistant",
        page_icon="🚢",
        layout="centered",
    )

    st.title("🚢 Titanic AI Survival Assistant")
    st.write(
        "A practical machine-learning application that estimates Titanic "
        "passenger survival probability from passenger details."
    )

    try:
        df = load_data()
        model, metrics = train_model(df)
    except Exception as exc:
        st.error("The application could not load or train the model.")
        st.exception(exc)
        return

    st.subheader("Passenger Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        fare = st.number_input("Fare", min_value=0.0, max_value=10000.0, value=32.0)
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["female", "male"])
    with col2:
        sib_sp = st.number_input(
            "Siblings / Spouses Aboard", min_value=0, max_value=20, value=0, step=1
        )
        parch = st.number_input(
            "Parents / Children Aboard", min_value=0, max_value=20, value=0, step=1
        )
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    passenger = pd.DataFrame(
        [
            {
                "Age": age,
                "Fare": fare,
                "Pclass": pclass,
                "SibSp": sib_sp,
                "Parch": parch,
                "Sex": sex,
                "Embarked": embarked,
            }
        ]
    )

    if st.button("Predict Survival", type="primary", use_container_width=True):
        try:
            prediction = int(model.predict(passenger)[0])
            probability = float(model.predict_proba(passenger)[0][1])

            if prediction == 1:
                st.success("Prediction: Likely to SURVIVE")
            else:
                st.error("Prediction: Likely NOT TO SURVIVE")

            st.metric("Estimated Survival Probability", f"{probability:.1%}")
            st.progress(probability)
        except Exception as exc:
            st.error("Prediction failed. Please check the entered values.")
            st.exception(exc)

    st.divider()
    st.subheader("Model Evaluation")
    st.caption("Metrics are calculated on the held-out 20% test set used by Task 2.")

    metric_cols = st.columns(4)
    for column, (name, value) in zip(metric_cols, metrics.items()):
        column.metric(name, f"{value:.1%}")

    with st.expander("Usefulness and limitations"):
        st.markdown(
            """
**Usefulness**
- Demonstrates an end-to-end machine-learning workflow.
- Provides an immediate, understandable prediction from user-entered data.
- Shows model performance alongside the prediction rather than hiding evaluation results.

**Limitations**
- The Titanic dataset is historical and small, so this is an educational demonstration rather than a real-world safety system.
- Predictions reflect patterns in historical data and can be wrong for individual passengers.
- The model only uses a limited set of passenger attributes.
- Estimated probability is a model output, not a guarantee of survival.
- The app depends on access to the public dataset URL when starting without cached data.
            """
        )

    with st.expander("Future improvements"):
        st.markdown(
            """
- Compare Logistic Regression with Random Forest and gradient-boosting models.
- Add cross-validation and hyperparameter tuning.
- Add model calibration so predicted probabilities are more reliable.
- Track experiments and model versions.
- Package the application for cloud deployment.
- Add automated tests and continuous integration.
- Use a richer interface with prediction history and downloadable results.
            """
        )


if __name__ == "__main__":
    main()
