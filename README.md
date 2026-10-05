# AI/ML Internship

## Project Overview

This repository contains the complete Artificial Intelligence & Machine Learning internship work, progressing from data exploration to a user-facing AI application.

## Task 1: Data Exploration

Uses the real-world Titanic passenger dataset. The project loads and inspects the data with Pandas, removes duplicates, handles invalid values and missing data, and creates basic visualizations.

Run:

    python task_1_data_exploration.py

Visualizations include survival count, age distribution, survival by class, gender vs survival, and fare distribution.

## Task 2: Machine Learning Model

Builds a **binary classification model** to predict whether a Titanic passenger survived.

- 80% training / 20% testing split
- Stratified split with random_state=42
- Logistic Regression using Scikit-learn
- Numeric imputation and scaling
- Categorical imputation and one-hot encoding
- Accuracy, precision, recall, F1-score, classification report, and confusion matrix

Run:

    python task_2_machine_learning_model.py

## Task 3: Prediction Application

Provides a command-line prediction application that accepts passenger details and produces a survival prediction and probability.

- Validates numeric ranges and categorical choices
- Handles invalid input and unexpected runtime errors
- Supports multiple predictions

Run:

    python task_3_prediction_app.py

## Task 4: Practical AI Project

### Titanic AI Survival Assistant

Task 4 combines the machine-learning model with a **Streamlit user interface** to solve a practical prediction problem: estimating a Titanic passenger's survival probability from passenger information.

### Problem and usefulness

The application demonstrates how a trained classification model can be turned into an accessible AI tool. A user can enter passenger characteristics and immediately receive a predicted outcome and estimated probability. Model evaluation metrics are shown in the same interface so users can understand that predictions have measurable performance rather than being guaranteed answers.

### Dataset

The project uses the Titanic passenger dataset from a public CSV mirror. The dataset contains passenger survival outcomes and attributes such as age, sex, passenger class, fare, family counts, and embarkation port. Duplicate and clearly invalid records are cleaned before training.

### Approach

1. Load and clean the Titanic dataset.
2. Split the data into 80% training and 20% testing sets using stratification.
3. Preprocess numerical and categorical features in a Scikit-learn pipeline.
4. Train a Logistic Regression classifier.
5. Evaluate the model on the held-out test set.
6. Present predictions through a Streamlit interface.

### Results

The application reports **accuracy, precision, recall, and F1 score** on the held-out test set. These metrics are calculated automatically when the app starts, so the displayed results remain tied to the exact implementation in the repository rather than being manually entered.

### Limitations

- The Titanic dataset is historical and relatively small.
- The model is intended for education and demonstration, not real-world safety or decision-making.
- Predictions can be wrong for individual passengers.
- Only a limited set of passenger attributes is used.
- The estimated probability is a model output, not a guarantee.
- The app requires access to the public dataset URL when the dataset is not already cached.

### Future improvements

- Compare Logistic Regression with Random Forest and gradient-boosting models.
- Add cross-validation and hyperparameter tuning.
- Calibrate predicted probabilities.
- Add experiment tracking and model versioning.
- Deploy the Streamlit application to the cloud.
- Add automated tests and continuous integration.
- Add prediction history and downloadable results.

### Run Task 4

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run task_4_ai_project.py

The browser interface accepts:

- Age
- Fare
- Passenger class
- Sex
- Siblings/spouses aboard
- Parents/children aboard
- Embarkation port

Then click **Predict Survival** to display the prediction and estimated survival probability.

## Project structure

    ai-ml-internship/
    ├── README.md
    ├── requirements.txt
    ├── task_1_data_exploration.py
    ├── task_2_machine_learning_model.py
    ├── task_3_prediction_app.py
    └── task_4_ai_project.py