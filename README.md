# AI/ML Internship

## Task 1: Data Exploration

This project completes the first internship task:

1. Select a real-world dataset.
2. Load and inspect the dataset using Pandas.
3. Clean missing, duplicate, and incorrect data.
4. Create basic visualizations to understand the dataset.

### Dataset

**Titanic passenger dataset** — a real-world dataset containing passenger demographics, ticket information, fares, and survival outcomes from the Titanic disaster.

The analysis script downloads the CSV from a public GitHub mirror and loads it directly with Pandas.

## Task 2: Machine Learning Model

Task 2 builds a **binary classification model** to predict whether a Titanic passenger survived.

### Implementation

- Uses the cleaned Titanic dataset from Task 1.
- Splits the data into **80% training** and **20% testing** sets.
- Uses a **stratified split** with random_state=42.
- Preprocesses numerical features with median imputation and standard scaling.
- Preprocesses categorical features with most-frequent imputation and one-hot encoding.
- Trains a **Logistic Regression** classifier using Scikit-learn.
- Evaluates the model using Accuracy, Precision, Recall, F1-score, classification report, and confusion matrix.

### Run Task 2

    pip install -r requirements.txt
    python task_2_machine_learning_model.py

A confusion-matrix image is generated in task_2_outputs/.

## Task 3: Prediction Application

Task 3 provides a command-line application that accepts passenger information and uses the trained Logistic Regression pipeline to predict Titanic survival.

### Implementation

- Accepts age, fare, passenger class, family counts, sex, and embarkation port.
- Validates numeric values and allowed categories.
- Rejects invalid, non-numeric, negative, or out-of-range inputs.
- Trains the Task 2 model pipeline on the Titanic dataset before making predictions.
- Displays a clear **SURVIVE / NOT TO SURVIVE** prediction.
- Displays the estimated survival probability.
- Supports making multiple predictions in one run.
- Handles Ctrl+C, end-of-input, and unexpected runtime errors safely.

### Run Task 3

    pip install -r requirements.txt
    python task_3_prediction_app.py

### Example input

    Age (0-100): 25
    Fare (0-10000): 50
    Passenger class (1, 2, or 3): 2
    Siblings/spouses aboard (0-20): 0
    Parents/children aboard (0-20): 0
    Sex (male/female): female
    Embarked (C/Q/S): S

The application then displays the predicted outcome and survival probability.

## Project structure

    ai-ml-internship/
    ├── README.md
    ├── requirements.txt
    ├── task_1_data_exploration.py
    ├── task_2_machine_learning_model.py
    └── task_3_prediction_app.py

### Task 1 Visualizations

Task 1 produces:

- Survival count
- Age distribution
- Survival by passenger class
- Gender vs survival
- Fare distribution