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
- Evaluates the model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report
  - Confusion matrix

### Features

The model uses:

- Age
- Fare
- Passenger class
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Sex
- Embarkation port

### Run Task 2

Install dependencies:

    pip install -r requirements.txt

Run:

    python task_2_machine_learning_model.py

A confusion-matrix image is generated in task_2_outputs/.

## Project structure

    ai-ml-internship/
    ├── README.md
    ├── requirements.txt
    ├── task_1_data_exploration.py
    └── task_2_machine_learning_model.py

### Task 1 Visualizations

Task 1 produces:

- Survival count
- Age distribution
- Survival by passenger class
- Gender vs survival
- Fare distribution