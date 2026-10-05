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

### Project structure

```
ai-ml-internship/
├── README.md
├── requirements.txt
└── task_1_data_exploration.py
```

### Run the task

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python task_1_data_exploration.py
```

The script will:

- Load the dataset with Pandas.
- Display shape, columns, data types, summary statistics, and missing values.
- Remove duplicate rows.
- Validate and correct invalid values.
- Fill missing numerical and categorical values using appropriate strategies.
- Print a before/after cleaning summary.
- Generate and save basic visualizations in a `plots/` directory.

### Visualizations

The analysis produces:

- Survival count
- Age distribution
- Survival by passenger class
- Gender vs survival
- Fare distribution

This provides a basic understanding of the dataset before any machine-learning model is built.
