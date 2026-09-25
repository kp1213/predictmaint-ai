# ⚙️ PredictMaint AI

PredictMaint AI is an explainable machine learning application for industrial predictive maintenance. The system analyzes machine sensor data, compares multiple classification models, recommends a model based on the user's maintenance objective, and estimates the probability of machine failure.

The project was developed for the **ABB Accelerator 2026 – Predictive Maintenance** challenge.

## Overview

Unexpected equipment failures can cause production downtime, maintenance costs, and operational disruption. PredictMaint AI demonstrates how machine learning can use historical operating conditions to identify equipment with elevated failure risk before a failure occurs.

The application allows users to:

- Upload industrial machine data
- Inspect dataset health and class distribution
- Configure model features
- Train and compare multiple machine learning models
- Select a maintenance objective
- Receive an automatically recommended model
- Review classification metrics and a confusion matrix
- Examine global feature importance
- Estimate failure risk for an individual machine
- Explain individual predictions using SHAP

## Dataset

The prototype uses the **AI4I 2020 Predictive Maintenance Dataset**, containing 10,000 machine observations.

The prediction target is `Machine failure`.

The primary model features are:

- Product Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

Identifier fields and individual failure-mode indicators are excluded from the default model features to reduce irrelevant information and potential target leakage.

The dataset is highly imbalanced, with machine failures representing approximately 3.4% of observations. Because of this, the project evaluates models using metrics beyond accuracy.

## Machine Learning Models

PredictMaint AI trains and compares three classification models:

1. Logistic Regression
2. Random Forest
3. XGBoost

Preprocessing is performed using a Scikit-learn pipeline. Numeric variables are standardized and categorical variables are one-hot encoded.

Class imbalance is addressed using class weighting for Logistic Regression and Random Forest and `scale_pos_weight` for XGBoost.
## Model Evaluation

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Because machine failures are rare in the dataset, precision, recall, F1, and PR-AUC provide more useful information than accuracy alone.

The application supports different maintenance objectives so the recommended model can reflect operational priorities such as balanced performance or identifying as many potential failures as possible.

## Explainable AI

PredictMaint AI includes explainability at both the global and individual prediction levels.

Global feature importance shows which machine characteristics have the greatest influence on the recommended model overall.

For individual predictions, SHAP is used to estimate how each sensor reading contributes to the machine's predicted failure risk. Positive contributions push the prediction toward failure, while negative contributions push it away from failure.

## Application Features

The Streamlit dashboard provides:

- CSV dataset upload and validation
- Dataset preview and health statistics
- Missing-value detection
- Class imbalance detection
- Feature configuration
- Automated model training
- Model leaderboard
- Operational model recommendation
- Confusion matrix analysis
- Global feature importance
- Interactive machine sensor inputs
- Machine failure probability estimation
- LOW, MEDIUM, and HIGH risk classifications
- SHAP-based prediction explanations
## Project Structure

```text
predictmaint-ai/
├── data/
│   └── ai4i2020.csv
├── docs/
├── models/
├── src/
│   ├── data_analysis.py
│   ├── explainability.py
│   ├── prediction.py
│   └── train_models.py
├── tests/
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

The `.venv/` and `__pycache__/` directories are generated locally during development and are excluded from version control through `.gitignore`.

## Installation

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit address displayed in the terminal.

Upload `data/ai4i2020.csv` through the application, select the maintenance objective, and train the models.

The dashboard will display model performance, the recommended model, confusion matrix results, global feature importance, and the machine failure risk prediction interface.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Altair
- Matplotlib

## Limitations

PredictMaint AI is a prototype and is not intended to replace a production industrial maintenance system.

Current limitations include:

- The application is designed around the AI4I 2020 predictive maintenance dataset schema.
- Model performance is evaluated using a train/test split of the available dataset.
- Predictions depend on patterns represented in the training data.
- Sensor input validation does not guarantee that every combination represents a physically valid machine state.
- The application does not currently process live industrial sensor streams.
- Additional validation using real-world industrial equipment would be required before deployment.

## Future Improvements

Future development could include:

- Real-time IoT sensor integration
- Time-series failure forecasting
- Cost-sensitive prediction thresholds
- Model and data drift monitoring
- Automated model retraining
- Support for additional industrial datasets
- Cloud deployment
- Automated maintenance alerts

## Purpose

PredictMaint AI demonstrates an end-to-end predictive maintenance workflow combining machine learning, model comparison, operational model selection, failure risk estimation, and explainable AI.

It was developed as a prototype for the ABB Accelerator 2026 predictive maintenance challenge.
