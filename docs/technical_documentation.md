PredictMaint AI — Technical Documentation

1. System Overview

PredictMaint AI is an explainable machine learning prototype designed to demonstrate an end-to-end predictive maintenance workflow for industrial equipment.

The application accepts historical machine operating data, validates the dataset, trains multiple classification models, compares their performance, recommends a model according to an operational maintenance objective, and estimates failure risk for individual machines.

The system also incorporates explainable AI techniques so users can understand both overall model behavior and the factors contributing to an individual failure-risk prediction.

The prototype was developed for the ABB Accelerator 2026 Predictive Maintenance challenge.

⸻

2. System Architecture

PredictMaint AI follows a modular machine learning architecture:

Industrial Dataset
        │
        ▼
Dataset Validation
        │
        ▼
Feature Configuration
        │
        ▼
Data Preprocessing
        │
        ▼
Model Training
   ┌────┼────┐
   ▼    ▼    ▼
Logistic  Random   XGBoost
Regression Forest
   └────┬────┘
        ▼
Model Evaluation
        │
        ▼
Operational Model Recommendation
        │
        ├──────────────► Global Feature Importance
        │
        ▼
Machine Sensor Inputs
        │
        ▼
Failure Probability
        │
        ▼
Risk Classification
        │
        ▼
SHAP Prediction Explanation

The Streamlit application provides the user interface and coordinates the individual data-processing, training, prediction, and explainability modules.

3. Dataset

The prototype uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains 10,000 machine observations and 14 columns representing identifiers, operating conditions, machine failure status, and individual failure modes.

The primary prediction target is:

Machine failure

The six default predictive features are:

* Type
* Air temperature [K]
* Process temperature [K]
* Rotational speed [rpm]
* Torque [Nm]
* Tool wear [min]

The application excludes the following fields from the default training features:

* UDI
* Product ID
* TWF
* HDF
* PWF
* OSF
* RNF

UDI and Product ID primarily function as identifiers rather than operating characteristics.

The individual failure-mode indicators are excluded because they directly describe known failure conditions and could introduce target leakage when predicting the overall Machine failure target.

⸻

4. Dataset Validation

Before model training, the Streamlit interface performs several validation checks.

The application displays:

* Number of rows
* Number of columns
* Missing-value count
* Duplicate-row count
* Target class distribution
* Selected training features

The training workflow prevents model training when:

* No features are selected
* The target contains missing values
* Selected training features contain missing values
* Required prediction features are unavailable
* The selected target is not binary and encoded as 0 and 1

These checks reduce the likelihood of invalid data reaching the machine learning pipeline.

⸻

5. Class Imbalance

Machine failure is a rare event within the AI4I dataset.

Approximately 3.4% of observations represent machine failures, while approximately 96.6% represent normal operation.

Because of this imbalance, accuracy alone can provide a misleading representation of model quality. A model predicting nearly every machine as healthy could achieve high accuracy while failing to identify actual equipment failures.

PredictMaint AI therefore evaluates several classification metrics and applies imbalance-aware training strategies.

Logistic Regression and Random Forest use class weighting.

XGBoost uses scale_pos_weight to increase the influence of minority-class observations during training.

⸻

6. Data Preprocessing

Model preprocessing is implemented using Scikit-learn pipelines and a ColumnTransformer.

Numerical features are standardized using StandardScaler.

The numerical variables are:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

The categorical Type feature is transformed using one-hot encoding.

Keeping preprocessing inside the model pipeline ensures that the same transformations applied during training are also applied when predicting new machine observations.

⸻

7. Train/Test Split

The dataset is divided into training and testing subsets using an 80/20 split.

Stratification is applied using the machine failure target so that approximately the same failure rate is maintained in both subsets.

For the AI4I dataset:

* Training observations: 8,000
* Testing observations: 2,000
* Training failure rate: approximately 3.39%
* Testing failure rate: approximately 3.40%

This provides a consistent evaluation set while preserving the original class imbalance.

⸻

8. Machine Learning Models

PredictMaint AI trains three classification algorithms.

Logistic Regression

Logistic Regression provides a linear baseline and uses class weighting to account for the minority failure class.

Random Forest

Random Forest combines multiple decision trees and can capture nonlinear relationships between machine operating conditions and failure outcomes.

Class weighting is used to increase sensitivity to failure observations.

XGBoost

XGBoost uses gradient-boosted decision trees and can model complex nonlinear relationships.

The model uses scale_pos_weight to compensate for class imbalance.

⸻

9. Model Evaluation

Each trained model is evaluated on the held-out test set using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

Precision measures how many machines predicted to fail actually failed.

Recall measures how many actual machine failures were successfully detected.

F1 combines precision and recall into a single metric.

ROC-AUC evaluates the model’s ability to distinguish between the two classes across classification thresholds.

PR-AUC summarizes the precision-recall relationship and is particularly useful when evaluating an imbalanced classification problem.

⸻

10. Operational Model Recommendation

Rather than selecting a model based only on accuracy, PredictMaint AI allows model recommendation to reflect a maintenance objective.

For example, a balanced-performance objective emphasizes F1 score, providing a compromise between precision and recall.

Other maintenance priorities can place greater emphasis on identifying potential failures or reducing unnecessary maintenance alerts.

This design demonstrates how machine learning model selection can be connected to operational priorities rather than relying on a single universal metric.

⸻

11. Current Prototype Results

Using the AI4I dataset and the balanced-performance objective, Random Forest is recommended by the current prototype.

On the 2,000-observation test set, the Random Forest confusion matrix is:

Result                      Count

True Negatives              1,915

False Positives             17

False Negatives             25

True Positives              43

These results indicate that the model correctly identifies most normal operating observations while detecting a portion of the relatively small failure class.

Because failures are rare, the application presents the confusion matrix alongside precision, recall, F1, ROC-AUC, and PR-AUC rather than relying on accuracy alone.

⸻

12. Global Explainability

PredictMaint AI provides global feature importance for the recommended model.

Global feature importance describes which transformed model features contribute most strongly to the model’s overall decision process.

For tree-based models such as Random Forest and XGBoost, feature importance provides a high-level view of the operating characteristics most influential across the trained model.

This information can help users understand which sensor measurements the model relies on most heavily.

⸻

13. Machine Failure Risk Prediction

After model training, users can enter current machine operating conditions through the Streamlit interface.

The prediction interface accepts:

* Product Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

The trained preprocessing and model pipeline then calculates a probability associated with machine failure.

The application displays this probability as a model-estimated failure risk.

⸻

14. Risk Classification

The predicted failure probability is converted into an operational risk level:

* LOW
* MEDIUM
* HIGH

The risk classification provides a simpler interpretation of the raw probability.

The interface also provides contextual guidance based on the risk level. For example, machines receiving elevated risk estimates can be prioritized for inspection or closer monitoring.

These classifications are prototype decision-support outputs and should not be interpreted as production maintenance thresholds without additional industrial validation.

⸻

15. Individual Prediction Explainability

PredictMaint AI uses SHAP to explain individual machine predictions.

For a selected machine observation, the application estimates the contribution of each feature toward or away from predicted machine failure.

Positive SHAP contributions push the prediction toward greater failure risk.

Negative SHAP contributions push the prediction away from failure risk.

The Streamlit interface presents these contributions visually and in tabular form so users can identify which machine conditions had the greatest influence on a particular prediction.

This provides local explainability in addition to the global feature-importance analysis.

⸻

16. Software Structure

The application is separated into several modules:

predictmaint-ai/
├── data/
│   └── ai4i2020.csv
├── docs/
│   └── technical_documentation.md
├── src/
│   ├── data_analysis.py
│   ├── explainability.py
│   ├── prediction.py
│   └── train_models.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt

app.py

Provides the Streamlit user interface and coordinates dataset upload, validation, model training, model recommendation, visualization, prediction, and explanation.

src/data_analysis.py

Contains dataset inspection and analysis functionality.

src/train_models.py

Handles preprocessing, model training, evaluation, leaderboard creation, and model comparison.

src/prediction.py

Handles machine failure probability estimation and risk classification.

src/explainability.py

Provides global feature importance and SHAP-based individual prediction explanations.

⸻

17. Technology Stack

The prototype uses:

* Python 3.12
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Altair
* Matplotlib

The project is designed to run locally using a Python virtual environment.

⸻

18. Error Handling and Input Validation

The application includes safeguards for several common failure conditions.

Model training validates selected features, missing values, required sensor variables, and target encoding before training begins.

Prediction inputs are constrained using observed dataset ranges.

Prediction execution is wrapped in error handling so invalid prediction operations produce a user-facing message rather than exposing an application traceback.

These protections improve prototype reliability and provide clearer feedback to users.

⸻

19. Current Limitations

PredictMaint AI is an experimental prototype and is not intended to replace an industrial predictive maintenance platform.

Current limitations include:

* The workflow is designed around the AI4I 2020 dataset schema.
* Evaluation currently uses a single train/test split.
* The dataset is synthetic and does not represent every industrial environment.
* Risk thresholds are prototype decision-support thresholds rather than validated maintenance policies.
* Sensor input ranges do not guarantee that every combination represents a physically valid machine state.
* The application does not currently consume live equipment telemetry.
* The system does not currently monitor model or data drift.
* Models are retrained manually through the application.
* Additional real-world validation would be required before production deployment.

⸻

20. Future Development

Potential future improvements include:

* Live industrial IoT sensor ingestion
* Time-series equipment monitoring
* Failure forecasting over future time windows
* Cost-sensitive classification thresholds
* Maintenance-cost optimization
* Model and data drift detection
* Automated retraining pipelines
* Cloud deployment
* Equipment-specific models
* Automated maintenance notifications
* Integration with maintenance-management systems

⸻

21. Conclusion

PredictMaint AI demonstrates an end-to-end explainable predictive maintenance workflow.

The prototype combines dataset validation, imbalance-aware machine learning, automated model comparison, operational model recommendation, machine-level failure-risk estimation, and explainable AI within a single interactive application.

The system is intended to demonstrate how machine learning can support more informed and proactive industrial maintenance decisions while keeping model behavior visible to the user.