PredictMaint AI — Project Summary

Project Overview

PredictMaint AI is an explainable machine learning prototype designed to support predictive maintenance for industrial equipment. The project was developed for the ABB Accelerator 2026 Predictive Maintenance challenge and demonstrates how machine operating data can be transformed into actionable failure-risk insights.

Unexpected equipment failures can result in production downtime, increased maintenance costs, and operational disruption. Traditional reactive maintenance addresses equipment after a problem occurs, while preventive maintenance can result in unnecessary inspections or component replacement. PredictMaint AI explores a data-driven approach in which machine learning identifies equipment with elevated failure risk and provides information that can help maintenance teams prioritize their attention.

Solution

PredictMaint AI provides an interactive Streamlit application that guides the user through an end-to-end predictive maintenance workflow.

Users can upload machine operating data, inspect dataset quality, configure model features, train multiple machine learning models, compare their performance, select an operational maintenance objective, and receive a recommended model.

The trained model can then evaluate the operating conditions of an individual machine and estimate its probability of failure. The application converts this probability into LOW, MEDIUM, or HIGH risk classifications to make the output easier to interpret.

Explainable AI is incorporated throughout the system so that users can understand not only the predicted risk but also the factors influencing the model’s decisions.

Data and Machine Learning Approach

The prototype uses the AI4I 2020 Predictive Maintenance Dataset, which contains 10,000 machine observations.

Six primary operating characteristics are used for prediction:

* Product type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

The target variable is machine failure.

Machine failures account for only approximately 3.4% of the dataset, creating a significant class-imbalance problem. PredictMaint AI therefore evaluates models using accuracy, precision, recall, F1 score, ROC-AUC, and precision-recall AUC rather than relying on accuracy alone.

Three classification algorithms are trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

The system also applies imbalance-aware training techniques, including class weighting and 'scale_pos_weight'.

Operational Model Selection

A central design goal of PredictMaint AI is to connect model selection with maintenance priorities.

Rather than declaring one algorithm universally best, the application allows the user to select an operational objective. For example, one maintenance environment may prioritize detecting as many potential failures as possible, while another may prioritize reducing unnecessary maintenance alerts.

Under the balanced-performance objective used in the prototype demonstration, Random Forest is recommended based on F1 score.

On the held-out 2,000-observation test set, Random Forest achieved:

* 97.90% accuracy
* 90.62% precision
* 42.65% recall
* 0.5800 F1 score
* 0.9636 ROC-AUC
* 0.7720 precision-recall AUC

Its confusion matrix contained 1,929 true negatives, 3 false positives, 39 false negatives, and 29 true positives.

These results illustrate the tradeoffs involved in predictive maintenance. Random Forest produces very few false alarms under this configuration, while other models such as XGBoost provide substantially higher failure recall and may therefore be preferable when failure detection is the primary objective.

Explainable AI

PredictMaint AI incorporates both global and local model explainability.

Global feature importance identifies which operating characteristics influence the recommended model most strongly across the dataset. In the current Random Forest model, rotational speed, torque, and tool wear are among the most influential features.

For individual machine predictions, SHAP is used to estimate how each operating condition contributes toward or away from predicted failure risk. This allows users to understand why a particular machine received its risk estimate rather than treating the machine learning model as a black box.

Industrial Potential

PredictMaint AI demonstrates how machine learning could support a more proactive maintenance workflow by combining sensor data, failure-risk estimation, operational model selection, and explainability.

The current system is a prototype based on a historical dataset rather than a production industrial deployment. Future development could incorporate real-time IoT sensor streams, time-series failure forecasting, equipment-specific models, automated maintenance alerts, model-drift monitoring, automated retraining, and cloud deployment.

The long-term goal is to provide maintenance teams with an interpretable decision-support system that helps identify equipment requiring attention before unexpected failures disrupt operations.