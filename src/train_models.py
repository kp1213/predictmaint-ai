import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


DATA_PATH = "data/ai4i2020.csv"
TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type",
]

def load_training_data():
    """Load the dataset and separate features from the prediction target."""

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y

def prepare_training_data(
    df,
    target_column,
    feature_columns,
):
    """Prepare selected features and automatically identify feature types."""

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return X, y, numeric_features, categorical_features

def split_data(X, y):
    """Split data while preserving the failure-class distribution."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

def create_preprocessor(
    numeric_features,
    categorical_features,
):
    """Create preprocessing for numeric and categorical features."""

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def train_logistic_regression(
    X_train, 
    y_train,
    numeric_features,
    categorical_features,
):
    """Train a logistic regression baseline model."""

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a trained classification model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix,
    }

def create_leaderboard(results):
    """Create a ranked comparison of trained models."""

    leaderboard = pd.DataFrame(results)

    if "confusion_matrix" in leaderboard.columns:
        leaderboard = leaderboard.drop(
            columns=["confusion_matrix"]
        )

    leaderboard = leaderboard.sort_values(
        by="f1",
        ascending=False,
    ).reset_index(drop=True)

    print("\n--- AutoML Model Leaderboard ---")
    print(
        leaderboard.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
                "pr_auc": "{:.4f}".format,
            },
        )
    )

    return leaderboard

def recommend_model(leaderboard, objective="balanced"):
    """Recommend a model based on the user's operational objective."""

    if objective == "balanced":
        metric = "f1"
        reason = "best balance between precision and recall"

    elif objective == "failure_detection":
        metric = "recall"
        reason = "highest ability to detect actual machine failures"

    elif objective == "minimize_false_alarms":
        metric = "precision"
        reason = "highest precision, reducing unnecessary maintenance alerts"

    else:
        raise ValueError(
            "Objective must be 'balanced', "
            "'failure_detection', or 'minimize_false_alarms'."
        )

    best_index = leaderboard[metric].idxmax()
    best_model = leaderboard.loc[best_index]

    print("\n--- AutoML Recommendation ---")
    print(f"Objective: {objective}")
    print(f"Recommended model: {best_model['model']}")
    print(
        f"Reason: {reason} "
        f"({metric} = {best_model[metric]:.4f})"
    )

    return best_model

def train_random_forest(
    X_train, 
    y_train,
    numeric_features,
    categorical_features,
):
    """Train a Random Forest classification model."""

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model

def train_xgboost(
    X_train, 
    y_train,
    numeric_features,
    categorical_features,
):
    """Train an XGBoost classification model."""

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model

if __name__ == "__main__":
    
    from prediction import predict_failure
    from explainability import (
        display_feature_importance,
        explain_prediction,
    )
    
    df = pd.read_csv(DATA_PATH)

    X, y, numeric_features, categorical_features = prepare_training_data(
    df,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    )

    print("\n--- Automatic Feature Detection ---")

    print(f"Numeric features: {numeric_features}")

    print(f"Categorical features: {categorical_features}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train,
        y_train,
        numeric_features,
        categorical_features,
    )

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
    )

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train,
        y_train,
        numeric_features,
        categorical_features,
    )

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test,
    )

    print("\nTraining XGBoost...")

    xgboost_model = train_xgboost(
        X_train,
        y_train,
        numeric_features,
        categorical_features,    
    )

    xgboost_results = evaluate_model(
        "XGBoost",
        xgboost_model,
        X_test,
        y_test,
    )

    trained_models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model,
    }

    results = [
        logistic_results,
        random_forest_results,
        xgboost_results,
    ]

    leaderboard = create_leaderboard(results)

    recommendation = recommend_model(
        leaderboard,
        objective="balanced",
    )

    recommended_model_name = recommendation["model"]
    recommended_model = trained_models[recommended_model_name]

    display_feature_importance(recommended_model)

    prediction = predict_failure(
        recommended_model,
        product_type="L",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1400,
        torque=50.0,
        tool_wear=180,
    )

    print("\n--- Machine Failure Prediction ---")
    print(
        f"Failure probability: "
        f"{prediction['failure_probability'] * 100:.2f}%"
    )
    print(f"Risk level: {prediction['risk_level']}")

    test_machine = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 298.9,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1410,
                "Torque [Nm]": 65.7,
                "Tool wear [min]": 191,
            }
        ]
    )

    explanation_df = explain_prediction(
        recommended_model,
        test_machine,
    )

    print("\n--- Local SHAP Explanation ---")
    print(f"Recommended model: {recommended_model_name}")
    print(explanation_df.to_string(index=False))
