import pandas as pd
import streamlit as st

from src.train_models import (
    prepare_training_data,
    split_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    create_leaderboard,
    recommend_model,
)

from src.explainability import (
    get_feature_importance,
    explain_prediction,
)
from src.prediction import predict_failure

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None

if "training_config" not in st.session_state:
    st.session_state.training_config = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None

st.set_page_config(
    page_title="PredictMaint AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ PredictMaint AI")
st.subheader("Agentic Predictive Maintenance Studio")

st.write(
    "Analyze industrial equipment data, compare machine learning models, "
    "and estimate machine failure risk."
)

st.divider()

st.header("Industrial Dataset")

uploaded_file = st.file_uploader(
    "Upload a machine sensor dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin analysis.")

else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error(
            "The uploaded file could not be read as a valid CSV. "
            "Please upload a properly formatted CSV file."
        )
        st.stop()

    if df.empty:
        st.error(
            "The uploaded dataset is empty. "
            "Please upload a CSV containing machine data."
        )
        st.stop()

    required_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    st.success("Dataset uploaded successfully.")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True,
    )

    st.subheader("Dataset Health")

    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{total_rows:,}")
    col2.metric("Columns", total_columns)
    col3.metric("Missing Values", f"{missing_values:,}")
    col4.metric("Duplicate Rows", f"{duplicate_rows:,}")

    st.subheader("Prediction Target")

    target_column = st.selectbox(
        "Select the column you want to predict",
        options=df.columns,
        index=df.columns.get_loc("Machine failure")
        if "Machine failure" in df.columns
        else 0,
    )

    target_counts = df[target_column].value_counts()
    target_percentages = df[target_column].value_counts(normalize=True) * 100

    target_summary = pd.DataFrame(
        {
            "Count": target_counts,
            "Percentage": target_percentages,
        }
    )

    st.dataframe(
        target_summary,
        use_container_width=True,
    )

    if len(target_percentages) == 2:
        minority_percentage = target_percentages.min()

        if minority_percentage < 10:
            st.warning(
                f"Class imbalance detected: the minority class represents "
                f"only {minority_percentage:.2f}% of the dataset. "
                "Accuracy alone may be misleading for this prediction task."
            )
        else:
            st.success(
                "The target classes are reasonably balanced."
            )

    st.subheader("Feature Configuration")

    available_features = [
        column
        for column in df.columns
        if column != target_column
    ]

    default_excluded_columns = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    default_features = [
        column
        for column in available_features
        if column not in default_excluded_columns
    ]

    required_prediction_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    selected_features = st.multiselect(
        "Select the features to use for model training",
        options=available_features,
        default=default_features,
    )

    st.write(
        f"**{len(selected_features)} features selected for training.**"
    )

    excluded_columns = [
        column
        for column in default_excluded_columns
        if column in available_features
    ]

    if excluded_columns:
        st.info(
            "Recommended exclusions: "
            + ", ".join(excluded_columns)
            + ". Identifier columns and known failure-mode indicators "
            "are excluded by default to reduce irrelevant features "
            "and potential target leakage."
        )
    st.divider()

    st.header("AutoML Training")

    st.write(
        "Choose the operational objective that should guide "
        "the model recommendation."
    )

    objective_label = st.selectbox(
        "Maintenance objective",
        options=[
            "Balanced performance",
            "Prioritize failure detection",
            "Minimize false alarms",
        ],
    )

    objective_mapping = {
        "Balanced performance": "balanced",
        "Prioritize failure detection": "failure_detection",
        "Minimize false alarms": "minimize_false_alarms",
    }

    objective = objective_mapping[objective_label]

    train_button = st.button(
        "Train Models",
        type="primary",
    )

    if train_button:
        if not selected_features:
            st.error(
            "Select at least one feature before training."
        )

        elif df[target_column].isnull().any():
            missing_target_values = df[target_column].isnull().sum()

            st.error(
            f"The selected target contains "
            f"{missing_target_values} missing value(s). "
            "Remove or fill the missing target values before training."
        )

        elif df[selected_features].isnull().any().any():
            missing_feature_values = (
                df[selected_features].isnull().sum().sum()
            )

            st.error(
                f"The selected training features contain "
                f"{missing_feature_values} missing value(s). "
                "Remove or fill the missing feature values before training."
            )

        elif not all(
            feature in selected_features
            for feature in required_prediction_features
        ):
            st.error(
            "The current prediction dashboard requires all six "
            "machine sensor features. Restore the recommended "
            "feature selection before training."
        )

        elif set(df[target_column].dropna().unique()) != {0, 1}:
            st.error(
                "The current prototype requires a binary target "
                "encoded as 0 and 1."
            )

        else:
            st.info("Training models...")

            X, y, numeric_features, categorical_features = prepare_training_data(
                df,
                target_column,
                selected_features,
            )

            X_train, X_test, y_train, y_test = split_data(
                X,
                y,
            )

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

            st.session_state.trained_models = trained_models
            st.session_state.leaderboard = leaderboard
            st.session_state.model_results = results
            st.session_state.training_config = {
                "target": target_column,
                "features": selected_features.copy(),
            }

    current_config = {
        "target": target_column,
        "features": selected_features.copy(),
    }

    config_matches = (
        st.session_state.training_config == current_config
    )

    if (
        st.session_state.trained_models is not None
        and config_matches
    ):
        trained_models = st.session_state.trained_models
        leaderboard = st.session_state.leaderboard

        recommendation = recommend_model(
            leaderboard,
            objective=objective,
        )

        recommended_model_name = recommendation["model"]
        recommended_model = trained_models[recommended_model_name]

        recommendation_details = {
            "balanced": {
                "metric": "f1",
                "label": "F1 score",
                "reason": (
                    "it provides the strongest balance between "
                    "precision and recall"
                ),
            },
            "failure_detection": {
                "metric": "recall",
                "label": "recall",
                "reason": (
                    "it detects the highest proportion of actual "
                    "machine failures"
                ),
            },
            "minimize_false_alarms": {
                "metric": "precision",
                "label": "precision",
                "reason": (
                    "it provides the highest precision, helping reduce "
                    "unnecessary maintenance alerts"
                ),
            },
        }

        detail = recommendation_details[objective]
        recommendation_metric = detail["metric"]

        st.success("Trained models are ready for the current configuration.")

        st.subheader("Model Leaderboard")

        st.dataframe(
            leaderboard,
            use_container_width=True,
        )

        st.subheader("Recommended Model")

        st.success(
            f"{recommendation['model']} is recommended because "
            f"{detail['reason']}, with a "
            f"{detail['label']} of "
            f"{recommendation[recommendation_metric]:.4f}."
        )

        importance_df = get_feature_importance(
            recommended_model
        )

        importance_df["feature"] = (
            importance_df["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        st.subheader("Confusion Matrix")

        model_result = next(
            result
            for result in st.session_state.model_results
            if result["model"] == recommended_model_name
        )

        matrix = model_result["confusion_matrix"]

        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual: No Failure", "Actual: Failure"],
            columns=["Predicted: No Failure", "Predicted: Failure"],
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
        )

        true_negatives = matrix[0, 0]
        false_positives = matrix[0, 1]
        false_negatives = matrix[1, 0]
        true_positives = matrix[1, 1]

        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        cm_col1.metric(
            "True Negatives",
            f"{true_negatives:,}",
            help="Machines correctly predicted as not failing.",
        )

        cm_col2.metric(
            "False Positives",
            f"{false_positives:,}",
            help="Machines incorrectly predicted to fail.",
        )

        cm_col3.metric(
            "False Negatives",
            f"{false_negatives:,}",
            help="Actual failures that the model missed.",
        )

        cm_col4.metric(
            "True Positives",
            f"{true_positives:,}",
            help="Machine failures correctly detected.",
        )

        st.subheader("Explainable AI")

        st.write(
            f"Feature importance for the recommended "
            f"{recommended_model_name} model."
        )

        st.bar_chart(
            importance_df.set_index("feature")["importance"]
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
        )

        st.divider()

        st.header("Machine Failure Risk Prediction")

        st.write(
            "Enter current machine sensor readings to estimate "
            "the risk of machine failure."
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            product_type = st.selectbox(
                "Product Type",
                options=["L", "M", "H"],
            )

            air_temperature = st.number_input(
                "Air Temperature [K]",
                min_value=float(df["Air temperature [K]"].min()),
                max_value=float(df["Air temperature [K]"].max()),
                value=300.0,
                step=0.1,
            )

            process_temperature = st.number_input(
                "Process Temperature [K]",
                min_value=float(df["Process temperature [K]"].min()),
                max_value=float(df["Process temperature [K]"].max()),
                value=310.0,
                step=0.1,
            )

        with input_col2:
            rotational_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=int(df["Rotational speed [rpm]"].min()),
                max_value=int(df["Rotational speed [rpm]"].max()),
                value=1400,
                step=10,
            )

            torque = st.number_input(
                "Torque [Nm]",
                min_value=float(df["Torque [Nm]"].min()),
                max_value=float(df["Torque [Nm]"].max()),
                value=50.0,
                step=0.1,
            )

            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=int(df["Tool wear [min]"].min()),
                max_value=int(df["Tool wear [min]"].max()),
                value=180,
                step=1,
            )

        predict_button = st.button(
            "Predict Failure Risk",
            type="primary",
        )

        if predict_button:
            try:
                prediction = predict_failure(
                recommended_model,
                product_type=product_type,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
            )

            except Exception:
                st.error(
                "The prediction could not be completed. "
                "Please verify the machine sensor values and try again."
                )
                st.stop()

            failure_probability = prediction["failure_probability"]
            risk_level = prediction["risk_level"]

            st.subheader("Prediction Result")

            st.metric(
                "Model-Estimated Failure Risk",
                f"{failure_probability * 100:.2f}%",
            )

            if risk_level == "HIGH":
                st.error(
                    f"Risk Level: {risk_level}"
                )
            elif risk_level == "MEDIUM":
                st.warning(
                    f"Risk Level: {risk_level}"
                )
            else:
                st.success(
                    f"Risk Level: {risk_level}"
                )

            st.caption(
                "This estimate is based on the selected model and the "
                "sensor readings entered above."
            )

            if risk_level == "HIGH":
                st.write(
                "The model indicates elevated failure risk. "
                "The machine should be prioritized for inspection."
            )

            elif risk_level == "MEDIUM":
                st.write(
                "The model indicates moderate failure risk. "
                "Consider monitoring the machine more closely."
            )

            else:
                st.write(
                "The model indicates low failure risk based on "
                "the current sensor readings."
            )
            
            machine_data = pd.DataFrame(
                [
                    {
                        "Type": product_type,
                        "Air temperature [K]": air_temperature,
                        "Process temperature [K]": process_temperature,
                        "Rotational speed [rpm]": rotational_speed,
                        "Torque [Nm]": torque,
                        "Tool wear [min]": tool_wear,
                    }
                ]
            )

            local_explanation = explain_prediction(
                recommended_model,
                machine_data,
            )

            st.subheader("Why This Prediction?")

            st.write(
                "Features with positive contributions push the prediction toward "
                "machine failure, while negative contributions push it away from failure."
            )

            chart_data = local_explanation[
                [
                    "feature",
                    "contribution",
                    "impact",
                ]
            ].copy()

            st.vega_lite_chart(
                chart_data,
                {
                    "mark": {
                        "type": "bar",
                        "tooltip": True,
                    },
                    "encoding": {
                        "y": {
                            "field": "feature",
                            "type": "nominal",
                            "sort": "-x",
                            "title": None,
                            "axis": {
                                "labelLimit": 220,
                            },
                        },
                        "x": {
                            "field": "contribution",
                            "type": "quantitative",
                            "title": "SHAP Contribution",
                        },
                        "color": {
                            "field": "impact",
                            "type": "nominal",
                            "title": "Impact",
                        },
                    },
                },
                use_container_width=True,
            )

            st.dataframe(
                local_explanation[
                    [
                        "feature",
                        "contribution",
                        "impact",
                    ]
                ],
                use_container_width=True,
            )

        