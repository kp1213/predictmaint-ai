import pandas as pd
import shap

def get_feature_importance(model):
    """Extract global feature importance from a trained model pipeline."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])

    else:
        raise ValueError(
            f"Feature importance is not supported for "
            f"{classifier.__class__.__name__}."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def display_feature_importance(model):
    """Display ranked global feature importance."""

    importance_df = get_feature_importance(model)

    print("\n--- Explainable AI: Feature Importance ---")

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "importance": "{:.4f}".format,
            },
        )
    )

    return importance_df


def explain_prediction(model, machine_data):
    """Explain an individual machine failure prediction using SHAP."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_data = preprocessor.transform(machine_data)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    explainer = shap.Explainer(classifier)
    shap_values = explainer(transformed_data)

    if shap_values.values.ndim == 3:
        contribution_values = shap_values.values[0, :, 1]
    else:
        contribution_values = shap_values.values[0]

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contribution_values,
        }
    )

    explanation_df["feature"] = (
        explanation_df["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    type_mask = explanation_df["feature"].str.startswith("Type_")

    if type_mask.any():
        type_contribution = explanation_df.loc[
            type_mask,
            "contribution",
        ].sum()

        explanation_df = explanation_df.loc[
            ~type_mask
        ].copy()

        type_row = pd.DataFrame(
            {
                "feature": [
                    f"Product Type ({machine_data['Type'].iloc[0]})"
                ],
                "contribution": [type_contribution],
            }
        )

        explanation_df = pd.concat(
            [
                explanation_df,
                type_row,
            ],
            ignore_index=True,
        )

    explanation_df["impact"] = explanation_df["contribution"].apply(
        lambda value: (
            "Increases failure risk"
            if value > 0
            else "Decreases failure risk"
        )
    )

    explanation_df["absolute_contribution"] = (
        explanation_df["contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        by="absolute_contribution",
        ascending=False,
    ).reset_index(drop=True)

    return explanation_df