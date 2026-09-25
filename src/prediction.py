import pandas as pd


def predict_failure(
    model,
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """Predict machine failure risk from sensor readings."""

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

    failure_probability = model.predict_proba(machine_data)[0][1]

    if failure_probability >= 0.70:
        risk_level = "HIGH"
    elif failure_probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "failure_probability": failure_probability,
        "risk_level": risk_level,
    }