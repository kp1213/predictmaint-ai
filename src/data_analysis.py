import pandas as pd

DATA_PATH = "data/ai4i2020.csv"


def load_dataset():
    """Load the predictive maintenance dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

def profile_dataset(df):
    """Generate a basic quality profile for a dataset."""

    print("\n--- Dataset Profile ---")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")

    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    print("\nData types:")
    print(df.dtypes)

def identify_feature_types(df):
    """Identify numeric and categorical columns."""

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    print("\n--- Feature Types ---")

    print("\nNumeric columns:")
    for column in numeric_columns:
        print(f"- {column}")

    print("\nCategorical columns:")
    for column in categorical_columns:
        print(f"- {column}")

    return numeric_columns, categorical_columns

def analyze_target(df, target_column):
    """Analyze the distribution of the selected prediction target."""

    print(f"\n--- Target Analysis: {target_column} ---")

    counts = df[target_column].value_counts()
    percentages = df[target_column].value_counts(normalize=True) * 100

    for value in counts.index:
        print(
            f"Class {value}: {counts[value]} records "
            f"({percentages[value]:.2f}%)"
        )

    minority_percentage = percentages.min()

    if minority_percentage < 10:
        print(
            f"\nWARNING: Class imbalance detected. "
            f"The minority class represents only "
            f"{minority_percentage:.2f}% of the dataset."
        )

def select_features(df, target_column, id_columns=None, leakage_columns=None):
    """Select usable model features while excluding IDs and leakage columns."""

    id_columns = id_columns or []
    leakage_columns = leakage_columns or []

    excluded_columns = [target_column] + id_columns + leakage_columns

    feature_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    print("\n--- Feature Selection ---")

    if id_columns:
        print("\nExcluded identifier columns:")
        for column in id_columns:
            print(f"- {column}")

    if leakage_columns:
        print("\nExcluded potential target leakage columns:")
        for column in leakage_columns:
            print(f"- {column}")

    print("\nSelected model features:")
    for column in feature_columns:
        print(f"- {column}")

    return feature_columns


if __name__ == "__main__":
    data = load_dataset()

    print("Dataset loaded successfully!")

    profile_dataset(data)

    numeric_columns, categorical_columns = identify_feature_types(data)

    analyze_target(data, "Machine failure")

    feature_columns = select_features(
        data,
        target_column = "Machine failure",
        id_columns=["UDI", "Product ID"],
        leakage_columns=["TWF", "HDF", "PWF", "OSF", "RNF"],
    )
   