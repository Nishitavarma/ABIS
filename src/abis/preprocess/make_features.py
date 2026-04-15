import pandas as pd
from typing import List, Tuple


DROP_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Target",          # some versions use Target
    "Failure Type",    # some versions use Failure Type
    "Machine failure", # some versions use Machine failure
]


def load_and_prepare_features(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the dataset and returns:
      - X: numeric feature dataframe (sensor columns only)
      - feature_names: list of columns used
    """
    df = pd.read_csv(csv_path)

    # Drop known non-sensor columns if they exist
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep ONLY numeric columns (sensor readings)
    X = df.select_dtypes(include=["number"]).copy()

    # If numeric columns are empty, something is wrong
    if X.shape[1] == 0:
        raise ValueError("No numeric sensor columns found. Check your CSV columns.")

    feature_names = list(X.columns)
    return X, feature_names
