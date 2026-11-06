import numpy as np
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file and raise an explicit error when it cannot be read."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"File not found: {file_path}") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"No data: {file_path} is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Parsing error: Could not parse {file_path}.") from error


def remove_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove identical rows and reset the index."""
    return data.drop_duplicates(ignore_index=True)


def standardized_mean_difference(
    data: pd.DataFrame, features: pd.Index
) -> pd.Series:
    """Rank features by the standardized difference between class means."""
    fraud = data[data["Class"] == 1][features]
    normal = data[data["Class"] == 0][features]
    pooled_std = np.sqrt((fraud.var() + normal.var()) / 2)
    smd = (fraud.mean() - normal.mean()) / pooled_std
    return smd.sort_values(key=abs, ascending=False)


if __name__ == "__main__":
    data = remove_duplicate_rows(load_data("data/creditcard.csv"))
    print(f"Dataset ready: {len(data):,} rows, {data['Class'].sum()} fraudulent transactions.")
