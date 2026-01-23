import numpy as np
import pandas as pd


def calculate_descriptive_stats(data: pd.DataFrame, columns: list) -> dict:
    """Calculates descriptive statistics for specified columns in a DataFrame.

    Args:
        data: pandas DataFrame containing the data.
        columns: List of column names for which to calculate statistics.

    Returns:
        A dictionary containing descriptive statistics for each specified column.
        Returns an empty dictionary if input data is invalid.
    """
    if not isinstance(data, pd.DataFrame) or not isinstance(columns, list):
        print("Error: Invalid input data type. Expecting pandas DataFrame and a list of column names.")
        return {}

    stats = {}
    for col in columns:
        if col in data.columns:
            stats[col] = {
                "mean": data[col].mean(),
                "median": data[col].median(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
            }
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return stats


def handle_missing_values(data: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """Handles missing values in a DataFrame.

    Args:
        data: pandas DataFrame containing the data.
        method: Method for handling missing values. Options: 'drop' (default), 'mean', 'median', 'ffill', 'bfill'.

    Returns:
        A DataFrame with missing values handled according to the specified method.
        Returns the original DataFrame if an invalid method is specified.
    """
    if not isinstance(data, pd.DataFrame):
        print("Error: Invalid input data type. Expecting pandas DataFrame.")
        return data # Return original DataFrame for invalid input

    if method == "drop":
        cleaned_data = data.dropna()
    elif method == "mean":
        cleaned_data = data.fillna(data.mean())
    elif method == "median":
        cleaned_data = data.fillna(data.median())
    elif method == "ffill":
        cleaned_data = data.fillna(method="ffill")
    elif method == "bfill":
        cleaned_data = data.fillna(method="bfill")
    else:
        print(f"Warning: Invalid method '{method}'. Returning original DataFrame.")
        return data # Return original DataFrame for invalid method

    return cleaned_data



def normalize_data(data: pd.DataFrame, columns: list, method: str = "minmax") -> pd.DataFrame:
    """Normalizes specified columns in a DataFrame.

    Args:
        data: pandas DataFrame containing the data.
        columns: List of column names to normalize.
        method: Normalization method. Options: 'minmax' (default), 'zscore'.

    Returns:
        A DataFrame with specified columns normalized.
        Returns the original DataFrame if input data is invalid or no columns are specified.
    """
    if not isinstance(data, pd.DataFrame) or not isinstance(columns, list):
        print("Error: Invalid input data type. Expecting pandas DataFrame and a list of column names.")
        return data # Return original DataFrame for invalid input

    if not columns:
        print("Warning: No columns specified for normalization. Returning original DataFrame.")
        return data # Return original DataFrame if no columns specified

    normalized_data = data.copy()
    for col in columns:
        if col in normalized_data.columns:
            if method == "minmax":
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if min_val == max_val:
                    print(f"Warning: Column '{col}' has constant value, skipping normalization.")
                else:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean_val = normalized_data[col].mean()
                std_val = normalized_data[col].std()
                if std_val == 0:
                    print(f"Warning: Column '{col}' has zero standard deviation, skipping normalization.")
                else:
                    normalized_data[col] = (normalized_data[col] - mean_val) / std_val
            else:
                print(f"Warning: Invalid normalization method '{method}'. Skipping column '{col}'.")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")

    return normalized_data
