import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Cleans the data by dropping missing and duplicate values."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df