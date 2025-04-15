import pandas as pd

def load_dataset(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_dataset(df, filepath):
    """Save DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
    