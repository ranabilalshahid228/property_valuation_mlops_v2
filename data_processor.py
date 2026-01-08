import pandas as pd
import numpy as np

def optimize_memory(df):
    """Downcast float64 to float32 and int64 to int32 to save RAM."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def clean_data(path):
    # Use 'usecols' to only load what's necessary if the file is massive
    df = pd.read_csv(path)
    df = optimize_memory(df)
    # Standard cleaning (handle NaNs, factorize categories)
    df = df.fillna(df.median(numeric_only=True))
    return df
