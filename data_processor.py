import pandas as pd
import numpy as np
from utils import quality_score, iqr_outliers
from config import CONFIG

def _age_category(age: int) -> str:
    if age <= 5: return "New"
    if age <= 20: return "Recent"
    if age <= 50: return "Established"
    return "Old"

def _size_category(sqft: int) -> str:
    if sqft < 1200: return "Small"
    if sqft < 2200: return "Medium"
    if sqft < 3200: return "Large"
    return "XLarge"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # engineered continuous
    df["price_per_sqft"] = (df.get("price", np.nan) / df["sqft"]).replace([np.inf,-np.inf], np.nan)
    df["bed_bath_ratio"] = (df["bedrooms"] / df["bathrooms"].replace(0, np.nan))
    df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"].replace(0, np.nan)
    df["lot_sqft_ratio"] = df["sqft"] / df["lot_size"].replace(0, np.nan)
    # categories
    df["age_category"] = df["age"].apply(_age_category)
    df["size_category"] = df["sqft"].apply(_size_category)
    return df

def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"n_rows": len(df)}
    # basic expected columns
    required = ["bedrooms","bathrooms","sqft","lot_size","age","location","property_type","garage"]
    missing_cols = [c for c in required if c not in df.columns]
    for c in missing_cols:
        df[c] = np.nan
    # cast numeric
    for c in ["bedrooms","bathrooms","sqft","lot_size","age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # fill missing
    df["bedrooms"].fillna(df["bedrooms"].median(), inplace=True)
    df["bathrooms"].fillna(df["bathrooms"].median(), inplace=True)
    df["sqft"].fillna(df["sqft"].median(), inplace=True)
    df["lot_size"].fillna(df["lot_size"].median(), inplace=True)
    df["age"].fillna(df["age"].median(), inplace=True)
    for c in ["location","property_type","garage"]:
        df[c].fillna("Unknown", inplace=True)

    # remove duplicates
    dup_cnt = len(df) - len(df.drop_duplicates())
    df = df.drop_duplicates()

    # outlier flags
    outlier_cols = ["sqft","lot_size","age"]
    outliers = {c:int(iqr_outliers(df[c]).sum()) for c in outlier_cols}

    # engineer
    df = engineer_features(df)

    # quality
    q_score = quality_score(df)

    report.update({"duplicates": int(dup_cnt), "outliers": outliers, "quality_score": q_score})
    return df, report
