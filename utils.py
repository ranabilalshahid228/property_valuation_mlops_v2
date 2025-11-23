import numpy as np
import pandas as pd

def quality_score(df: pd.DataFrame) -> float:
    # simple 0-1 scoring based on missingness and duplicates
    miss = df.isna().mean().mean()
    dups = (len(df) - len(df.drop_duplicates())) / max(len(df),1)
    score = max(0.0, 1.0 - (miss*0.7 + dups*0.3))
    return round(score,3)

def iqr_outliers(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (series < lower) | (series > upper)

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected).astype(float)
    actual = np.asarray(actual).astype(float)
    if len(expected)==0 or len(actual)==0:
        return 0.0
    cuts = np.quantile(expected, np.linspace(0,1,bins+1))
    cuts[0]-=1e-6; cuts[-1]+=1e-6
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_perc = np.where(e_counts==0, 1e-6, e_counts/ e_counts.sum())
    a_perc = np.where(a_counts==0, 1e-6, a_counts/ a_counts.sum())
    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi)

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default
