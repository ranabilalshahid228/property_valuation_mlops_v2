"""
melbourne_to_schema.py  (robust version)
Maps Kaggle Melbourne Housing to your app schema:
bedrooms, bathrooms, sqft, lot_size, age, location, property_type, garage, price
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

FT_PER_M2 = 10.7639

def infer_bedrooms(df: pd.DataFrame) -> pd.Series:
    if "Bedroom2" in df.columns and df["Bedroom2"].notna().any():
        b = pd.to_numeric(df["Bedroom2"], errors="coerce")
    elif "Rooms" in df.columns:
        b = pd.to_numeric(df["Rooms"], errors="coerce")  # approximation
    else:
        b = pd.Series([np.nan] * len(df))
    return b.fillna(3).clip(1, 10).astype(int)

def map_property_type(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(["Single Family"] * 0)  # will be expanded later
    def _map(x):
        x = str(x).strip().lower()
        if x == "h": return "Single Family"
        if x == "u": return "Condo"
        if x == "t": return "Townhouse"
        return "Multi Family"
    return s.apply(_map)

def map_garage(car: pd.Series | None, n: int) -> pd.Series:
    if car is None:
        return pd.Series(["None"] * n)
    def _g(v):
        try:
            v = int(v)
        except Exception:
            return "None"
        if v <= 0:  return "None"
        if v == 1:  return "Carport"
        return "Attached"  # ≥2
    return car.apply(_g)

def year_from_date(date_series: pd.Series) -> pd.Series:
    # Dates like "3/09/2016" etc.
    try:
        d = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
        return d.dt.year
    except Exception:
        return pd.Series([np.nan] * len(date_series))

def map_location_row(row) -> str:
    region = str(row.get("Regionname", "")).lower()
    council = str(row.get("CouncilArea", "")).lower()
    dist = row.get("Distance", np.nan)
    try:
        dist = float(dist)
    except Exception:
        dist = np.nan
    yb = row.get("YearBuilt", np.nan)
    try:
        yb = float(yb)
    except Exception:
        yb = np.nan

    # 1) Waterfront – rough heuristic using coastal councils / southern metro
    if any(k in council for k in ["bayside", "port phillip", "mornington", "kingston", "frankston"]) \
       or ("southern" in region and "metropolitan" in region):
        return "Waterfront"
    # 2) Downtown – near CBD
    if not np.isnan(dist) and dist <= 5:
        return "Downtown"
    # 3) New Development – very recent stock
    if not np.isnan(yb) and yb >= 2015:
        return "New Development"
    # 4) Historic – older builds
    if not np.isnan(yb) and yb <= 1950:
        return "Historic"
    # 5) Otherwise
    return "Suburbs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="Path to Melbourne CSV (any name)")
    ap.add_argument("--out", dest="output_path", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Read CSV (handles your double .csv path too)
    df = pd.read_csv(args.input_path)

    out = pd.DataFrame(index=df.index)

    # bedrooms
    out["bedrooms"] = infer_bedrooms(df)

    # bathrooms
    baths = pd.to_numeric(df.get("Bathroom", np.nan), errors="coerce").fillna(1)
    out["bathrooms"] = baths.clip(1, 5).astype(int)

    # sqft – prefer BuildingArea (m2 → ft2). If missing, estimate by rooms.
    ba = pd.to_numeric(df.get("BuildingArea", np.nan), errors="coerce")
    sqft_from_ba = (ba * FT_PER_M2)
    # Estimate: base 500 ft2 + 350 ft2 per bedroom as fallback
    sqft_est = 500 + out["bedrooms"] * 350
    sqft = sqft_from_ba.fillna(sqft_est)
    out["sqft"] = sqft.round().clip(500, 10000).astype(int)

    # lot_size – Landsize (m2 → ft2), fallback 4000
    ls = pd.to_numeric(df.get("Landsize", np.nan), errors="coerce")
    lot = (ls * FT_PER_M2).fillna(4000)
    out["lot_size"] = lot.round().clip(1000, 50000).astype(int)

    # age – YearBuilt preferred; fallback from Date; else 20
    current_year = datetime.now().year
    yb = pd.to_numeric(df.get("YearBuilt", np.nan), errors="coerce")
    if yb.isna().all() and "Date" in df.columns:
        yb = year_from_date(df["Date"])
    age = (current_year - yb).where(~yb.isna(), other=20)
    out["age"] = age.clip(0, 120).astype(int)

    # location – optional inputs; default Suburbs
    if "Distance" not in df.columns: df["Distance"] = np.nan
    if "CouncilArea" not in df.columns: df["CouncilArea"] = ""
    if "Regionname" not in df.columns: df["Regionname"] = ""
    out["location"] = df.apply(map_location_row, axis=1)

    # property_type – from Type (h/u/t). If missing, default Single Family.
    if "Type" in df.columns:
        out["property_type"] = map_property_type(df["Type"])
    else:
        out["property_type"] = pd.Series(["Single Family"] * len(df))

    # garage – from Car if present; else None
    out["garage"] = map_garage(df["Car"] if "Car" in df.columns else None, len(df))

    # target price
    out["price"] = pd.to_numeric(df.get("Price", np.nan), errors="coerce")

    # Keep rows with price only (training)
    out = out.dropna(subset=["price"]).copy()
    out["price"] = out["price"].clip(100000, 2_000_000).astype(int)

    # Final caps
    out["bedrooms"] = out["bedrooms"].clip(1, 10).astype(int)
    out["bathrooms"] = out["bathrooms"].clip(1, 5).astype(int)
    out["sqft"] = out["sqft"].clip(500, 10000).astype(int)
    out["lot_size"] = out["lot_size"].clip(1000, 50000).astype(int)
    out["age"] = out["age"].clip(0, 120).astype(int)

    out.to_csv(args.output_path, index=False)
    print(f"Saved mapped dataset to: {args.output_path}")
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()
