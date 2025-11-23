import os
from model_registry import register_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from config import CONFIG
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def get_models():
    models = {
        "Linear Regression": (LinearRegression(), {}),
        "Random Forest": (RandomForestRegressor(random_state=CONFIG.random_state),
                          {"model__n_estimators":[200], "model__max_depth":[None,10,20]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=CONFIG.random_state),
                              {"model__n_estimators":[200], "model__learning_rate":[0.05,0.1]}),
    }
    if HAS_XGB:
        models["XGBoost"] = (XGBRegressor(random_state=CONFIG.random_state, objective="reg:squarederror"),
                             {"model__n_estimators":[400], "model__max_depth":[4,6], "model__learning_rate":[0.05,0.1]})
    return models

def build_preprocessor():
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, CONFIG.categorical_cols),
        ("num", num_pipe, CONFIG.numeric_cols),
    ])

def train_and_select(df: pd.DataFrame):
    y = df[CONFIG.target]
    X = df.drop(columns=[CONFIG.target])
    pre = build_preprocessor()

    results = []
    best_artifact = None
    kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)

    for name, (estimator, grid) in get_models().items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        if grid:
            search = GridSearchCV(pipe, grid, cv=kf, scoring="r2", n_jobs=-1)
            search.fit(X, y)
            model = search.best_estimator_
        else:
            model = pipe.fit(X, y)

        # simple holdout eval for dashboard (train/test split)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=CONFIG.random_state)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))

        r2 = r2_score(yte, pred)
        results.append({"Model": name, "MAE": int(mae), "RMSE": int(rmse), "R2": round(r2,2)})
        # choose best by r2
        if best_artifact is None or r2 > best_artifact["r2"]:
            best_artifact = {"name": name, "pipeline": model, "r2": r2}

    # persist
    joblib.dump(best_artifact["pipeline"], "models/active_model.joblib")
    pd.DataFrame(results).to_csv("reports/model_performance.csv", index=False)
    # registry persist
    os.makedirs("models", exist_ok=True)
    tmp_path = "models/tmp_active.joblib"
    joblib.dump(best_artifact["pipeline"], tmp_path)
    version = register_model(best_artifact["name"], tmp_path, {"R2": round(best_artifact["r2"],4)})
    return results, f"{best_artifact['name']} (v{version})"
