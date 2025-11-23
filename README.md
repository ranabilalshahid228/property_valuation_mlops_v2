#  Residential Property Valuation MLOps Solution (Streamlit)

Production‑ready, end‑to‑end MLOps project for residential property valuation. The app matches the UI/flows shown in the screenshots: Dashboard, Data Processing, Model Training, Predictions, and Model Monitoring.

##  Quick Start
```bash
# 1) create env and install deps
pip install -r requirements.txt

# 2) run
streamlit run app.py
```

##  Structure
```
property_valuation_mlops/
├─ app.py                # Streamlit dashboard with 5 sections
├─ config.py             # Central configuration & thresholds
├─ data_processor.py     # Ingestion, validation, feature engineering, quality score
├─ model_trainer.py      # Multi‑model training + GridSearchCV
├─ model_predictor.py    # Single & batch predictions + logging
├─ model_monitor.py      # KPIs and drift proxy via PSI
├─ utils.py              # Helpers (PSI, data quality)
├─ requirements.txt
├─ data/                 # Sample/training data (place your csv here)
├─ models/               # Persisted joblib model
├─ logs/                 # recent_predictions.csv
└─ reports/              # model_performance.csv
```

##  Features
- Feature engineering: `price_per_sqft`, `bed_bath_ratio`, `sqft_per_bedroom`, `lot_sqft_ratio`, `age_category`, `size_category`.
- Models: Linear Regression, Random Forest, Gradient Boosting, (optional) XGBoost.
- Metrics: MAE, RMSE, R² with 5‑fold CV & simple holdout display.
- Explanations & Monitoring: Recent predictions table, KPI cards, PSI‑based drift score.

##  Notes
- Train the model in **Model Training** before predicting.
- For batch predictions, upload a CSV with the core feature columns (no target needed).
- XGBoost is optional—if not available, the app falls back to other models.
```

