import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import base64
from config import CONFIG
from data_processor import validate_and_clean, engineer_features
from model_trainer import train_and_select
from model_predictor import predict_single, predict_batch
from model_monitor import compute_perf_summary, load_recent_predictions
from model_registry import list_versions, activate_version
from datetime import datetime
from io import BytesIO

# --- Lead storage helpers ---
LEADS_CSV = "data/user_leads.csv"
UPLOADS_DIR = "data/uploads"

def _ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def save_user_lead(row: dict) -> None:
    """Append a single lead row to CSV with header if new."""
    _ensure_dirs()
    df = pd.DataFrame([row])
    header = not os.path.exists(LEADS_CSV)
    df.to_csv(LEADS_CSV, mode="a", header=header, index=False)

def save_uploaded_images(files, lead_id: str) -> list[str]:
    """Save uploaded images to a per-lead folder and return relative paths."""
    _ensure_dirs()
    if not files:
        return []
    folder = os.path.join(UPLOADS_DIR, lead_id)
    os.makedirs(folder, exist_ok=True)
    saved = []
    for i, f in enumerate(files, start=1):
        # keep jpg/png extensions where possible
        ext = ".jpg"
        name = getattr(f, "name", f"img_{i}.jpg")
        if isinstance(name, str) and name.lower().endswith((".png", ".jpeg", ".jpg")):
            ext = os.path.splitext(name)[1].lower()
        path = os.path.join(folder, f"photo_{i}{ext}")
        with open(path, "wb") as out:
            out.write(f.read())
        saved.append(path)
    return saved
st.set_page_config(page_title="Property Valuation MLOps", layout="wide", page_icon="🏠")

st.sidebar.header("🏠 Property Valuation MLOps")
page = st.sidebar.selectbox("Navigate to:", ["Dashboard", "Data Processing", "Model Training", "Predictions", "Model Monitoring"])

def kpi_card(col, title, value, delta=None):
    with col:
        st.metric(label=title, value=value, delta=delta)

if page == "Dashboard":
    st.title("Residential Property Valuation MLOps Dashboard")
    kpis = compute_perf_summary()
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_card(c1, "Models Trained", kpis["models_trained"], "+2" if kpis["models_trained"] else None)
    kpi_card(c2, "Active Model Accuracy", f"{kpis['active_model_accuracy']*100:.1f}%", "+1.2%")
    kpi_card(c3, "Predictions Made", kpis["predictions_made"], "+156")
    kpi_card(c4, "Data Drift Score", kpis["data_drift_score"], "-0.03")
    kpi_card(c5, "Avg Latency", f"{kpis['avg_latency_ms']} ms")
    kpi_card(c6, "Throughput", f"{kpis['throughput_per_min']} /min")

    st.subheader("Model Performance Overview")
    perf_csv = "reports/model_performance.csv"
    if not os.path.exists(perf_csv):
        st.info("No performance report yet. Train models in the 'Model Training' section.")
    else:
        dfp = pd.read_csv(perf_csv)
        st.dataframe(dfp)
        colA, colB = st.columns(2)
        with colA:
            fig = px.bar(dfp, x="Model", y="MAE", title="Mean Absolute Error by Model")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.bar(dfp, x="Model", y="R2", title="R² Score by Model")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Predictions")
    rec = load_recent_predictions()
    st.dataframe(rec)

elif page == "Data Processing":
    st.title("Data Processing")
    f = st.file_uploader("Upload CSV with property data", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        st.write("Raw Data", df.head())
        clean, report = validate_and_clean(df)
        st.write("Engineered & Cleaned Data", clean.head())
        st.write("Data Quality Report", report)
        st.download_button("Download Cleaned CSV", clean.to_csv(index=False).encode("utf-8"),
                           file_name="cleaned_properties.csv", mime="text/csv")
    else:
        st.info("Upload a CSV to see processing and quality checks.")

elif page == "Model Training":
    st.title("Model Training")
    st.write("Upload a historical dataset with target column `price`.")
    f = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
    if f is not None:
        df = pd.read_csv(f)
        if "price" not in df.columns:
            st.error("No 'price' column found.")
        else:
            df, _ = validate_and_clean(df)
            with st.spinner("Training models..."):
                results, best_name = train_and_select(df)
            st.success(f"Training complete. Best model: {best_name}")
            st.dataframe(pd.DataFrame(results))
    else:
        st.info("Upload training data to start.")

elif page == "Predictions":
    st.title("Predictions")
    st.subheader("Single Property Valuation")
    cols1 = st.columns(2)
    with cols1[0]:
        bedrooms = st.number_input("Number of bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Number of bathrooms", 1, 5, 2)
        sqft = st.number_input("Square footage", 500, 10000, 2000)
        lot_size = st.number_input("Lot size (sqft)", 1000, 50000, 8000)
    with cols1[1]:
        age = st.number_input("Property age (years)", 0, 100, 15)
        garage = st.selectbox("Garage", CONFIG.garages)
        location = st.selectbox("Location", CONFIG.locations)
        property_type = st.selectbox("Property Type", CONFIG.property_types)
    if st.button("Predict Property Value"):
        payload = {"bedrooms":bedrooms,"bathrooms":bathrooms,"sqft":sqft,"lot_size":lot_size,"age":age,
                   "garage":garage,"location":location,"property_type":property_type}
        res = predict_single(payload)
        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"Predicted value: ${res['prediction']:,} | Confidence: {res['confidence']*100:.1f}%")
            if res.get('explanation'):
                st.subheader('Feature Importance for this Prediction')
                exp_df = pd.DataFrame({'feature': list(res['explanation'].keys()), 'importance': list(res['explanation'].values())})
                fig = px.bar(exp_df, x='importance', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Batch Property Predictions")
    f = st.file_uploader("Upload CSV file with property features", type=["csv"], key="batch_csv")
    if f is not None:
        df = pd.read_csv(f)
        try:
            out = predict_batch(df)
            st.write(out.head())
            st.download_button("Download Predictions CSV", out.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

elif page == "Model Monitoring":
    st.title("Model Monitoring")
    kpis = compute_perf_summary()
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_card(c1, "Models Trained", kpis["models_trained"])
    kpi_card(c2, "Active Model Accuracy", f"{kpis['active_model_accuracy']*100:.1f}%")
    kpi_card(c3, "Predictions Made", kpis["predictions_made"])
    kpi_card(c4, "Data Drift Score", kpis["data_drift_score"])
    st.metric("Avg Latency", f"{kpis['avg_latency_ms']} ms")
    st.metric("Throughput", f"{kpis['throughput_per_min']} /min")

    rec = load_recent_predictions()
    st.subheader("Recent Predictions")
    st.dataframe(rec)

    st.subheader("Model Registry")
    versions = list_versions()
    if versions:
        reg_df = pd.DataFrame(versions)
        st.dataframe(reg_df[["version","name","created_at","metrics"]])
        choose = st.selectbox("Activate version", [v["version"] for v in versions], key="ver_select")
        if st.button("Activate Selected Version"):
            try:
                activate_version(choose)
                st.success(f"Activated version {choose}.")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("No versions in registry yet. Train a model to create one.")
