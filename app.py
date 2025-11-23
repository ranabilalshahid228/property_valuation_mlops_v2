import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import base64
from config import CONFIG
from data_processor import validate_and_clean, engineer_features
from datetime import datetime
from io import BytesIO

# --- Lead storage helpers (same as before) ---
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


# ---------- GLOBAL STYLING HELPERS ----------

def set_background("assets/new_bg.jpg"):
    """Set a full-page background image if file exists."""
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_global_style():
    """Inject custom CSS for a modern, property-style look."""
    st.markdown(
        """
        <style>
        /* Center main content slightly and add padding */
        .main {
            padding: 2rem 3rem;
        }

        /* Card-like look for metrics */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.9);
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }

        /* Titles */
        h1, h2, h3, h4 {
            font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102A43 0%, #243B53 40%, #F0F4F8 100%);
            color: white;
        }

        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] label {
            color: #E0E7FF !important;
        }

        /* DataFrames */
        .stDataFrame, .stTable {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 10px;
            padding: 0.5rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 999px;
            padding: 0.4rem 1.2rem;
            border: none;
            background: #2563EB;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: #1D4ED8;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 999px;
            padding: 0.3rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- PAGE CONFIG & BRANDING ----------

ICON_PATH = "assets/logo.png"
if os.path.exists(ICON_PATH):
    st.set_page_config(
        page_title="Property Valuation MLOps",
        layout="wide",
        page_icon=ICON_PATH,
    )
else:
    st.set_page_config(
        page_title="Property Valuation MLOps",
        layout="wide",
        page_icon="🏠",
    )

set_background("assets/background.jpg")
set_global_style()

if os.path.exists(ICON_PATH):
    st.sidebar.image(ICON_PATH, use_column_width=True)
st.sidebar.header("🏠 Property Valuation MLOps")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Data Processing", "Model Training", "Predictions", "Model Monitoring"],
)


def kpi_card(col, title, value, delta=None):
    with col:
        st.metric(label=title, value=value, delta=delta)


# ---------- PAGES ----------

if page == "Dashboard":
    from model_monitor import compute_perf_summary, load_recent_predictions

    st.title("🏙️ Residential Property Valuation MLOps Dashboard")

    # KPIs
    kpis = compute_perf_summary()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "Models Trained", kpis["models_trained"], "+2" if kpis["models_trained"] else None)
    kpi_card(c2, "Active Model Accuracy", f"{kpis['active_model_accuracy']*100:.1f}%", "+1.2%")
    kpi_card(c3, "Predictions Made", kpis["predictions_made"], "+156")
    kpi_card(c4, "Data Drift Score", kpis["data_drift_score"], "-0.03")
    kpi_card(c5, "Avg Latency", f"{kpis['avg_latency_ms']} ms")
    kpi_card(c6, "Throughput", f"{kpis['throughput_per_min']} /min")

    st.markdown("### 📊 Model Performance & Activity")

    perf_csv = "reports/model_performance.csv"
    recent = load_recent_predictions()

    tab_perf, tab_recent = st.tabs(["Model Performance", "Recent Predictions"])

    # --- Interactive model performance tab ---
    with tab_perf:
        if not os.path.exists(perf_csv):
            st.info("No performance report yet. Train models in the **Model Training** section.")
        else:
            dfp = pd.read_csv(perf_csv)

            st.markdown("#### Raw performance table")
            st.dataframe(dfp, use_container_width=True)

            st.markdown("#### Interactive performance view")

            cols_ctrl = st.columns([2, 2, 2])
            with cols_ctrl[0]:
                metric = st.selectbox("Metric", ["MAE", "RMSE", "R2"], index=2)
            with cols_ctrl[1]:
                chart_type = st.radio("Chart type", ["Bar", "Line"], horizontal=True)
            with cols_ctrl[2]:
                models_selected = st.multiselect(
                    "Models",
                    options=dfp["Model"].tolist(),
                    default=dfp["Model"].tolist(),
                )

            dfp_f = dfp[dfp["Model"].isin(models_selected)].copy()

            if dfp_f.empty:
                st.warning("No models selected – please choose at least one.")
            else:
                title = f"{metric} by Model"
                if chart_type == "Bar":
                    fig = px.bar(
                        dfp_f,
                        x="Model",
                        y=metric,
                        title=title,
                        text=metric,
                    )
                    fig.update_traces(textposition="outside")
                else:
                    fig = px.line(
                        dfp_f,
                        x="Model",
                        y=metric,
                        title=title,
                        markers=True,
                    )
                st.plotly_chart(fig, use_container_width=True)

    # --- Interactive recent predictions tab ---
    with tab_recent:
        if recent.empty:
            st.info("No predictions logged yet. Make some predictions to see activity here.")
        else:
            st.markdown("#### Filters")

            c1, c2 = st.columns(2)
            with c1:
                locations = ["All"] + sorted(recent["location"].dropna().unique().tolist())
                choice_loc = st.selectbox("Location", locations)
            with c2:
                max_days = 60
                days_back = st.slider(
                    "Show predictions from last N days",
                    min_value=1,
                    max_value=max_days,
                    value=30,
                )

            rf = recent.copy()
            rf["timestamp"] = pd.to_datetime(rf["timestamp"], errors="coerce")

            if choice_loc != "All":
                rf = rf[rf["location"] == choice_loc]

            cutoff = rf["timestamp"].max() - pd.Timedelta(days=days_back)
            rf = rf[rf["timestamp"] >= cutoff]

            st.markdown("#### Filtered predictions table")
            st.dataframe(rf.sort_values("timestamp", ascending=False), use_container_width=True)

            if not rf.empty:
                rf_ts = rf.sort_values("timestamp")
                fig = px.line(
                    rf_ts,
                    x="timestamp",
                    y="predicted_value",
                    title="Predicted Values Over Time",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download filtered predictions",
                    rf.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No predictions match the current filters.")


elif page == "Data Processing":
    st.title("🧼 Data Processing & Quality Checks")
    st.write("Upload raw property data and see how it’s cleaned and enriched with features.")

    f = st.file_uploader("Upload CSV with property data", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        st.markdown("#### Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        clean, report = validate_and_clean(df)

        st.markdown("#### Engineered & Cleaned Data")
        st.dataframe(clean.head(), use_container_width=True)

        st.markdown("#### Data Quality Report")
        st.json(report)

        st.download_button(
            "⬇️ Download Cleaned CSV",
            clean.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_properties.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV to see processing and quality checks.")


elif page == "Model Training":
    from model_trainer import train_and_select

    st.title("🧠 Model Training")
    st.write("Upload a historical dataset with target column `price` to train valuation models.")

    f = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
    if f is not None:
        df = pd.read_csv(f)
        if "price" not in df.columns:
            st.error("No `price` column found. Please include the target variable.")
        else:
            df, _ = validate_and_clean(df)
            with st.spinner("Training models... this can take a bit on larger datasets."):
                results, best_name = train_and_select(df)
            st.success(f"✅ Training complete. Best model: **{best_name}**")

            res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
            st.dataframe(res_df, use_container_width=True)
    else:
        st.info("Upload training data to start.")


elif page == "Predictions":
    from model_predictor import predict_single, predict_batch

    st.title("🔮 Property Value Predictions")

    tab_single, tab_batch = st.tabs(["Single Property", "Batch Properties"])

    # --- Single property tab ---
    with tab_single:
        st.subheader("🏠 Single Property Valuation")

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

        if st.button("Predict Property Value", key="single_predict"):
            payload = {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "lot_size": lot_size,
                "age": age,
                "garage": garage,
                "location": location,
                "property_type": property_type,
            }
            res = predict_single(payload)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(
                    f"Estimated market value: **${res['prediction']:,}** "
                    f"(confidence: {res['confidence']*100:.1f}%)"
                )
                if res.get("explanation"):
                    st.subheader("Feature Importance for this Prediction")
                    exp_df = pd.DataFrame(
                        {
                            "feature": list(res["explanation"].keys()),
                            "importance": list(res["explanation"].values()),
                        }
                    )
                    fig = px.bar(exp_df, x="importance", y="feature", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)

    # --- Batch predictions tab ---
    with tab_batch:
        st.subheader("📁 Batch Property Predictions")

        f = st.file_uploader(
            "Upload CSV file with property features",
            type=["csv"],
            key="batch_csv",
        )
        if f is not None:
            df = pd.read_csv(f)
            try:
                out = predict_batch(df)
                st.markdown("#### Preview of Predictions")
                st.dataframe(out.head(), use_container_width=True)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Upload a CSV with property features (no `price` column needed).")


elif page == "Model Monitoring":
    from model_monitor import compute_perf_summary, load_recent_predictions
    from model_registry import list_versions, activate_version

    st.title("📈 Model Monitoring & Registry")

    kpis = compute_perf_summary()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "Models Trained", kpis["models_trained"])
    kpi_card(c2, "Active Model Accuracy", f"{kpis['active_model_accuracy']*100:.1f}%")
    kpi_card(c3, "Predictions Made", kpis["predictions_made"])
    kpi_card(c4, "Data Drift Score", kpis["data_drift_score"])
    kpi_card(c5, "Avg Latency", f"{kpis['avg_latency_ms']} ms")
    kpi_card(c6, "Throughput", f"{kpis['throughput_per_min']} /min")

    rec = load_recent_predictions()
    st.subheader("Recent Predictions")
    st.dataframe(rec, use_container_width=True)

    st.subheader("Model Registry")
    versions = list_versions()
    if versions:
        reg_df = pd.DataFrame(versions)
        st.dataframe(reg_df[["version", "name", "created_at", "metrics"]], use_container_width=True)

        choose = st.selectbox(
            "Activate version",
            [v["version"] for v in versions],
            key="ver_select",
        )
        if st.button("Activate Selected Version"):
            try:
                activate_version(choose)
                st.success(f"Activated version {choose}.")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("No versions in registry yet. Train a model to create one.")
