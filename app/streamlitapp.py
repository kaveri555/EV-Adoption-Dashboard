# ============================================================ 
# EV Adoption Dashboard — Full Version with ML & Advanced Tabs
# ============================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier

# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="EV Adoption Insights", layout="wide", page_icon="🔋")
st.title("🔋 Electric Vehicle Adoption Across U.S. States")

# ------------------------------------------------------------
# SIDEBAR (Logo + Title + Help + Filters)
# ------------------------------------------------------------
with st.sidebar:
    logo_path = "app/logo"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/e/e4/Electric_car_icon.png",
            use_container_width=True,
        )

    st.title("EV-Adoption Dashboard")
    st.caption("Developed by Kaveri | CMSE 830")

    with st.expander("❓ How to use this app", expanded=False):
        st.markdown(
            """
            - Use the filters below to control **all tabs**.
            - Start with **🏠 Home** for a quick overview.
            - Use **🔮 Predictive Models**, **⚡ What-If**, and **⚖ Fairness** for advanced analysis.
            """
        )

# ------------------------------------------------------------
# File Paths
# ------------------------------------------------------------
DATA_CANDIDATES = [
    "data/processed/ev_master_merged.csv",        # preferred
    "data/processed/ev_charging_income_state.csv" # fallback
]

STATE_YEAR_STATS_PATH = "data/processed/ev_state_year_stats.csv"
STATE_RECENT_STATS_PATH = "data/processed/ev_state_recent_stats.csv"
DF_KNN_PATH = "data/processed/df_knn.csv"  # optional, if you saved it

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            st.success(f"✅ Loaded: {p} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return df
    st.error("❌ No data file found. Please check your data/processed folder.")
    st.stop()

def clean_state_codes(df):
    USPS = {
        "Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA",
        "Colorado": "CO","Connecticut": "CT","Delaware": "DE","District of Columbia": "DC",
        "Florida": "FL","Georgia": "GA","Hawaii": "HI","Idaho": "ID","Illinois": "IL",
        "Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA",
        "Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN",
        "Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV",
        "New Hampshire": "NH","New Jersey": "NJ","New Mexico": "NM","New York": "NY",
        "North Carolina": "NC","North Dakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR",
        "Pennsylvania": "PA","Rhode Island": "RI","South Carolina": "SC","South Dakota": "SD",
        "Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA",
        "Washington": "WA","West Virginia": "WV","Wisconsin": "WI","Wyoming": "WY",
    }
    df = df.copy()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
        df["state_usps"] = df["state"].map(USPS)
        df = df.dropna(subset=["state_usps"])
        df["state_usps"] = df["state_usps"].str.upper()
    return df

def assign_region(usps_series):
    usps_series = usps_series.astype(str).str.upper().str.strip()
    northeast = {"ME","NH","VT","MA","RI","CT","NY","NJ","PA"}
    midwest   = {"OH","MI","IN","IL","WI","MN","IA","MO","ND","SD","NE","KS"}
    south     = {"DE","MD","DC","VA","WV","NC","SC","GA","FL","KY","TN","AL","MS","AR","LA","OK","TX"}
    west      = {"MT","ID","WY","CO","NM","AZ","UT","NV","WA","OR","CA","AK","HI"}

    def region(usps):
        if usps in northeast: return "Northeast"
        if usps in midwest:   return "Midwest"
        if usps in south:     return "South"
        if usps in west:      return "West"
        return "Other"

    return usps_series.map(region)

def guard_empty(df, context_msg=""):
    if df.empty:
        st.warning(f"⚠ No data available for the selected filters. {context_msg}")
        st.stop()

def generate_summary_report(df):
    buffer = io.StringIO()
    print("=" * 70, file=buffer)
    print("🔋 ELECTRIC VEHICLE ADOPTION — ANALYSIS SUMMARY REPORT", file=buffer)
    print("=" * 70, file=buffer)
    print(f"Total Records: {len(df)}", file=buffer)
    print(f"Total States: {df['state'].nunique()}", file=buffer)
    print("-" * 70, file=buffer)

    key_cols = [c for c in ["EV_Count", "station_count", "median_income", "EV_per_1000"] if c in df.columns]
    if key_cols:
        print("\n📈 BASIC DESCRIPTIVE STATISTICS\n", file=buffer)
        desc = df[key_cols].describe().T.round(2)
        print(desc, file=buffer)

    corr_cols = [c for c in ["EV_Count","station_count","median_income","EV_per_station","EV_per_1000"] if c in df.columns]
    if corr_cols:
        print("\n🔗 CORRELATION MATRIX\n", file=buffer)
        corr = df[corr_cols].corr().round(2)
        print(corr, file=buffer)

    print("\n⚠️ OUTLIER SUMMARY (Z-Score > 3)\n", file=buffer)
    if key_cols:
        z_df = df[key_cols].apply(lambda x: np.abs(zscore(x, nan_policy="omit")))
        outlier_counts = (z_df > 3).sum()
        print(outlier_counts, file=buffer)

    if "Income_Q" in df.columns:
        print("\n⚖️ FAIRNESS CHECK BY INCOME QUARTILE\n", file=buffer)
        fairness = df.groupby("Income_Q")[key_cols].mean().round(2)
        print(fairness, file=buffer)

    print("\n💡 INTERPRETATION & KEY INSIGHTS", file=buffer)
    print("• High-income states show greater EV adoption and charger density.", file=buffer)
    print("• Strong EV–Station correlation confirms infrastructure alignment.", file=buffer)
    print("• Income and policy support appear to influence EV accessibility.", file=buffer)
    print("=" * 70, file=buffer)

    report_text = buffer.getvalue()
    return report_text

# ------------------------------------------------------------
# Load Data (base + optional state-year stats + df_knn)
# ------------------------------------------------------------
df_base = load_first_existing(DATA_CANDIDATES)
df_base = clean_state_codes(df_base)

# Numeric cleaning
for col in ["EV_Count", "station_count", "median_income"]:
    if col in df_base.columns:
        df_base[col] = (
            df_base[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .replace("", np.nan)
        )
        df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

# Population-normalized metrics if possible
if "population" in df_base.columns and "EV_Count" in df_base.columns:
    df_base["EV_per_1000"] = (df_base["EV_Count"] / df_base["population"]) * 1000

if "population" in df_base.columns and "station_count" in df_base.columns:
    df_base["Stations_per_100k"] = (df_base["station_count"] / df_base["population"]) * 100000

# EV_per_station
if {"EV_Count","station_count"}.issubset(df_base.columns):
    df_base["EV_per_station"] = df_base["EV_Count"] / df_base["station_count"].replace(0, np.nan)

# Renewable share if columns exist
if {"renewable_share_pct"}.issubset(df_base.columns):
    df_base["renewable_share"] = df_base["renewable_share_pct"]
elif {"renewable_energy_btu","total_energy_btu"}.issubset(df_base.columns):
    df_base["renewable_share"] = (df_base["renewable_energy_btu"] / df_base["total_energy_btu"]) * 100

# Charger gap if columns exist
if {"EV_Count", "station_count"}.issubset(df_base.columns):
    df_base["ideal_station_count"] = df_base["EV_Count"] / 20.0
    df_base["charger_gap"] = df_base["ideal_station_count"] - df_base["station_count"]

# Median income sanity
if "median_income" in df_base.columns:
    df_base["median_income"] = pd.to_numeric(df_base["median_income"], errors="coerce")
    df_base["median_income"].fillna(df_base["median_income"].median(), inplace=True)

# Policy score naming sanity
if "policy_score" in df_base.columns:
    df_base["policy"] = df_base["policy_score"]
elif "policy" in df_base.columns:
    pass
else:
    df_base["policy"] = np.nan

# Region column
if "state_usps" in df_base.columns:
    df_base["region"] = assign_region(df_base["state_usps"])
else:
    df_base["region"] = "Other"

# Income & policy quartiles (for fairness & presets)
if "median_income" in df_base.columns:
    df_base["Income_Q"] = pd.qcut(
        df_base["median_income"], 4,
        labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

if "policy" in df_base.columns and df_base["policy"].notna().any():
    df_base["Policy_Q"] = pd.qcut(
        df_base["policy"], 4,
        labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

# Optional: state-year stats for trends/forecast
state_year_df = None
if os.path.exists(STATE_YEAR_STATS_PATH):
    try:
        state_year_df = pd.read_csv(STATE_YEAR_STATS_PATH)
    except Exception:
        state_year_df = None

# Optional: df_knn if you saved it; else we just use df_base as "already imputed"
df_knn = None
if os.path.exists(DF_KNN_PATH):
    try:
        df_knn = pd.read_csv(DF_KNN_PATH)
        df_knn = clean_state_codes(df_knn)
        if "region" not in df_knn.columns and "state_usps" in df_knn.columns:
            df_knn["region"] = assign_region(df_knn["state_usps"])
    except Exception:
        df_knn = None

if df_knn is None:
    df_knn = df_base.copy()

# ------------------------------------------------------------
# Global Filters + Presets (apply to df_base / df_knn for visuals)
# ------------------------------------------------------------
st.sidebar.markdown("### 🔍 Global Filters")

all_states = sorted(df_base["state"].unique())
all_regions = sorted(df_base["region"].unique())

selected_regions = st.sidebar.multiselect(
    "Regions:", options=all_regions, default=all_regions
)

selected_states = st.sidebar.multiselect(
    "States:", options=all_states, default=all_states
)

# Income slider
if "median_income" in df_base.columns and df_base["median_income"].notna().any():
    min_income = int(df_base["median_income"].min())
    max_income = int(df_base["median_income"].max())
    income_range = st.sidebar.slider(
        "Median Income Range ($)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        step=1000,
    )
else:
    income_range = (None, None)

# Policy slider if available
if "policy" in df_base.columns and df_base["policy"].notna().any():
    min_policy = float(df_base["policy"].min())
    max_policy = float(df_base["policy"].max())
    policy_range = st.sidebar.slider(
        "Policy Score Range",
        min_value=float(min_policy),
        max_value=float(max_policy),
        value=(float(min_policy), float(max_policy)),
        step=0.5,
    )
else:
    policy_range = (None, None)

# Filter presets (implemented as overrides during filtering)
preset = st.sidebar.selectbox(
    "Presets",
    [
        "None",
        "High-Income States Only",
        "High-Policy States Only",
        "Low-Charger States (High gap)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("These filters apply to all visualization and EDA tabs.")

def apply_global_filters(df_in):
    df_out = df_in.copy()

    # Region
    df_out = df_out[df_out["region"].isin(selected_regions)]

    # States
    df_out = df_out[df_out["state"].isin(selected_states)]

    # Income
    if income_range[0] is not None and "median_income" in df_out.columns:
        df_out = df_out[
            df_out["median_income"].between(income_range[0], income_range[1])
        ]

    # Policy
    if policy_range[0] is not None and "policy" in df_out.columns:
        df_out = df_out[df_out["policy"].between(policy_range[0], policy_range[1])]

    # Presets override / refine
    if preset == "High-Income States Only" and "Income_Q" in df_out.columns:
        df_out = df_out[df_out["Income_Q"].isin(["Q3", "Q4 (High)"])]
    elif preset == "High-Policy States Only" and "Policy_Q" in df_out.columns:
        df_out = df_out[df_out["Policy_Q"].isin(["Q3", "Q4 (High)"])]
    elif preset == "Low-Charger States (High gap)" and "charger_gap" in df_out.columns:
        # top quartile of charger gap
        q75 = df_out["charger_gap"].quantile(0.75)
        df_out = df_out[df_out["charger_gap"] >= q75]

    return df_out

df_filtered = apply_global_filters(df_base)
guard_empty(df_filtered, "Try changing filters or presets.")

# ------------------------------------------------------------
# ML: Regression + Classification + PCA + Clustering (using df_knn, not filtered)
# ------------------------------------------------------------
# Prepare modelling data
model_df = df_knn.copy()

# Ensure key columns
if "EV_per_1000" not in model_df.columns and "EV_per_1000_pop" in model_df.columns:
    model_df["EV_per_1000"] = model_df["EV_per_1000_pop"]

# Build feature list robustly
feature_candidates = [
    "station_count",
    "Stations_per_100k",
    "Stations_per_100k_pop",
    "median_income",
    "policy",
    "renewable_share",
    "bev_share_latest",
    "phev_share_latest",
    "charger_gap",
]
feature_cols_reg = [c for c in feature_candidates if c in model_df.columns]

target_col_reg = "EV_per_1000" if "EV_per_1000" in model_df.columns else None

reg_models = {}
reg_metrics = []
clf_models = {}
pca_model = None
kmeans_model = None

# Default: assume no ML (so other tabs can degrade gracefully)
reg_metrics_df = pd.DataFrame()
model_df["PC1"] = np.nan
model_df["PC2"] = np.nan
model_df["cluster"] = np.nan

if target_col_reg is not None and feature_cols_reg:
    # Drop rows with missing target or features
    model_df_model = model_df.dropna(subset=[target_col_reg] + feature_cols_reg).copy()

    # Guard: not enough rows for ML
    if model_df_model.shape[0] < 5:
        st.warning(
            f"⚠ Not enough complete rows for ML (have {model_df_model.shape[0]}, need ≥ 5). "
            "ML tabs will be disabled but the app will still run."
        )
    else:
        try:
            X = model_df_model[feature_cols_reg].astype(float)
            y = model_df_model[target_col_reg].astype(float)

            # Scaling for PCA & some models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA
            pca_model = PCA(n_components=2, random_state=42)
            X_pca = pca_model.fit_transform(X_scaled)
            model_df_model["PC1"] = X_pca[:, 0]
            model_df_model["PC2"] = X_pca[:, 1]

            # KMeans clustering on PC space
            try:
                kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
                model_df_model["cluster"] = kmeans_model.fit_predict(X_pca)
            except Exception:
                kmeans_model = None

            # Push PC1/PC2/cluster back onto full model_df (by state)
            for col in ["PC1", "PC2", "cluster"]:
                model_df[col] = model_df_model.set_index("state")[col].reindex(model_df["state"]).values

            # Regression train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            def eval_regressor(name, model):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
                reg_models[name] = model
                reg_metrics.append(
                    {
                        "model": name,
                        "R2_test": r2,
                        "MAE_test": mae,
                        "RMSE_test": rmse,
                        "CV5_R2": cv_r2,
                    }
                )
                return model

            # Fit core regressors
            eval_regressor("LinearRegression", LinearRegression())
            eval_regressor("Ridge(alpha=1.0)", Ridge(alpha=1.0))
            eval_regressor("Lasso(alpha=0.01)", Lasso(alpha=0.01, max_iter=10000))
            eval_regressor(
                "RandomForestRegressor",
                RandomForestRegressor(n_estimators=300, random_state=42),
            )
            eval_regressor(
                "GradientBoostingRegressor",
                GradientBoostingRegressor(random_state=42),
            )

            reg_metrics_df = pd.DataFrame(reg_metrics).sort_values(
                "R2_test", ascending=False
            )

            # Classification label: high vs low EV adoption
            threshold = model_df_model[target_col_reg].median()
            model_df_model["high_ev"] = (
                model_df_model[target_col_reg] >= threshold
            ).astype(int)

            X_clf = model_df_model[feature_cols_reg].astype(float)
            y_clf = model_df_model["high_ev"]

            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                X_clf, y_clf, test_size=0.25, stratify=y_clf, random_state=42
            )

            def eval_classifier(name, model):
                model.fit(Xc_train, yc_train)
                y_pred = model.predict(Xc_test)
                y_prob = (
                    model.predict_proba(Xc_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )
                acc = accuracy_score(yc_test, y_pred)
                f1 = f1_score(yc_test, y_pred, average="macro")
                clf_models[name] = {
                    "model": model,
                    "y_test": yc_test,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "accuracy": acc,
                    "f1_macro": f1,
                }
                return clf_models[name]

            eval_classifier("LogisticRegression", LogisticRegression(max_iter=1000))
            eval_classifier(
                "RandomForestClassifier",
                RandomForestClassifier(n_estimators=300, random_state=42),
            )

        except ValueError as e:
            # If sklearn chokes on shape / NaNs, don't crash the whole app
            st.warning(
                "⚠ Skipping ML block due to a data/shape issue passed to scikit-learn. "
                "Check that numeric columns exist and have non-missing values."
            )
            reg_metrics_df = pd.DataFrame()
            pca_model = None
            kmeans_model = None
            model_df["PC1"] = np.nan
            model_df["PC2"] = np.nan
            model_df["cluster"] = np.nan
else:
    # No valid features / target
    st.warning(
        "⚠ Could not find both EV_per_1000 and at least one feature column for ML. "
        "Predictive / clustering tabs will show limited content."
    )

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
(
    tab_home,
    tab_viewer,
    tab_dict,
    tab_quality,
    tab_geo,
    tab_trends,
    tab_corr,
    tab_regions,
    tab_models,
    tab_gap,
    tab_fairness,
    tab_interp,
    tab_about,
) = st.tabs(
    [
        "🏠 Home",
        "📁 Dataset Viewer",
        "📘 Data Dictionary",
        "🧹 Data Quality & Imputation",
        "🌍 Geographic View",
        "📈 Trends & Forecast",
        "🔗 Correlations & Multivariate",
        "🧭 Regional Profiles & Clusters",
        "🔮 Predictive Models",
        "⚡ Charger Gap & What-If",
        "⚖ Fairness & Bias",
        "🔍 Interpretability",
        "ℹ️ About & Architecture",
    ]
)

# ------------------------------------------------------------
# 🏠 0. Home
# ------------------------------------------------------------
with tab_home:
    st.markdown("You are here → **🏠 Home Overview**")
    st.info(
        "This page gives a quick overview of EV adoption, infrastructure, and why this analysis matters."
    )

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Why this matters")
        st.markdown(
            """
            Electric vehicles are not just a technology story — they’re also about **equity**, **infrastructure**, and **policy**.

            This dashboard helps you explore:
            - Which states lead in EV adoption and charger rollout  
            - How **income** and **policy support** shape EV accessibility  
            - Where **charger gaps** might be emerging  
            """
        )

    with colB:
        st.subheader("Key Snapshot")
        st.metric("States in view", df_filtered["state"].nunique())
        if "EV_Count" in df_filtered.columns:
            st.metric("Avg EV Count", f"{df_filtered['EV_Count'].mean():.0f}")
        if "station_count" in df_filtered.columns:
            st.metric("Avg Stations", f"{df_filtered['station_count'].mean():.0f}")
        if "median_income" in df_filtered.columns:
            st.metric("Median Income", f"${df_filtered['median_income'].median():,.0f}")

    st.markdown("---")
    st.subheader("Top 10 States by EVs per 1,000 Residents")

    if "EV_per_1000" in df_filtered.columns:
        top_ev = df_filtered.nlargest(10, "EV_per_1000")
        fig = px.bar(
            top_ev,
            x="state",
            y="EV_per_1000",
            hover_data=["EV_Count", "station_count", "median_income"],
            labels={"EV_per_1000": "EVs per 1,000 residents"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("EV_per_1000 not available; computed per-pop metrics may be in another file.")

    st.markdown("---")
    with st.expander("🧩 Quick summary report (text)", expanded=False):
        report_text = generate_summary_report(df_filtered)
        st.text_area("Summary Output", report_text, height=250)
        st.download_button(
            label="💾 Download EV Summary Report",
            data=report_text,
            file_name="EV_Analysis_Summary.txt",
            mime="text/plain",
        )

# ------------------------------------------------------------
# 📁 1. Dataset Viewer
# ------------------------------------------------------------
with tab_viewer:
    st.markdown("You are here → **📁 Dataset Viewer**")
    st.info("View raw/processed tables and download filtered copies.")

    dataset_choice = st.radio(
        "Choose dataset to view:",
        [
            "Filtered master (current view)",
            "Full master (all states)",
            "Modeling frame (df_knn)",
            "State-year EV stats (if available)",
        ],
        horizontal=True,
    )

    if dataset_choice == "Filtered master (current view)":
        df_view = df_filtered
    elif dataset_choice == "Full master (all states)":
        df_view = df_base
    elif dataset_choice == "Modeling frame (df_knn)":
        df_view = df_knn
    else:
        df_view = state_year_df if state_year_df is not None else pd.DataFrame()

    if df_view is None or df_view.empty:
        st.warning("No data available for this dataset.")
    else:
        st.write(f"Shape: **{df_view.shape[0]} rows × {df_view.shape[1]} columns**")
        st.dataframe(df_view.head(200))

        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download this dataset as CSV",
            data=csv_bytes,
            file_name="ev_dataset_export.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------
# 📘 2. Data Dictionary
# ------------------------------------------------------------
with tab_dict:
    st.markdown("You are here → **📘 Data Dictionary**")
    st.info("High-level reference of important variables and their meaning.")

    dict_entries = [
        ("state", "string", "State name", "Merged sources", "Standardized from input names"),
        ("state_usps", "string", "2-letter state code", "Derived", "Mapped from state name"),
        ("EV_Count", "numeric", "Total registered EVs", "EV registration dataset", "Cleaned & coerced numeric"),
        ("station_count", "numeric", "Number of public charging stations", "AFDC charging data", "Aggregated to state level"),
        ("population", "numeric", "Population (ACS)", "ACS", "Used for per-capita metrics"),
        ("EV_per_1000", "numeric", "EVs per 1,000 residents", "Derived", "EV_Count / population * 1000"),
        ("Stations_per_100k", "numeric", "Stations per 100k residents", "Derived", "station_count / population * 100000"),
        ("median_income", "numeric", "State median household income", "ACS", "Dollar values, cleaned & imputed"),
        ("policy", "numeric", "EV policy support score", "Policy dataset/laws", "Normalized score from laws/incentives"),
        ("renewable_share", "numeric", "Renewable energy share (%)", "SEDS", "Renewable / total energy * 100"),
        ("charger_gap", "numeric", "Gap between ideal and actual station counts", "Derived", "EV_Count/20 - station_count"),
        ("PC1", "numeric", "1st principal component", "PCA on features", "Linear combination of scaled predictors"),
        ("PC2", "numeric", "2nd principal component", "PCA on features", "Captures 2nd axis of variation"),
        ("cluster", "int", "Cluster ID", "K-means", "State grouping by multi-feature similarity"),
    ]

    dd_df = pd.DataFrame(
        dict_entries,
        columns=["Column", "Type", "Description", "Source", "Transformation / Notes"],
    )
    st.dataframe(dd_df)

# ------------------------------------------------------------
# 🧹 3. Data Quality & Imputation
# ------------------------------------------------------------
with tab_quality:
    st.markdown("You are here → **🧹 Data Quality & Imputation**")
    st.info("Shows missingness, simple imputation, and KNN-imputed frame (df_knn).")

    raw = df_base.copy()
    knn = df_knn.copy()

    # Missingness
    st.subheader("Missingness overview (raw master)")
    na_frac = raw.isna().mean().sort_values(ascending=False)
    st.dataframe(na_frac.to_frame("missing_fraction").style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    na_frac[na_frac > 0].plot(kind="bar", ax=ax)
    ax.set_ylabel("Fraction missing")
    ax.set_title("Missingness by column (non-zero only)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Simple vs KNN imputation ✨")
    st.markdown(
        """
        - **Simple imputation**: median for numeric, mode for categorical.  
        - **KNN imputation**: each missing value inferred from similar states.  
        """
    )

    st.write("Shape of KNN-imputed frame:", knn.shape)
    st.write("Any NAs left in df_knn?", bool(knn.isna().any().any()))

# ------------------------------------------------------------
# 🌍 4. Geographic View (Map + Bubble)
# ------------------------------------------------------------
with tab_geo:
    st.markdown("You are here → **🌍 Geographic View**")
    st.info("Map-based view of EV adoption and charging infrastructure.")

    metric_for_color = st.selectbox(
        "Metric for map color:",
        [m for m in ["EV_Count", "EV_per_1000", "station_count", "Stations_per_100k", "charger_gap"] if m in df_filtered.columns],
    )

    fig_map = px.choropleth(
        df_filtered,
        locations="state_usps",
        locationmode="USA-states",
        scope="usa",
        color=metric_for_color,
        color_continuous_scale="Viridis",
        hover_data=[
            c for c in ["state", "EV_Count", "EV_per_1000", "station_count", "Stations_per_100k", "median_income", "policy"]
            if c in df_filtered.columns
        ],
    )
    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Bubble view: EV vs Charging Infrastructure")
    if {"EV_per_1000", "Stations_per_100k"}.issubset(df_filtered.columns):
        fig_bubble = px.scatter(
            df_filtered,
            x="Stations_per_100k",
            y="EV_per_1000",
            size="EV_Count" if "EV_Count" in df_filtered.columns else None,
            color="region",
            hover_name="state",
            hover_data=["median_income", "policy"] if "policy" in df_filtered.columns else ["median_income"],
            labels={
                "Stations_per_100k": "Stations per 100k residents",
                "EV_per_1000": "EVs per 1,000 residents",
            },
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("Per-capita metrics not available for bubble plot.")

# ------------------------------------------------------------
# 📈 5. Trends & Forecast
# ------------------------------------------------------------
with tab_trends:
    st.markdown("You are here → **📈 Trends & Forecast**")
    st.info("Year-over-year EV growth and a simple forecast where data is available.")

    if state_year_df is None or state_year_df.empty:
        st.warning(
            "State-year EV stats file not found (`ev_state_year_stats.csv`). "
            "Run your notebook merge script to generate it and place in `data/processed/`."
        )
    else:
        # Expect columns: state, model_year (or year), n_vehicles
        year_col = "model_year" if "model_year" in state_year_df.columns else "year"
        val_col = "n_vehicles" if "n_vehicles" in state_year_df.columns else "EV_Count"

        states_for_trend = st.multiselect(
            "Select states to visualize:",
            options=sorted(state_year_df["state"].unique()),
            default=["California", "Texas", "Florida"]
            if "California" in state_year_df["state"].values
            else sorted(state_year_df["state"].unique())[:3],
        )

        trend_df = state_year_df[state_year_df["state"].isin(states_for_trend)]

        if trend_df.empty:
            st.warning("No data for selected states.")
        else:
            fig_trend = px.line(
                trend_df,
                x=year_col,
                y=val_col,
                color="state",
                markers=True,
                labels={year_col: "Year", val_col: "Registered EVs"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Simple forecast for one chosen state
        st.markdown("### Simple linear forecast for a single state")
        state_for_forecast = st.selectbox(
            "State for forecast:",
            options=sorted(state_year_df["state"].unique()),
        )
        sub = state_year_df[state_year_df["state"] == state_for_forecast].dropna(subset=[year_col, val_col])
        if len(sub) >= 3:
            X_year = sub[year_col].values.reshape(-1, 1)
            y_ev = sub[val_col].values
            lr = LinearRegression()
            lr.fit(X_year, y_ev)

            future_years = np.arange(sub[year_col].max() + 1, sub[year_col].max() + 6)
            y_pred = lr.predict(future_years.reshape(-1, 1))

            fig_fore = go.Figure()
            fig_fore.add_trace(
                go.Scatter(x=sub[year_col], y=y_ev, mode="lines+markers", name="Historical")
            )
            fig_fore.add_trace(
                go.Scatter(
                    x=future_years,
                    y=y_pred,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(dash="dash"),
                )
            )
            fig_fore.update_layout(
                xaxis_title="Year",
                yaxis_title="Registered EVs",
                height=400,
            )
            st.plotly_chart(fig_fore, use_container_width=True)
            st.caption("Simple linear trend forecast (for illustration only, not policy advice).")
        else:
            st.info("Not enough years for a meaningful forecast.")

# ------------------------------------------------------------
# 🔗 6. Correlations & Multivariate
# ------------------------------------------------------------
with tab_corr:
    st.markdown("You are here → **🔗 Correlations & Multivariate**")
    st.info("Correlation matrix, regression relationships, and pairwise patterns.")

    num_cols = [c for c in ["EV_Count","EV_per_1000","station_count","Stations_per_100k","median_income","policy","renewable_share"] if c in df_filtered.columns]
    if len(num_cols) >= 2:
        corr = df_filtered[num_cols].corr()
        st.subheader("Correlation matrix")
        st.dataframe(corr.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Pairwise scatter (small pairplot)")
        pair_cols = num_cols[:4]
        pair_df = df_filtered.dropna(subset=pair_cols)
        if not pair_df.empty:
            sns.set(style="whitegrid")
            fig_pair = sns.pairplot(pair_df[pair_cols])
            st.pyplot(fig_pair.fig)
            plt.close(fig_pair.fig)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

# ------------------------------------------------------------
# 🧭 7. Regional Profiles & Clusters
# ------------------------------------------------------------
with tab_regions:
    st.markdown("You are here → **🧭 Regional Profiles & Clusters**")
    st.info("Compare states by multi-dimensional EV profiles and clusters.")

    if "PC1" not in model_df.columns or model_df["PC1"].isna().all():
        st.warning("PCA / clusters not available (missing key columns). Check data or rerun notebook preprocessing.")
    else:
        # Radar chart for selected states
        radar_vars = [c for c in ["EV_per_1000","Stations_per_100k","median_income","policy","renewable_share","bev_share_latest"] if c in model_df.columns]
        st.subheader("Radar profiles for selected states")
        states_to_plot = st.multiselect(
            "Select states for radar chart:",
            options=sorted(model_df["state"].unique()),
            default=["California","Texas","Florida"] if "California" in model_df["state"].values else sorted(model_df["state"].unique())[:3],
        )
        radar_df = model_df[model_df["state"].isin(states_to_plot)].set_index("state")[radar_vars]
        if not radar_df.empty:
            # normalize 0-1
             # normalize 0–1 column-wise
             radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
            from math import pi
            labels = radar_vars
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            fig_r = plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)

            for state_name, row in radar_norm.iterrows():
                values = row.values.tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, label=state_name)
                ax.fill(angles, values, alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title("State EV Profile Radar Chart", y=1.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig_r)
            plt.close(fig_r)

        # PCA scatter with clusters
        st.subheader("PCA scatter with clusters")
        scatter_df = model_df.dropna(subset=["PC1", "PC2"])
        
        if not scatter_df.empty:
            # Build hover_data safely based on existing columns
            hover_candidates = [
                "EV_per_1000",
                "Stations_per_100k",
                "median_income",
                "policy",
                "EV_Count",
                "station_count",
            ]
            hover_cols = [c for c in hover_candidates if c in scatter_df.columns]
        
            fig_pca = px.scatter(
                scatter_df,
                x="PC1",
                y="PC2",
                color="cluster" if "cluster" in scatter_df.columns else None,
                hover_name="state" if "state" in scatter_df.columns else None,
                hover_data=hover_cols,
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                title="PCA of EV Adoption & Infrastructure Features by State",
            )
            fig_pca.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("No rows with valid PC1/PC2 to plot.")

# ------------------------------------------------------------
# 🔮 8. Predictive Models (Regression + Classification)
# ------------------------------------------------------------
with tab_models:
    st.markdown("You are here → **🔮 Predictive Models**")
    st.info("Regression (EV per 1,000) and classification (High vs Low EV adoption).")

    if reg_metrics_df.empty or not clf_models:
        st.warning("Models could not be fit. Check that EV_per_1000 and feature columns exist.")
    else:
        model_type = st.radio(
            "Choose model type:",
            ["Regression (EV_per_1000)", "Classification (High vs Low EV)"],
            horizontal=True,
        )

        if model_type.startswith("Regression"):
            st.subheader("Regression Model Performance")
            st.dataframe(reg_metrics_df.style.format({"R2_test": "{:.3f}", "MAE_test": "{:.2f}", "RMSE_test": "{:.2f}", "CV5_R2": "{:.3f}"}))

            # pick best by R2_test
            best_name = reg_metrics_df.iloc[0]["model"]
            best_model = reg_models[best_name]
            st.markdown(f"**Best model by test R²:** `{best_name}`")

            # Pred vs Actual plot
            X = model_df[feature_cols_reg].astype(float)
            y = model_df[target_col_reg].astype(float)
            _, X_test_reg, _, y_test_reg = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            y_pred_reg = best_model.predict(X_test_reg)
            fig_scatter = px.scatter(
                x=y_test_reg,
                y=y_pred_reg,
                labels={"x": "Actual EV_per_1000", "y": "Predicted EV_per_1000"},
                title="Predicted vs Actual (best regression model)",
            )
            fig_scatter.add_shape(
                type="line",
                x0=y_test_reg.min(),
                y0=y_test_reg.min(),
                x1=y_test_reg.max(),
                y1=y_test_reg.max(),
                line=dict(dash="dash"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.subheader("Classification Model Performance (High vs Low EV)")

            for name, info in clf_models.items():
                st.markdown(f"#### {name}")
                st.write(f"Accuracy: **{info['accuracy']:.3f}** | Macro F1: **{info['f1_macro']:.3f}**")
                st.text("Classification report:")
                st.text(classification_report(info["y_test"], info["y_pred"]))

                if info["y_prob"] is not None:
                    auc = roc_auc_score(info["y_test"], info["y_prob"])
                    st.write(f"ROC AUC: **{auc:.3f}**")

                    # Confusion matrix (show for RF only to avoid duplication)
                    if "RandomForest" in name:
                        cm = confusion_matrix(info["y_test"], info["y_pred"])
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax_cm)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        ax_cm.set_title("Confusion Matrix (RandomForest)")
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

# ------------------------------------------------------------
# ⚡ 9. Charger Gap & What-If
# ------------------------------------------------------------
with tab_gap:
    st.markdown("You are here → **⚡ Charger Gap & What-If**")
    st.info("Explore how changes in stations or policy could affect EV adoption.")

    if "charger_gap" not in df_filtered.columns or target_col_reg is None or "RandomForestRegressor" not in reg_models:
        st.warning("Charger gap or regression model not available. Check pre-processing and columns.")
    else:
        state_for_scenario = st.selectbox(
            "Select a state for scenario analysis:",
            options=sorted(df_filtered["state"].unique()),
        )
        base_row = df_filtered[df_filtered["state"] == state_for_scenario]
        guard_empty(base_row, "State not found in filtered dataset.")
        base_row = base_row.iloc[0]

        st.write("**Current baseline for this state:**")
        st.json(
            {
                "EV_per_1000": float(base_row.get("EV_per_1000", np.nan)),
                "station_count": float(base_row.get("station_count", np.nan)),
                "Stations_per_100k": float(base_row.get("Stations_per_100k", np.nan))
                if "Stations_per_100k" in base_row
                else None,
                "policy": float(base_row.get("policy", np.nan)),
                "charger_gap": float(base_row.get("charger_gap", np.nan)),
            }
        )

        st.markdown("### Adjust scenario inputs")
        new_station_mult = st.slider(
            "Increase stations by (%)",
            min_value=-50,
            max_value=200,
            value=0,
            step=10,
        )
        new_policy_add = st.slider(
            "Change policy score by (absolute points)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
        )

        # Build scenario row in feature space
        scenario = base_row.copy()
        if "station_count" in scenario:
            scenario["station_count"] = scenario["station_count"] * (1 + new_station_mult / 100.0)
        if "policy" in scenario:
            scenario["policy"] = scenario["policy"] + new_policy_add

        # Use best regression model (RFReg if available)
        rf_name = None
        for n in reg_models.keys():
            if "RandomForestRegressor" in n:
                rf_name = n
                break
        if rf_name is None:
            rf_name = list(reg_models.keys())[0]
        best_model = reg_models[rf_name]

        scenario_X = []
        for c in feature_cols_reg:
            scenario_X.append(float(scenario.get(c, np.nan)))
        scenario_X = np.array(scenario_X, dtype=float).reshape(1, -1)

        if np.isnan(scenario_X).any():
            st.warning("Scenario feature vector has NaNs; check that all feature columns exist for this state.")
        else:
            predicted_ev = best_model.predict(scenario_X)[0]
            baseline_ev = float(base_row.get("EV_per_1000", np.nan))

            st.markdown("### Scenario result")
            col1, col2 = st.columns(2)
            col1.metric("Baseline EV per 1,000", f"{baseline_ev:.2f}")
            col2.metric("Scenario EV per 1,000", f"{predicted_ev:.2f}", f"{predicted_ev - baseline_ev:+.2f}")

            if "station_count" in scenario:
                ideal_station_count = base_row["EV_Count"] / 20.0 if "EV_Count" in base_row else np.nan
                new_charger_gap = ideal_station_count - scenario["station_count"]
                st.write(
                    f"Ideal stations (1 per 20 EVs): **{ideal_station_count:.1f}**; "
                    f"Scenario stations: **{scenario['station_count']:.1f}**, "
                    f"New charger gap: **{new_charger_gap:.1f}**"
                )

            st.caption(
                "This is a model-based what-if; it captures correlations, not causal effects. "
                "Treat it as exploratory, not prescriptive planning."
            )

# ------------------------------------------------------------
# ⚖ 10. Fairness & Bias
# ------------------------------------------------------------
with tab_fairness:
    st.markdown("You are here → **⚖ Fairness & Bias**")
    st.info("Group-wise model performance across regions, income, and policy quartiles.")

    if not clf_models or "RandomForestClassifier" not in clf_models:
        st.warning("Classification model not available.")
    else:
        rf_info = clf_models["RandomForestClassifier"]
        rf_model = rf_info["model"]

        # we re-use full model_df with high_ev label
        if "high_ev" not in model_df.columns:
            st.warning("Label high_ev not found; check modelling setup.")
        else:
            group_options = []
            if "region" in model_df.columns: group_options.append("region")
            if "Income_Q" in model_df.columns: group_options.append("Income_Q")
            if "Policy_Q" in model_df.columns: group_options.append("Policy_Q")

            if not group_options:
                st.info("No grouping columns (region/income/policy quartiles) available.")
            else:
                group_by = st.selectbox("Group fairness by:", group_options)

                # Evaluate on all states (not split)
                X_all = model_df[feature_cols_reg].astype(float)
                y_all = model_df["high_ev"].astype(int)
                y_pred_all = rf_model.predict(X_all)

                model_df["pred_high_ev"] = y_pred_all

                rows = []
                for g, sub in model_df.groupby(group_by):
                    if len(sub) < 3:
                        continue
                    acc_g = accuracy_score(sub["high_ev"], sub["pred_high_ev"])
                    f1_g = f1_score(sub["high_ev"], sub["pred_high_ev"])
                    rows.append({"group": str(g), "n_states": len(sub), "accuracy": acc_g, "f1": f1_g})

                if rows:
                    fairness_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
                    st.dataframe(fairness_df.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}"}))

                    fig_bar = px.bar(
                        fairness_df,
                        x="group",
                        y="f1",
                        hover_data=["n_states","accuracy"],
                        title=f"F1 score by {group_by}",
                        labels={"f1":"F1 score","group":group_by},
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Not enough states per group to compute fairness metrics.")

        st.markdown("---")
        st.subheader("Ethical & Real-World Considerations")
        st.markdown(
            """
            - **Misclassification impact:** Over- or under-estimating EV adoption may lead to **overbuilt** or **underbuilt** charging infrastructure.  
            - **Data limitations:** Models operate on **state-level aggregates** and ignore within-state disparities.  
            - **Non-policy disclaimer:** This tool is for **educational and exploratory purposes only**, not official forecasts or investment advice.  
            """
        )

# ------------------------------------------------------------
# 🔍 11. Interpretability
# ------------------------------------------------------------
with tab_interp:
    st.markdown("You are here → **🔍 Interpretability**")
    st.info("Feature importance from tree-based models and PCA structure.")

    if reg_models and "RandomForestRegressor" in reg_models:
        rf_reg = reg_models["RandomForestRegressor"]
        if hasattr(rf_reg, "feature_importances_"):
            st.subheader("Random Forest Regressor — feature importance")
            importances = rf_reg.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_cols_reg, "importance": importances}).sort_values("importance", ascending=False)
            st.dataframe(fi_df)

            fig_fi = px.bar(fi_df, x="feature", y="importance", title="Regressor feature importance")
            st.plotly_chart(fig_fi, use_container_width=True)

    if clf_models and "RandomForestClassifier" in clf_models:
        rf_clf = clf_models["RandomForestClassifier"]["model"]
        if hasattr(rf_clf, "feature_importances_"):
            st.subheader("Random Forest Classifier — feature importance")
            importances_c = rf_clf.feature_importances_
            fic_df = pd.DataFrame({"feature": feature_cols_reg, "importance": importances_c}).sort_values("importance", ascending=False)
            st.dataframe(fic_df)

            fig_fic = px.bar(fic_df, x="feature", y="importance", title="Classifier feature importance")
            st.plotly_chart(fig_fic, use_container_width=True)

    if pca_model is not None:
        st.subheader("PCA Explained Variance")
        evr = pca_model.explained_variance_ratio_
        pca_df = pd.DataFrame({"PC": ["PC1","PC2"], "explained_variance_ratio": evr})
        st.dataframe(pca_df.style.format({"explained_variance_ratio": "{:.3f}"}))

        fig_pca_var = px.bar(
            pca_df,
            x="PC",
            y="explained_variance_ratio",
            title="Explained variance by principal component",
        )
        st.plotly_chart(fig_pca_var, use_container_width=True)

# ------------------------------------------------------------
# ℹ️ 12. About & Architecture
# ------------------------------------------------------------
with tab_about:
    st.markdown("You are here → **ℹ️ About & Architecture**")

    st.markdown(
        """
        ## Project Overview  

        **Title:** Electric Vehicle (EV) Adoption and Charging Infrastructure Across U.S. States  
        **Course:** CMSE 830  
        **Developed by:** Kaveri Palicherla  

        ---
        ### Objectives  
        - Explore **where** EV adoption is strongest or weakest.  
        - Examine how **income, policy support, and renewable energy** relate to EV uptake.  
        - Identify potential **charger gaps** where infrastructure lags behind EV growth.  
        - Demonstrate a full **data-science + ML workflow** in a web app.

        ---
        ### Data Pipeline (Conceptual Architecture)  

        1. **Data Collection**  
           - EV registrations by state  
           - Public charging stations (AFDC)  
           - Income (ACS)  
           - Energy (SEDS)  
           - Policies / laws  

        2. **Data Wrangling & Cleaning**  
           - Type coercion, de-duplication  
           - Joining on state / state codes  
           - Creating `EV_per_1000`, `Stations_per_100k`, `charger_gap`  
           - Imputation (median/mode + KNN)  

        3. **Feature Engineering & Linear Algebra**  
           - Normalization & scaling  
           - PCA (PC1, PC2)  
           - Clustering with K-means  

        4. **Modeling**  
           - Regression: predict **EV_per_1000**  
           - Classification: **High vs Low** EV adoption  
           - Evaluation: R², MAE, RMSE, Accuracy, F1, ROC AUC  

        5. **Interpretability & Fairness**  
           - Feature importance (Random Forest)  
           - Group-wise performance by region, income, policy quartiles  

        6. **Deployment**  
           - Streamlit app with multi-tab navigation  
           - Interactive filtering, downloads, and what-if scenarios.  

        ---
        ### Tools & Libraries  

        - **Python**: pandas, numpy, scikit-learn, seaborn, matplotlib, plotly, streamlit  
        - **Stats & ML**: regression, classification, PCA, clustering, fairness slicing  
        - **Visualization**: choropleths, bubble maps, radar charts, dual-axis trends  

        ---
        ### Ethics & Disclaimer  

        - This application is built for **educational purposes**, not official forecasting.  
        - Models capture **correlations**, not causation.  
        - State-level data hides within-state inequalities and local context.  
        - Any policy implications must be interpreted cautiously and supplemented with domain expertise.

        ---
        ### Use of AI  

        Parts of this dashboard (code structure, wording) were assisted by AI tools  
        (ChatGPT, OpenAI, 2025). All content was reviewed and adapted manually.  

        """
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "This app integrates data wrangling, EDA, ML, fairness, and interpretability into a single EV adoption dashboard."
)
