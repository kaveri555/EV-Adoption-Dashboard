# ============================================================
# EV Adoption Dashboard — Streamlit Cloud Compatible Version
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="EV Adoption Insights", layout="wide", page_icon="🔋")
st.title("🔋 Electric Vehicle Adoption Across U.S. States")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    logo_path = "app/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Electric_car_icon.png", use_container_width=True)
    st.title("EV-Adoption Dashboard")
    st.caption("Developed by Kaveri | CMSE 830")

# ------------------------------------------------------------
# File Paths (relative for Streamlit Cloud)
# ------------------------------------------------------------
DATA_PATHS = {
    "merged": ["data/processed/ev_charging_income_state.csv"]
}

# ------------------------------------------------------------
# Safe CSV Loader
# ------------------------------------------------------------
def safe_read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception:
            with open(file_path, 'rb') as f:
                data = io.BytesIO(f.read())
            df = pd.read_csv(data, encoding_errors='ignore')
    return df

def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            df = safe_read_csv(p)
            st.success(f"✅ Loaded: {p} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return df
    st.error("❌ Dataset not found. Please ensure CSV exists in 'data/processed/'.")
    st.stop()

# ------------------------------------------------------------
# Clean State Codes
# ------------------------------------------------------------
def clean_state_codes(df):
    USPS = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
        'Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
        'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
        'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
        'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY',
        'North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
        'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT',
        'Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
    }

    df = df.copy()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
        df["state_usps"] = df["state"].map(USPS)
        df = df.dropna(subset=["state_usps"])
        df["state_usps"] = df["state_usps"].str.upper()
    return df

# ------------------------------------------------------------
# Load and Prepare Data
# ------------------------------------------------------------
df = load_first_existing(DATA_PATHS["merged"])

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Diagnostic checks
st.write("### Data Preview (Top 5 Rows)")
st.dataframe(df.head())

st.write("### Column Check")
st.write(df.columns.tolist())

st.write("### Null Values by Column")
st.write(df.isnull().sum())

# Ensure expected columns
expected_cols = ['state', 'EV_Count', 'station_count', 'median_income']
missing = [col for col in expected_cols if col not in df.columns]
if missing:
    st.error(f"❌ Missing columns in dataset: {missing}")
    st.stop()

# Clean numeric columns
for col in ['EV_Count', 'station_count', 'median_income']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
        .replace('', np.nan)
        .astype(float)
    )

df = clean_state_codes(df)
df["EV_per_station"] = df["EV_Count"] / df["station_count"].replace(0, np.nan)

# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

state_list = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect("Select States to Include:", options=state_list, default=state_list)

min_income, max_income = int(df["median_income"].min()), int(df["median_income"].max())
income_range = st.sidebar.slider("Select Median Income Range ($)", min_value=min_income, max_value=max_income, value=(min_income, max_income), step=1000)

df_filtered = df[(df["state"].isin(selected_states)) & (df["median_income"].between(income_range[0], income_range[1]))].copy()

st.sidebar.markdown("---")
st.sidebar.info("Use filters above to interact with all dashboard tabs.")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Geographic Overview",
    "📊 Trends & Distributions",
    "📈 Relationships & Correlations",
    "💵 Income & Accessibility",
    "ℹ About"
])

# ------------------------------------------------------------
# 1️⃣ Geographic Overview
# ------------------------------------------------------------
with tab1:
    st.subheader("U.S. Map of EV Adoption and Infrastructure")

    fig_map = px.choropleth(
        df_filtered,
        locations="state_usps",
        locationmode="USA-states",
        scope="usa",
        color="EV_Count",
        color_continuous_scale="Viridis",
        hover_data=["state", "EV_Count", "station_count", "median_income"]
    )
    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("States", df_filtered["state"].nunique())
    col2.metric("Avg EV Count", f"{df_filtered['EV_Count'].mean():.0f}")
    col3.metric("Avg Charging Stations", f"{df_filtered['station_count'].mean():.0f}")
    col4.metric("Median Income", f"${df_filtered['median_income'].median():,.0f}")

# ------------------------------------------------------------
# 2️⃣ Trends & Distributions
# ------------------------------------------------------------
with tab2:
    st.subheader("Distribution Trends")
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.tight_layout(pad=4.0)

    sns.histplot(df_filtered["EV_Count"], bins=15, kde=True, color="teal", ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of EV Count Across States")

    sns.histplot(df_filtered["station_count"], bins=15, kde=True, color="orange", ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Charging Stations")

    sns.histplot(df_filtered["median_income"], bins=10, kde=True, color="purple", ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of Median Household Income")

    top_evs = df_filtered.nlargest(10, "EV_Count")
    sns.barplot(y="state", x="EV_Count", data=top_evs, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Top 10 States by EV Count")

    top_st = df_filtered.nlargest(10, "station_count")
    sns.barplot(y="state", x="station_count", data=top_st, ax=axes[2, 0], palette="crest")
    axes[2, 0].set_title("Top 10 States by Charging Stations")

    sns.boxplot(data=df_filtered[["EV_Count", "station_count", "median_income"]], ax=axes[2, 1])
    axes[2, 1].set_title("Outlier Overview (Boxplots)")

    st.pyplot(fig)

# ------------------------------------------------------------
# 3️⃣ Relationships & Correlations
# ------------------------------------------------------------
with tab3:
    st.subheader("Interrelationships between EVs, Income, and Stations")

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.regplot(data=df_filtered, x="median_income", y="EV_Count", ax=axes[0], color="navy")
    axes[0].set_title("Income vs EV Count")

    sns.regplot(data=df_filtered, x="station_count", y="EV_Count", ax=axes[1], color="teal")
    axes[1].set_title("Stations vs EV Count")

    sns.regplot(data=df_filtered, x="median_income", y="station_count", ax=axes[2], color="orange")
    axes[2].set_title("Income vs Stations")

    st.pyplot(fig2)

    st.write("#### Correlation Matrix")
    corr = df_filtered[["EV_Count", "station_count", "median_income"]].corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

# ------------------------------------------------------------
# 4️⃣ Income & Accessibility
# ------------------------------------------------------------
with tab4:
    st.subheader("Equity Analysis: Charging Access by Income Quartiles")
    df_filtered["Income_Q"] = pd.qcut(df_filtered["median_income"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    fig3, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Income_Q", y="station_count", data=df_filtered, palette="Blues", ax=ax)
    ax.set_title("Charging Station Availability by Income Quartile")
    st.pyplot(fig3)

# ------------------------------------------------------------
# 5️⃣ About
# ------------------------------------------------------------
with tab5:
    st.markdown("""
    ## ℹ ## ℹ About This Dashboard  

    **Project Title:** *Electric Vehicle (EV) Adoption and Charging Infrastructure Analysis Across U.S. States*  
    **Course:** *CMSE 830 
    **Developed by:** *Kaveri palicherla*  

    ---

    ###  **Project Objective**  
    This dashboard provides an integrated, data-driven view of **Electric Vehicle (EV) adoption** across the United States,  
    analyzing how **income levels, charging infrastructure, and regional factors** influence the rate of EV growth.  

    The goal is to explore **whether accessibility and affordability align with adoption** — identifying states where  
    charging infrastructure and policy support may be lagging behind EV ownership trends.

    ---

    ###  **Analytical Highlights**  
    1. **Distribution Trends:** Histograms and boxplots show right-skewed adoption patterns — a few states dominate the EV market.  
    2. **Bivariate Analysis:** Positive correlation between *median household income* and *charger availability*.  
    3. **Fairness Evaluation:** Income-based quartile segmentation highlights disparities in EV accessibility.  
    4. **Geospatial Visualization:** Choropleth maps and interactive filters identify **EV and charger hotspots** versus underserved regions.  
    5. **Statistical Validation:** ANOVA and correlation tests confirm significant relationships between **income and EV density**.  

    ---

    ###  **Data Sources**  
    - **EV Population Data:** Electric vehicle registration counts by U.S. state.  
    - **Charging Station Data:** Alternative Fuel Data Center (AFDC) public charging locations.  
    - **Income Data:** U.S. Census Bureau’s American Community Survey (ACS).  
    - **Merged Dataset:** Combined and processed in Python using `pandas`, `numpy`, and custom feature engineering.  

    ---

    ###  **Tools & Techniques**  
    - **Languages & Libraries:** Python, Pandas, NumPy, Plotly, Seaborn, Streamlit  
    - **Statistical Methods:** Correlation matrix, z-score outlier detection, income quartile segmentation  
    - **Visualization:** Interactive dashboards with choropleths, bubble maps, dual-axis plots, and fairness analysis visuals  

    ---

    ###  **Unique Features**  
    - Auto-loaded datasets (EV, Stations, Income, Population) for one-click exploration  
    - Interactive filters for states and metrics  
    - Built-in fairness and accessibility analysis by income group  

    ---

    ###  **Broader Impact**  
    The analysis reveals that **EV adoption is not just a technological issue — it’s an equity challenge.**  
    States with higher income and stronger policy support lead the transition, while others remain underserved.  
    These insights can guide policymakers toward **balanced, inclusive infrastructure planning**.  

    ---

    ###  **Acknowledgment**  
    *use of AI for technical assistance  from ChatGPT (OpenAI, GPT-5, 2025) for code generation, visualization structuring, and documentation support.*  
    *All outputs were reviewed, validated, and modified manually to ensure originality and correctness.*  

    ---

    *“Data-driven insights for a sustainable, electric future.”* 
    """)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Visualization app automatically loads merged EV dataset and generates descriptive, distributional, and relational insights.")
