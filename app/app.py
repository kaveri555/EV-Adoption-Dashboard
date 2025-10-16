# ============================================================
# EV-Adoption Dashboard
# Developed by Kaveri | CMSE 830
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------
# 1. Page Config & Styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Adoption Dashboard",
    layout="wide",
    #page_icon=""
)

# Custom CSS: clean layout + bounding boxes around tab headers
st.markdown("""
    <style>
        /* Layout & padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            zoom: 0.9;
        }
        footer {visibility: hidden;}

        /* Tab style boxes */
        [data-testid="stTabs"] button {
            border: 1px solid #0a9396 !important;
            border-radius: 5px !important;
            color: #0a9396 !important;
            font-weight: 600 !important;
            background-color: #f7f7f7 !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #0a9396 !important;
            color: white !important;
        }

        /* Metric spacing */
        div[data-testid="metric-container"] {
            margin-top: -10px;
            margin-bottom: -10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\palicher\Downloads\DATA\processed\ev_charging_income_state.csv")
    return df

merged_df = load_data()

# ------------------------------------------------------------
# 3. Sidebar (Minimal)
# ------------------------------------------------------------
with st.sidebar:
    st.image(r"C:\Users\palicher\Downloads\logo", width=150)
    st.title("EV-Adoption Dashboard")
    st.caption("Developed by Kaveri | CMSE 830")

# ------------------------------------------------------------
# 4. Page Title
# ------------------------------------------------------------
st.markdown("### EV Adoption Insights Across U.S. States")
st.markdown("---")

# ------------------------------------------------------------
# 5. Tabs (Clean headings)
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "EV Distribution",
    "Chargers vs Income",
    "Map Insights",
    "About"
])

# ------------------------------------------------------------
# TAB 1 — EV Distribution
# ------------------------------------------------------------
with tab1:
    st.subheader("EV Adoption Trends Across States")

    # --- Independent filters for this tab ---
    with st.expander("Filters"):
        states_ev = sorted(merged_df['state'].unique())
        selected_states_ev = st.multiselect(
            "Select States:",
            options=states_ev,
            default=states_ev[:10],
            key="ev_states"
        )
        min_income, max_income = int(merged_df['median_income'].min()), int(merged_df['median_income'].max())
        income_range_ev = st.slider(
            "Select Income Range ($)",
            min_value=min_income,
            max_value=max_income,
            value=(min_income, max_income),
            key="ev_income"
        )

    filtered_ev = merged_df[
        (merged_df['state'].isin(selected_states_ev)) &
        (merged_df['median_income'].between(income_range_ev[0], income_range_ev[1]))
    ]

    # --- KPI row ---
    total_evs = int(filtered_ev['EV_Count'].sum()) if not filtered_ev.empty else 0
    avg_income = filtered_ev['median_income'].mean()
    c1, c2 = st.columns(2)
    c1.metric("Total EVs", total_evs)
    c2.metric("Avg Income ($)", f"{int(avg_income):,}" if pd.notna(avg_income) else "N/A")

    st.markdown(" ")

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            filtered_ev.sort_values("EV_Count", ascending=False),
            x="state",
            y="EV_Count",
            color="EV_Count",
            color_continuous_scale="tealgrn",
            title="EV Adoption by State",
            labels={"EV_Count": "Number of EVs", "state": "State"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_box = px.box(
            filtered_ev,
            x="median_income",
            y="EV_Count",
            points="all",
            color_discrete_sequence=["#00CC96"],
            title="Income Variability vs EV Adoption",
            labels={"median_income": "Median Income ($)", "EV_Count": "EV Count"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    
    # --- EV Range Trend (Improved Line Chart) ---
    st.markdown("### EV Range and Adoption Trends Over Years")

    ev_data = pd.read_csv(r"C:\Users\palicher\Downloads\DATA\processed\ev_cleaned.csv")

    # Clean and ensure model_year is numeric
    ev_data = ev_data.dropna(subset=['model_year', 'electric_range'])
    ev_data['model_year'] = pd.to_numeric(ev_data['model_year'], errors='coerce')

    # Aggregate by state and year
    ev_trend = (
        ev_data.groupby(['state', 'model_year'])
        .agg({'electric_range': 'mean'})
        .reset_index()
    )

    # Filter only the states visible in the EV filter
    ev_trend = ev_trend[ev_trend['state'].isin(selected_states_ev)]

    # Create line chart for trend visualization
    fig_trend = px.line(
        ev_trend,
        x='model_year',
        y='electric_range',
        color='state',
        markers=True,
        title="EV Range Improvement Over Model Years by State",
        labels={
            'electric_range': 'Average Electric Range (miles)',
            'model_year': 'Model Year'
        },
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig_trend.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis=dict(range=[0, ev_trend['electric_range'].max() + 20]),
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(size=13),
        hovermode='x unified'
    )

    st.plotly_chart(fig_trend, use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — Chargers vs Income
# ------------------------------------------------------------
with tab2:
    st.subheader("Charging Infrastructure and EV Relationship")

    with st.expander("Filters"):
        states_ch = sorted(merged_df['state'].unique())
        selected_states_ch = st.multiselect(
            "Select States:",
            options=states_ch,
            default=states_ch[:10],
            key="ch_states"
        )

    filtered_ch = merged_df[merged_df['state'].isin(selected_states_ch)]

    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(
            filtered_ch,
            x="station_count",
            y="EV_Count",
            color="median_income",
            hover_name="state",
            title="EV Count vs Charging Stations",
            labels={
                "station_count": "Charging Stations",
                "EV_Count": "Number of EVs",
                "median_income": "Median Income ($)"
            },
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        corr = filtered_ch[["EV_Count", "station_count", "median_income"]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Correlation Between EVs, Chargers, and Income"
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ------------------------------------------------------------
# TAB 3 — Map Insights
# ------------------------------------------------------------
with tab3:
    st.subheader("Geospatial EV and Infrastructure Distribution")

    with st.expander("Filters"):
        selected_states_map = st.multiselect(
            "Select States:",
            options=sorted(merged_df['state'].unique()),
            default=sorted(merged_df['state'].unique())[:15],
            key="map_states"
        )

    filtered_map = merged_df[merged_df['state'].isin(selected_states_map)]

    col1, col2 = st.columns(2)

    with col1:
        fig_map = px.choropleth(
            filtered_map,
            locations="state",
            locationmode="USA-states",
            color="EV_Count",
            color_continuous_scale="plasma",
            title="EV Adoption Across the U.S.",
            scope="usa"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        fig_density = px.scatter(
            filtered_map,
            x="median_income",
            y="station_count",
            color="EV_Count",
            hover_name="state",
            color_continuous_scale="Plasma",
            title="Charger Availability vs Income Level",
            labels={"median_income": "Median Income ($)", "station_count": "Charging Stations"}
        )
        st.plotly_chart(fig_density, use_container_width=True)


# ------------------------------------------------------------
# TAB 4 — About
# ------------------------------------------------------------
with tab4:
    st.subheader("About This Project")
    st.markdown("""
    ### Project Overview  
    This dashboard analyzes **electric vehicle (EV) adoption trends** across U.S. states  
    using datasets from Kaggle, AFDC, and the U.S. Census Bureau.

    ### Key Features  
    - Independent filters for each visualization  
    - Clean, responsive two-column layout  
    - Animated EV range trends over years  
    - Minimal design for readability

    ### Developed by  
    **Kaveri | CMSE 830 | Michigan State University (2025)**
    """)
