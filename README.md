# Introduction

This project explores Electric Vehicle (EV) adoption trends across U.S. states and investigates how factors like charging infrastructure availability, median income, population, and policy incentives influence adoption rates.
The goal is to visualize and interpret regional disparities, identify relationships between accessibility and adoption, and build an interactive dashboard for data-driven insights that can help policymakers, manufacturers, and researchers.

This work draws from my professional experience in the automotive domain, where understanding EV adoption and charging density was crucial for designing sustainable mobility solutions.
The midterm phase focuses on data collection, cleaning, exploratory data analysis (EDA), and visualization, while the final phase will extend this into predictive modeling and forecasting using machine learning.

Additionally, the dashboard integrates fairness and equity analysis, examining how income levels affect EV accessibility, and includes an automated summary generator that produces analytical insights directly from the dashboard interface.

# Why I Chose This Dataset

I selected this topic because EV adoption analytics sits at the intersection of technology, economics, and sustainability — all core themes in modern automotive systems.
During my internship in an automotive company, regional EV data and infrastructure metrics were key in planning validation test cases and deployment feasibility.
This project allows me to apply real-world understanding to academic data science — connecting domain expertise with quantitative analysis.

It also addresses a current industry gap: companies often lack an integrated dashboard that correlates EV counts, infrastructure, and socioeconomic factors.
The project demonstrates how such a tool could support strategic decision-making for equitable infrastructure expansion.

# Key Features

Multiple Data Sources integrated from Kaggle, the U.S. Department of Energy (AFDC), and the U.S. Census Bureau.

Data Cleaning included removal of duplicates, handling missing values, normalizing column names, and harmonizing state codes.

Data Integration combined datasets by state to correlate EV count, charger availability, population, and median income.

# Derived Metrics:

EV-to-Charger Ratio (EV_Count / station_count)

EVs per 1,000 residents (EV_per_1000_pop)

Charging Stations per 100,000 residents (Stations_per_100k_pop)

Fairness and Equity Analysis: Income quartiles were used to compare EV and charger density across socioeconomic groups.

Outlier Detection: Z-score–based detection highlighted states (e.g., California, Washington, Texas) that dominate national averages.

Statistical Validation: ANOVA and correlation analysis confirmed income’s significant effect on EV accessibility.

Advanced Visualizations: Pair plots, violin plots, correlation heatmaps, bubble maps, and dual-axis trend plots for dynamic comparison.

Automated Summary Generator: Creates an EV_Analysis_Summary.txt report from within the Streamlit app and allows users to download it.

Streamlit App: Interactive filters, multi-tab structure, and real-time updates for user exploration.

# Dataset

The EV Population dataset and the charging dataset could not be uploaded directly to the repository due to file size constraints.
However, both datasets are publicly available and can be downloaded using the following links:

EV Population Dataset (Kaggle): Electric Vehicle Population Data

Alternative Fuel Stations Dataset (AFDC): AFDC Station Data

Median Household Income Dataset: U.S. Census Bureau (ACS S1903).

All datasets were aggregated at the state level and merged into a unified dataset ev_charging_income_state.csv for analysis.

Preprocessing Steps Completed

Inspected and standardized data types and column names.

Removed invalid entries and duplicates.

Cleaned income data and converted values to numeric format.

Aggregated datasets by state and merged them using consistent naming conventions.

Created derived metrics (EV-to-Charger ratio, population-normalized rates).

Applied missing value imputation using mean/mode substitution for numerical and categorical columns.

Saved processed datasets under data/processed/ for reproducibility.

Generated summary statistics and IQR-based outlier flags to detect extreme values.

# What I Learned from IDA/EDA

Exploratory analysis revealed strong relationships among economic, infrastructural, and technological factors influencing EV growth.

Income vs EV adoption: Higher-income states showed significantly greater EV penetration.

Charging infrastructure correlation: States with more chargers had proportionally higher EV counts (r ≈ 0.9).

Geographic insights: California and Washington lead adoption; central and southern states lag behind.

Distribution shape: EV and charger counts were right-skewed — adoption is concentrated among a few states.

Fairness lens: Income quartile plots exposed inequitable charger access for low-income states.

Outliers: California, Washington, and Texas emerged as positive outliers due to large EV fleets and infrastructure.

EDA established that income, policy incentives, and charger density together drive EV adoption patterns.

# Exploratory Data Analysis Results

Descriptive statistics summarized EV, charger, and income distributions.

Histograms (log-scaled) revealed right-skewed distributions.

Boxplots exposed variance across states, confirming inequity in adoption.

Violin plots visualized charger and EV density across income quartiles.

Scatter plots demonstrated the positive correlation between chargers and EVs.

Choropleth maps displayed spatial clusters of EV dominance (CA, WA, NY, TX).

Pair plots and heatmaps highlighted correlated features and multivariate relationships.

Log transformation reduced skewness for better visualization and regression stability.

Fairness testing (ANOVA/t-test) confirmed statistically significant differences in adoption by income group.

# What I’ve Tried with Streamlit So Far

Implemented multi-tab layout separating Geographic Overview, Trends, Correlations, Equity, Summary, and About sections.

Added interactive sidebar filters for state and income range.

Integrated Plotly visualizations for dynamic, responsive exploration.

Added automatic summary generator tab displaying analysis text and download option.

Included global filter synchronization across all tabs for consistency.

Used Streamlit caching (st.cache_data) to optimize loading times.

Styled visualizations with Seaborn and Plotly themes for clear contrast and readability.

Incorporated regression trendlines and interactive correlation matrix heatmaps for advanced insight.

# Things That Worked and Challenges

# Worked Well:

Data merging, preprocessing, and feature engineering.

Multi-tab app organization and Plotly chart responsiveness.

Successful integration of fairness, statistical testing, and report generation.

Rich interactivity through filters and metrics panels.

# Challenges Faced:

Handling large CSVs and inconsistent state naming conventions.

Balancing color scales for dual maps.

Normalizing population and income data for fair comparisons.

Managing memory and runtime in Streamlit for large datasets.

# Conclusion and Future Work

The midterm phase built a complete visualization and equity analysis platform connecting infrastructure, economics, and policy to EV growth.
Key findings include:

EV adoption strongly correlates with income and charger availability.

Infrastructure growth remains uneven across regions.

Outlier states dominate national averages, skewing aggregate trends.

# Future extensions include:

Predictive modeling using Linear Regression and Random Forest to forecast future EV adoption by state.

Adding temporal data for year-wise EV growth visualization.

Integrating urban vs rural segmentation and population density metrics.

Creating a policy simulation tool to estimate adoption impact from charger expansion.

The goal is to evolve this dashboard into a forecasting and decision-support platform for sustainable mobility planning.

# Folder Structure
EV-Adoption-Dashboard/
├── data/
│   ├── raw/ (Unprocessed datasets)
│   └── processed/ (Cleaned and merged datasets)
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   └── EDA.ipynb
│
├── app/
│   └── app.py (Streamlit dashboard)
│
├── reports/
│   └── EV_Analysis_Summary.txt (auto-generated summary)
│
├── README.md
└── requirements.txt

# How to Run the App Locally
# Clone the repository
git clone https://github.com/kaveri555/EV-Adoption-Dashboard.git

# Navigate into the project folder
cd EV-Adoption-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py

Streamlit App (Deployed Online)

# Live Dashboard: 

The online dashboard allows users to explore EV adoption, charger density, income correlation, fairness insights, and a downloadable automated summary report.

# Use of ChatGPT

Use of ChatGPT (OpenAI GPT-5, 2025) for technical guidance on Streamlit layout, interactive logic, feature design, and improving documentation clarity and flow.
All AI-assisted outputs were verified, validated, and manually edited for originality and accuracy in compliance with academic integrity policies.

# References

Kaggle: Electric Vehicle Population Data

U.S. Department of Energy (AFDC): Alternative Fueling Station Data

U.S. Census Bureau: Median Income Dataset (ACS S1903)

OpenAI (2025): GPT-5 Model Responses for Documentation Structuring and Analysis Support
