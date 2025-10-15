# EV-Adoption-Dashboard
Streamlit dashboard analyzing EV adoption trends and charging infrastructure across U.S. states.
# Introduction
This project explores electric vehicle (EV) adoption trends across U.S. states and investigates how factors like charging infrastructure availability, median income, and policy incentives influence adoption rates.
The goal is to visualize and interpret regional disparities, identify patterns between EV adoption and accessibility, and develop an interactive dashboard that supports data-driven insights for policymakers, manufacturers, and researchers.
This work is inspired by my prior internship experience in the automotive domain, where understanding EV adoption metrics and infrastructure density was key to designing sustainable mobility solutions.
The midterm phase focuses on data cleaning, exploratory data analysis (EDA), and visualization, while the final phase will extend this into predictive modeling and forecasting using machine learning.

# Key Features

Multiple Data Sources: Integrated data from Kaggle, U.S. Department of Energy (AFDC), and U.S. Census Bureau.

Data Cleaning: Removal of duplicates, normalization of column names, and handling of missing values using imputation methods.

Data Integration: Combined datasets by state to correlate EV count, charger availability, and median income.

Visualizations (EDA):

(i) Histograms for EV distribution by state

(ii) Boxplots and bar charts for regional comparison

(iii) Scatter plots showing relationship between charging stations and EV count

(iv) Choropleth maps of EV adoption across the U.S.

(v)Correlation heatmaps for numeric variables

Derived Metrics: EV-to-Charger Ratio (EV_Count / Charging_Stations) to assess infrastructure adequacy.

Streamlit App: Interactive dashboard with sidebar filters (state, year, normalization) and multiple tabs for easy navigation.

All preprocessing and analysis steps are performed in two Jupyter notebooks:
# Notebook
/notebooks/data_cleaning.ipynb:	Data import, missing value treatment, duplicate removal, column standardization, and dataset merging.
/notebooks/EDA.ipynb:	Exploratory Data Analysis using visualizations and statistical summaries based on CMSE 830 class methods.
# Dataset
EV Population datset and charging dataset couldnt be uploaded to data folder due to size constraints 
Here are the link to the dataset 
EV Population Kaggle dataset : https://www.kaggle.com/datasets/utkarshx27/electric-vehicle-population-data
Alternate charging stations : https://afdc.energy.gov/stations#/find/nearest
# Key techniques applied:

Encoding categorical variables (region → numeric labels)

Imputation (mean and median)

Correlation matrix computation (df.corr())

Data scaling for visualization (normalization for comparative graphs)

# Exploratory Data Analysis Results
Statistical Summaries

Descriptive statistics computed for EV count, charger density, and income.

Found positive correlation between median income and EV adoption, with clustering of high-EV states in the West Coast region.

Visual Insights

Histogram: Distribution of EVs by state shows high skewness (California dominates).

Boxplot: Regional EV distribution highlights variability across income levels.

Scatter Plot: Strong linear trend between charging station count and EV adoption.

Choropleth Map: Geographic visualization showing adoption hotspots in CA, WA, NY, and TX.

Correlation Heatmap: Reveals multi-factor relationships between EV counts, charging stations, and economic factors.

# Things That Worked & Challenges
Worked Well :

(i) Clean integration of datasets and visualizations into Streamlit.

(ii) Interactive layout using st.sidebar(), st.tabs(), and st.plotly_chart().

(iii) Successful caching implementation for faster reloads.

Challenges :

(i) Minor discrepancies in dataset formatting across sources.

(ii) Processing large CSVs (charging stations dataset) required memory optimization.

(iii) Choropleth visualization took extra time to align state codes correctly.

# Conclusion & Future Work

The midterm phase establishes a strong foundation for understanding the relationship between infrastructure and EV adoption.
Key findings include:

EV adoption increases with income and charger density.

Infrastructure growth is uneven across regions.

EV-to-Charger ratio can serve as an indicator of charging network sufficiency.

In the final project, I plan to:

(i) Implement feature engineering (e.g., normalized EV growth rate, income-weighted charger density).

(ii) Apply machine learning models (e.g., Linear Regression, Random Forest) to predict EV adoption trends.

(iii) Include forecasting of EV count per state and simulate the impact of charger expansion.

# How to Run the App Locally
"
# Clone the repository
git clone https://github.com/kaveri555/EV-Adoption-Dashboard.git

# Navigate into the project folder
cd EV-Adoption-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app  "
streamlit run app/app.py


# Streamlit App (Deployed Online)

Live Dashboard: https://share.streamlit.io/kaveri555/EV-Adoption-Dashboard

# Folder Structure  
EV-Adoption-Dashboard/
├── data/
│   ├── raw/               # Unprocessed datasets
│   └── processed/         # Cleaned and merged datasets
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   └── EDA.ipynb
│
├── app/
│   └── app.py             # Streamlit dashboard
│
├── README.md
└── requirements.txt
 


# Acknowledgments
Use of ChatGPT for suggestions on Streamlit app layout, interactive feature design, and overall content organization, including guidance on structuring the dashboard into logical sections and improving user documentation (OpenAI chatgpt-5 , 2025)

# References

Kaggle: Electric Vehicle Population Data

U.S. Department of Energy (AFDC): Alternative Fueling Station Data

U.S. Census Bureau: Median Income Dataset

OpenAI (2025). GPT-5 model responses used for Streamlit and documentation structure.


