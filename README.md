## EV-Adoption-Dashboard

Streamlit dashboard analyzing EV adoption trends and charging infrastructure across U.S. states.

# Introduction:

This project explores Electric Vehicle (EV) adoption trends across U.S. states and investigates how factors like charging infrastructure availability, median income, and policy incentives influence adoption rates.
The goal is to visualize and interpret regional disparities, identify relationships between accessibility and adoption, and build an interactive dashboard for data-driven insights that can help policymakers, manufacturers, and researchers.

This work draws from my professional experience in the automotive domain, where understanding EV adoption and charging density was crucial for designing sustainable mobility solutions.
The midterm phase focuses on data collection, cleaning, exploratory data analysis (EDA), and visualization, while the final phase will extend this into predictive modeling and forecasting using machine learning.

# Why I Chose This Dataset:

I selected this topic because EV adoption analytics sits at the intersection of technology, economics, and sustainability — all core themes in modern automotive systems.
During my internship in an automotive company, regional EV data and infrastructure metrics were key in planning validation test cases and deployment feasibility.
This project allows me to apply real-world understanding to academic data science — connecting domain expertise with quantitative analysis.

# Key Features:

Multiple Data Sources were integrated from Kaggle, the U.S. Department of Energy (AFDC), and the U.S. Census Bureau.
Data Cleaning included removal of duplicates, handling missing values, and normalization of column names.
Data Integration combined datasets by state to correlate EV count, charger availability, and median income.
Derived Metrics such as EV-to-Charger Ratio (EV_Count / Charging_Stations) were computed to assess infrastructure adequacy.
Visualizations included histograms, boxplots, scatter plots, and choropleth maps that reveal geographic and economic trends.
The Streamlit App contains interactive sidebar filters, multiple tabs, and real-time chart updates for user exploration.

# Dataset :

The EV Population dataset and the charging dataset could not be uploaded directly to the repository due to file size constraints.
However, both datasets are publicly available and can be downloaded using the following links:

EV Population Dataset (Kaggle): https://www.kaggle.com/datasets/utkarshx27/electric-vehicle-population-data

Alternative Fuel Stations Dataset (U.S. Department of Energy AFDC): https://afdc.energy.gov/stations#/find/nearest

The median household income dataset was obtained from the U.S. Census Bureau (ACS S1903) and cleaned for state-level analysis.

# Preprocessing Steps Completed :

The project began with dataset inspection, exploring data types and missing values. Columns were normalized to maintain consistency across sources.
Invalid entries and duplicates were removed, and datasets were aggregated by state for meaningful comparisons.
The income dataset was cleaned to select only the relevant state and median income columns and converted from string to numeric values.
Finally, the datasets were merged on the state column to produce a unified dataset for visualization and further analysis.
The cleaned datasets were saved under the “data/processed” folder for reproducibility.

# What I Learned from IDA/EDA :

EDA helped uncover strong relationships between economic factors, infrastructure, and EV growth. Higher median income and charging infrastructure density showed a clear correlation with higher EV adoption.
Geographic analysis revealed that the West Coast, especially California and Washington, leads in adoption — supported by strong policy and infrastructure networks.
The distribution of EVs was highly skewed, showing that a few states dominate the national adoption trend.
Boxplots highlighted disparities in EV adoption across income groups, showing that wealthier states tend to have both more EVs and greater variation in adoption rates.
Overall, the analysis confirmed that EV growth is an interplay of technology readiness, affordability, and infrastructure investment.

# Exploratory Data Analysis Results :

Statistical summaries were generated for EV counts, charger density, and income levels, revealing a positive correlation between income and adoption.
The histogram (on a log scale) showed the skewed distribution where only a few states lead adoption.
The boxplot indicated variability among high-income states, showing that EV adoption is not just higher but also more diverse where income is greater.
The scatter plot confirmed a strong linear relationship between EV adoption and the number of charging stations.
Choropleth maps highlighted regional clusters of high adoption — particularly California, Washington, New York, and Texas.
A correlation heatmap visualized how economic and infrastructural indicators together drive adoption rates.

# What I’ve Tried with Streamlit So Far :

The Streamlit app includes a multi-tab layout separating EDA, Insights, and About sections.
Interactive filters allow users to select states, filter by year, and compare normalized metrics.
Plotly visualizations were integrated for dynamic interactivity and responsiveness.
Caching mechanisms using “st.cache_data()” were implemented to optimize reload speeds.
The layout and documentation structure were refined based on user experience testing and visual clarity.

# Things That Worked and Challenges :

The integration of multiple datasets and visualizations into Streamlit worked smoothly. The dashboard layout was intuitive and responsive, with interactive Plotly charts performing well.
The main challenges involved handling large CSV files, aligning state codes for maps, and cleaning inconsistencies between sources.
Memory optimization was required for the charging dataset due to its size, and mapping abbreviations with full state names took additional preprocessing time.

# Conclusion and Future Work :

The midterm phase establishes a strong foundation for understanding how infrastructure and socioeconomic factors influence EV adoption.
Key findings include that EV adoption rises with both income and charger density, but infrastructure growth is uneven across regions.
The EV-to-Charger ratio proved useful as an indicator of charging network sufficiency.
In the final project, I plan to engineer new features such as normalized EV growth rate and income-weighted charger density.
Machine learning models like Linear Regression and Random Forest will be implemented to predict adoption trends and forecast future EV growth by state.
The goal is to turn this analysis into a forecasting tool that can simulate the effect of infrastructure expansion.

# Folder Structure :

EV-Adoption-Dashboard/
├── data/
│ ├── raw/ (Unprocessed datasets)
│ └── processed/ (Cleaned and merged datasets)
│
├── notebooks/
│ ├── data_cleaning.ipynb
│ └── EDA.ipynb
│
├── app/
│ └── app.py (Streamlit dashboard)
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

# Streamlit App (Deployed Online)

Live Dashboard: https://share.streamlit.io/kaveri555/EV-Adoption-Dashboard

Use of ChatGPT (OpenAI GPT-5, 2025) for suggestions on Streamlit layout and logic, interactive feature design, and improving project documentation and structure.

# References

Kaggle: Electric Vehicle Population Data
U.S. Department of Energy (AFDC): Alternative Fueling Station Data
U.S. Census Bureau: Median Income Dataset
OpenAI (2025): GPT-5 Model Responses for Documentation Structuring
