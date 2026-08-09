# flood-risk-prototype
AI-based flood risk prediction prototype for South Australia
The Flood Risk Prototype is an AI-based flood risk prediction system designed to support early flood risk identification in South Australia.

The project aims to use historical and environmental data, including rainfall, river levels, weather conditions, and other relevant variables, to develop a machine learning model capable of estimating flood risk.

The system is designed as a prototype that integrates data processing, artificial intelligence, a backend API, and an interactive dashboard to provide accessible flood risk information.

## Problem Statement

Flooding is one of the most significant natural hazards affecting communities, infrastructure, and the environment. Traditional flood forecasting systems can be complex and may require significant technical infrastructure.

This project explores how Artificial Intelligence and Machine Learning can be used to analyse historical and environmental data to identify patterns associated with flood events and provide an accessible flood risk prediction tool.

The prototype focuses on South Australia and aims to support improved disaster preparedness, climate resilience, and informed decision-making.

## Project Architecture

The proposed system follows a modular architecture consisting of:

1. **Data Sources** – Historical flood, rainfall, weather, river level, and environmental data.
2. **Data Processing** – Data cleaning, preprocessing, feature engineering, and preparation for machine learning.
3. **AI/ML Model** – A machine learning model used to predict or classify flood risk.
4. **Backend API** – A FastAPI-based backend that connects the trained model with the application.
5. **Dashboard** – An interactive user interface for displaying flood risk predictions and relevant information.

### Architecture Diagram

The architecture diagram from the project report will be added here.

![Flood Risk Prediction System Architecture](docs/architecture-diagram.png)

## Repository Structure

    flood-risk-prototype/
    ├── data/          # Datasets and processed data
    ├── notebooks/     # Data analysis and machine learning experiments
    ├── backend/       # FastAPI backend and model integration
    ├── dashboard/     # User interface and visualisation dashboard
    ├── docs/          # Project documentation and architecture diagrams
    ├── .gitignore
    └── README.md

## Technologies

- Python
- Machine Learning / Artificial Intelligence
- FastAPI
- React
- Docker
- Cloud technologies

## Project Status

This project is currently under development as a prototype for an AI-based flood risk prediction system in South Australia.

## Model Evaluation
xychart-beta
    title "F1-Score Comparison"
    x-axis ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"]
    y-axis "F1 Score" 0.79 --> 0.81
    bar [0.800, 0.799, 0.802, 0.806]
Performance Summary
Model	F1	MCC	RMSE	Brier	NSE
Logistic Regression	0.800	0.741	0.300	0.090	0.505
Random Forest	0.799	0.734	0.264	0.070	0.618
XGBoost	0.802	0.738	0.276	0.076	0.582
Ensemble	0.806	0.743	0.271	0.074	0.596

The Ensemble model achieved the highest **F1-score (0.806)** and **Matthews Correlation Coefficient (0.743)**, indicating the best overall balance between precision and recall as well as the strongest classification performance across both classes.

Although **Random Forest** obtained the lowest **RMSE (0.264)**, the lowest **Brier Score (0.070)**, and the highest **Nash–Sutcliffe Efficiency (0.618)**, the Ensemble model was selected as the final deployment model because it provided the most balanced and reliable classification performance for flood-risk prediction.

The comparison also shows that all machine learning models outperformed the **Persistence Baseline**, confirming the value of the proposed AI-based approach for flood-risk assessment.

## Data & Licensing
The flood-risk prototype uses publicly available environmental and hydrological data relevant to South Australia, particularly the River Murray catchment region.

The primary sources include:

Bureau of Meteorology (BOM) – rainfall observations, weather measurements, and climate records.
South Australian Department for Environment and Water (DEW) – river and catchment information.
Australian Water Resources Assessment (AWRA) – soil moisture and hydrological indicators where available.
Rainfall and river level data sourced from the Bureau of Meteorology (BoM)
and the Murray-Darling Basin Authority (MDBA), © Commonwealth of Australia,
licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Only data that are publicly accessible for research, educational, and non-commercial purposes were used in this prototype.

Data Processing

The collected datasets were cleaned, standardized, and merged into a unified CSV dataset used for exploratory analysis, feature engineering, model training, and evaluation. Missing values were handled using interpolation or removal when appropriate, and temporal records were aligned to a common date format.

Licensing Considerations

BOM data are provided under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence unless otherwise stated. Users of the data must provide appropriate attribution to the Bureau of Meteorology.

This project is an academic prototype developed for educational purposes. No proprietary or confidential data were used, and all external datasets remain subject to their respective licences and terms of use.

## References

Bureau of Meteorology. (2026). Climate Data Online. Australian Government. https://www.bom.gov.au/climate/data/

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

ISO. (2023). ISO/IEC 23894:2023 Information technology — Artificial intelligence — Risk management. International Organization for Standardization.

OECD. (2019). OECD recommendation of the Council on Artificial Intelligence. OECD Publishing. https://oecd.ai/en/ai-principles

South Australian Department for Environment and Water. (2026). River Murray information. Government of South Australia. https://www.environment.sa.gov.au/
