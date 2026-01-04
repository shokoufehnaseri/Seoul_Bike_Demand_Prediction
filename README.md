# Seoul Bike Sharing Demand Prediction (Hourly)

## Overview
Rental bikes must be available at the right time to reduce waiting and meet demand. This project predicts the **number of bikes rented per hour** using weather and time-based features and compares multiple machine learning models to find the best-performing approach.

## Dataset
- **Target:** Rented Bike Count (hourly)
- **Size:** 11,680 rows × 24 columns :contentReference[oaicite:1]{index=1}
- **Feature groups:** :contentReference[oaicite:2]{index=2}  
  - **Weather:** Temperature, Humidity, Wind speed, Visibility, Dew point, Solar Radiation, Snowfall, Rainfall  
  - **Time:** Hour, Day, Month, Year, Weekday, Holiday/Working day  
  - **Engineered:** feat01–feat08, feat10  

## Methodology
A complete methodology walkthrough is provided in the project presentation: `Methodology.pdf`. :contentReference[oaicite:3]{index=3}

Main steps:
1. **EDA**
   - Demand patterns across hours, months, and seasons :contentReference[oaicite:4]{index=4}
2. **Preprocessing**
   - Removed `id`, extracted time features from `Date`, encoded categorical features, and applied MinMax scaling :contentReference[oaicite:5]{index=5}
3. **Feature Selection**
   - Used **Boruta** (Random Forest wrapper) to select relevant predictors :contentReference[oaicite:6]{index=6}
4. **Modeling**
   - Decision Tree Regression (with GridSearchCV tuning) :contentReference[oaicite:7]{index=7}
   - Bagging Regressor (tuned with GridSearchCV and RandomizedSearchCV) :contentReference[oaicite:8]{index=8}
   - Gradient Boosting Regressor (tuned with GridSearchCV) :contentReference[oaicite:9]{index=9}
   - Neural Network (tested across epochs/batch size/learning rate) :contentReference[oaicite:10]{index=10}
5. **Evaluation Metrics**
   - RMSE, MAE, MAPE, R² :contentReference[oaicite:11]{index=11}

## Results (Summary)
Based on the final comparison in the presentation, the **tuned Bagging Regressor** achieved the best balance of training/testing performance with the **lowest RMSE (~169.23)** and **highest R² (~0.92)**. :contentReference[oaicite:12]{index=12}

## Project Files
- `Methodology.pdf` — full presentation (dataset, workflow, model tuning, results) :contentReference[oaicite:13]{index=13}
- Notebook / scripts — data processing, feature selection, model training, evaluation

## Tech Stack
Python: pandas, numpy, seaborn, matplotlib, scikit-learn, boruta, tensorflow/keras  
(Optionally: LightGBM, XGBoost for additional benchmarking)

## How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
