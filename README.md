# Superstore Sales Forecasting 📈
A machine learning web application that predicts future sales trends using time-series forecasting.

## Project Overview
This project uses historical superstore data to build a predictive model. The backend is powered by **FastAPI**, and the forecasting logic is implemented using the **SARIMA** (Seasonal Autoregressive Integrated Moving Average) model.

## Features
- **Historical Data API:** Serves past sales data for visualization.
- **Forecast API:** Predicts future sales based on seasonal trends.
- **Model Persistence:** Uses `joblib` to load pre-trained SARIMA models for fast inference.

## Tech Stack
- **Backend:** FastAPI (Python)
- **ML Models:** SARIMA (Statsmodels), Joblib
- **Data Handling:** Pandas, NumPy
