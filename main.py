from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI()

# Enable CORS so your UI can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model and data
model = joblib.load('sarima_model.pkl')
history_df = pd.read_csv('cleaned_monthly_sales.csv')

@app.get("/")
def read_root():
    return {"Status": "API is Online", "Project": "Superstore Sales Forecasting"}

@app.get("/data/history")
def get_history():
    # Sending back historical data for the chart
    return history_df.to_dict(orient='records')

@app.get("/data/forecast")
def get_forecast(months: int = 12):
    # Perform prediction using the loaded SARIMA model
    forecast_obj = model.get_forecast(steps=months)
    forecast_values = forecast_obj.summary_frame()['mean']
    
    # Generate future dates
    last_date = pd.to_datetime(history_df['Order Date']).max()
    forecast_dates = pd.date_range(start=last_date, periods=months + 1, freq='MS')[1:]
    
    # Format the response for the UI
    forecast_data = [
        {"Order Date": str(date.date()), "Sales": round(value, 2)}
        for date, value in zip(forecast_dates, forecast_values)
    ]
    return forecast_data