import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Loading dataset...")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "Sample - Superstore 3.csv",
    encoding="latin1"
)

print("Dataset Shape:", df.shape)

# =====================================================
# DATA CLEANING
# =====================================================

df.drop_duplicates(inplace=True)

df["Order Date"] = pd.to_datetime(df["Order Date"])

# =====================================================
# DAILY SALES
# =====================================================

daily_sales = (
    df.groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

daily_sales["Year"] = daily_sales["Order Date"].dt.year
daily_sales["Month"] = daily_sales["Order Date"].dt.month
daily_sales["Day"] = daily_sales["Order Date"].dt.day
daily_sales["Weekday"] = daily_sales["Order Date"].dt.dayofweek
daily_sales["Quarter"] = daily_sales["Order Date"].dt.quarter

# =====================================================
# GRAPH 1
# HISTORICAL SALES TREND
# =====================================================

plt.figure(figsize=(14,6))

plt.plot(
    daily_sales["Order Date"],
    daily_sales["Sales"],
    linewidth=2
)

plt.title(
    "Historical Daily Sales Trend",
    fontsize=16
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "01_historical_sales_trend.png",
    dpi=300
)

plt.close()

print("Generated: 01_historical_sales_trend.png")

# =====================================================
# GRAPH 2
# MONTHLY SALES
# =====================================================

monthly_sales = (
    daily_sales
    .groupby("Month")["Sales"]
    .mean()
)

plt.figure(figsize=(10,6))

monthly_sales.plot(
    kind="bar"
)

plt.title(
    "Average Monthly Sales",
    fontsize=16
)

plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.savefig(
    "02_monthly_sales.png",
    dpi=300
)

plt.close()

print("Generated: 02_monthly_sales.png")

# =====================================================
# GRAPH 3
# YEARLY SALES
# =====================================================

yearly_sales = (
    daily_sales
    .groupby("Year")["Sales"]
    .sum()
)

plt.figure(figsize=(10,6))

yearly_sales.plot(
    kind="bar"
)

plt.title(
    "Yearly Sales",
    fontsize=16
)

plt.xlabel("Year")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    "03_yearly_sales.png",
    dpi=300
)

plt.close()

print("Generated: 03_yearly_sales.png")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split = int(len(daily_sales) * 0.8)

train = daily_sales[:split]
test = daily_sales[split:]

features = [
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Quarter"
]

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]

# =====================================================
# MODEL
# =====================================================

print("Training model...")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nMODEL PERFORMANCE")
print("----------------------")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2  :", round(r2,4))

# =====================================================
# GRAPH 4
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(15,6))

plt.plot(
    test["Order Date"],
    y_test,
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    test["Order Date"],
    predictions,
    label="Predicted Sales",
    linewidth=2
)

plt.title(
    "Actual vs Predicted Sales",
    fontsize=16
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "04_actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("Generated: 04_actual_vs_predicted.png")

# =====================================================
# FUTURE FORECAST
# =====================================================

future_dates = pd.date_range(
    start=daily_sales["Order Date"].max()
    + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_df = pd.DataFrame()

future_df["Date"] = future_dates

future_df["Year"] = future_df["Date"].dt.year
future_df["Month"] = future_df["Date"].dt.month
future_df["Day"] = future_df["Date"].dt.day
future_df["Weekday"] = future_df["Date"].dt.dayofweek
future_df["Quarter"] = future_df["Date"].dt.quarter

future_predictions = model.predict(
    future_df[
        [
            "Year",
            "Month",
            "Day",
            "Weekday",
            "Quarter"
        ]
    ]
)

forecast = pd.DataFrame(
    {
        "Date": future_dates,
        "Predicted Sales": future_predictions
    }
)

forecast.to_csv(
    "future_forecast.csv",
    index=False
)

# =====================================================
# GRAPH 5
# FUTURE FORECAST
# =====================================================

plt.figure(figsize=(15,6))

plt.plot(
    forecast["Date"],
    forecast["Predicted Sales"],
    marker="o",
    linewidth=2
)

plt.title(
    "30 Day Sales Forecast",
    fontsize=16
)

plt.xlabel("Date")
plt.ylabel("Predicted Sales")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "05_future_forecast.png",
    dpi=300
)

plt.close()

print("Generated: 05_future_forecast.png")

# =====================================================
# FORECAST SUMMARY
# =====================================================

avg_sales = round(
    forecast["Predicted Sales"].mean(),
    2
)

print("\nBUSINESS FORECAST SUMMARY")
print("-------------------------")

print(
    f"Average predicted daily sales "
    f"for next 30 days: {avg_sales}"
)

print("\nBusiness Use Cases:")
print("1. Inventory Planning")
print("2. Revenue Forecasting")
print("3. Staff Scheduling")
print("4. Demand Estimation")

print("\nForecast saved as future_forecast.csv")

print("\nPROJECT COMPLETE")