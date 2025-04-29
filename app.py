import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# App title
st.title("📈 Stock Price Predictor using Linear Regression")

# Sidebar for user input
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Load data from Yahoo Finance
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data loaded. Please check the ticker symbol or date range.")
    st.stop()

# Display raw data
st.subheader("Raw Data")
st.dataframe(df.tail())

# Feature engineering
df["SMA_5"] = df["Close"].rolling(window=5).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# Features and target
X = df[["SMA_5", "SMA_20"]]
y = df["Target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Combine into a DataFrame
comparison_df = pd.DataFrame({
    "Date": X_test.index,
    "Actual": y_test.values,
    "Predicted": y_pred
}).set_index("Date")

# Display predictions
st.subheader("Prediction Results")
st.dataframe(comparison_df.tail())

# Plot predictions
st.subheader("📊 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(comparison_df.index, comparison_df["Actual"], label="Actual", color="blue")
ax.plot(comparison_df.index, comparison_df["Predicted"], label="Predicted", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{ticker} Price Prediction using Linear Regression")
ax.legend()
st.pyplot(fig)
