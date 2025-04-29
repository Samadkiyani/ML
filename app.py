import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# --- Background GIF setup (optional) ---
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            data = file.read()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/gif;base64,{data.encode('base64').decode()}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found.")

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Financial Stock Predictor using Linear Regression")
st.markdown("---")

# --- Sidebar ---
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
sma_window = st.sidebar.slider("SMA Window", 5, 50, 20)

# --- Load Data ---
st.subheader("1. Load Yahoo Finance Data")
if st.button("Load Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found. Try different parameters.")
    else:
        st.success("Data loaded successfully.")
        st.dataframe(data.tail())

        # --- Preprocessing ---
        st.subheader("2. Preprocessing")
        data.dropna(inplace=True)
        st.write("After dropping missing values:", data.shape)

        # --- Feature Engineering ---
        st.subheader("3. Feature Engineering")
        data['SMA'] = data['Close'].rolling(window=sma_window).mean()
        data.dropna(inplace=True)
        st.line_chart(data[['Close', 'SMA']])

        # --- Train/Test Split ---
        st.subheader("4. Train/Test Split")
        data['Target'] = data['Close'].shift(-1)
        data.dropna(inplace=True)
        X = data[['SMA']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("Training Set:", X_train.shape, "Test Set:", X_test.shape)

        # --- Model Training ---
        st.subheader("5. Model Training")
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.success("Model trained successfully.")

        # --- Evaluation ---
        st.subheader("6. Evaluation")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("Mean Squared Error", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        # --- Results Visualization ---
        st.subheader("7. Results Visualization")
        result_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }, index=X_test.index)
        result_df.sort_index(inplace=True)
        fig = px.line(result_df, title="Actual vs Predicted Prices")
        st.plotly_chart(fig)

        # --- Download Results ---
        st.subheader("📥 Download Results")
        csv = result_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{ticker}_prediction.csv",
            mime='text/csv'
        )

        # --- Optional: Feature Importance ---
        st.subheader("📊 Feature Coefficient")
        st.write(f"SMA Coefficient: {model.coef_[0]:.4f}")
