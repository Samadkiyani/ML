# app.py - Financial ML Platform with Rate Limit Protection
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Configure page
st.set_page_config(
    page_title="StockML Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2a4a7c; border-bottom: 2px solid #2a4a7c;}
    .stTextInput>div>div>input {background-color: #f0f2f6;}
    .stButton>button {background-color: #2a4a7c; color: white; border-radius: 5px;}
    .stAlert {border-left: 3px solid #2a4a7c;}
    .ticker-error {color: #dc3545; font-weight: bold;}
    .date-warning {color: #ffc107; background-color: #fff3cd; padding: 10px; border-radius: 5px;}
    .countdown {color: #e67e22; font-weight: bold; padding: 10px; border: 2px solid #e67e22; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_RETRIES = 3
BASE_DELAY = 8.0
JITTER = 4.0
MIN_DATA_POINTS = 30
COOLDOWN_PERIOD = 1800  # 30 minutes in seconds
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X...)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0)...',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X)...',
    'Mozilla/5.0 (Linux; Android 13; SM-S901B)...'
]

def validate_ticker(ticker):
    return re.match(r'^[A-Za-z.-]{1,10}$', ticker) is not None

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_download(ticker, start_date, end_date):
    for attempt in range(MAX_RETRIES):
        try:
            delay = (BASE_DELAY * (2 ** attempt)) + random.uniform(0, JITTER)
            with st.spinner(f"⏳ Strategic delay {delay:.1f}s..."):
                time.sleep(delay)

            df = yf.download(
                ticker,
                start=start_date - datetime.timedelta(days=3),
                end=end_date + datetime.timedelta(days=3),
                progress=False
            )

            if df.empty:
                raise ValueError("Empty response from server")

            return df.loc[start_date:end_date].reset_index()

        except Exception as e:
            if "YFRateLimitError" in str(e) or "429" in str(e):
                remaining_time = COOLDOWN_PERIOD - (attempt * 600)
                st.error(f"""
                🔥 Critical Rate Limit Hit!
                ⏲️ Recommended cooldown: {remaining_time//60} minutes
                🛠️ Try VPN / use CSV
                """)
                with st.empty():
                    for i in range(remaining_time, 0, -1):
                        st.markdown(f"<div class='countdown'>⏳ Retry in: {i//60:02d}:{i%60:02d}</div>", unsafe_allow_html=True)
                        time.sleep(1)
                    st.markdown("🟢 Ready to try again!")
                continue
            raise
    raise ValueError(f"Failed after {MAX_RETRIES} attempts")

def main():
    st.title("📈 Robust Stock Analysis Platform")
    st.markdown("---")
    st.markdown("""
    💡 **Rate Limit Tips**:
    - Use date ranges < 6 months
    - Avoid rapid reloads
    - Use CSV for heavy data
    """)

    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "CSV Upload"])
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Enter Stock Ticker:", "").strip().upper()
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])
        st.markdown("---")
        st.header("Model Settings")
        model_type = st.selectbox("Algorithm:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    st.header("1. Data Acquisition")
    if st.button("📥 Load Market Data"):
        if data_source == "Yahoo Finance":
            if not ticker or not validate_ticker(ticker):
                st.error("🛑 Invalid or empty ticker")
                return
            if start_date >= end_date:
                st.error("📅 Start date must be before end date")
                return
            try:
                with st.spinner("🔍 Validating ticker..."):
                    info = yf.Ticker(ticker).info
                    if not info.get('regularMarketPrice'):
                        raise ValueError("Invalid or delisted ticker")
                df = safe_download(ticker, start_date, end_date)
                if len(df) < MIN_DATA_POINTS:
                    raise ValueError("Insufficient data points")
                st.session_state.data = df
                st.success("✅ Data loaded successfully")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❗ Error: {str(e)}")
        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if not {'Date', 'Close'}.issubset(df.columns):
                        raise ValueError("CSV must include 'Date' and 'Close'")
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.data = df.sort_values('Date')
                    st.success("✅ CSV loaded successfully")
                except Exception as e:
                    st.error(f"CSV Error: {str(e)}")
            else:
                st.warning("⚠️ Upload a valid CSV")

    if st.session_state.data is not None:
        st.header("2. Data Preparation")
        df = st.session_state.data
        if st.button("🧼 Clean Data"):
            with st.spinner("Processing..."):
                original = len(df)
                df_clean = df.dropna()
                st.session_state.data = df_clean
                st.success(f"Cleaned {original - len(df_clean)} rows")
        st.subheader("Data Statistics")
        st.dataframe(df.describe())

    if st.session_state.data is not None and len(st.session_state.data) > 50:
        st.header("3. Feature Engineering")
        df = st.session_state.data
        if st.button("⚙️ Generate Features"):
            with st.spinner("Generating..."):
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = compute_rsi(df['Close'])
                df.dropna(inplace=True)
                st.session_state.data = df
                st.success("✅ Features added")

        st.header("4. Model Training")
        if st.button("🚀 Train Model"):
            with st.spinner("Training..."):
                df = st.session_state.data
                features = df[['SMA_20', 'SMA_50', 'RSI']]
                target = df['Close']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=test_size)

                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.session_state.model = model
                st.success(f"🎯 Model trained | MSE: {mse:.2f}, R²: {r2:.2f}")

        st.header("5. Forecast Visualization")
        if st.session_state.model:
            df = st.session_state.data
            features = df[['SMA_20', 'SMA_50', 'RSI']]
            X_scaled = StandardScaler().fit_transform(features)
            df['Predicted'] = st.session_state.model.predict(X_scaled)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], name='Predicted'))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
