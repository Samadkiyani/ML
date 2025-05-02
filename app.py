# app.py - Financial ML Platform (Closing Price Only)
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
    .stButton>button {background-color: #2a4a7c; color: white;}
    .error-box {padding: 15px; background-color: #ffe6e6; border: 1px solid #ffcccc;}
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_RETRIES = 3
BASE_DELAY = 10.0
JITTER = 5.0
MIN_DATA_POINTS = 30
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def validate_ticker(ticker):
    return re.match(r'^[A-Za-z.-]{1,10}$', ticker) is not None

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def safe_fetch_close(ticker, start_date, end_date):
    """Fetch only closing prices with robust error handling"""
    for attempt in range(MAX_RETRIES):
        try:
            delay = BASE_DELAY + random.uniform(0, JITTER)
            time.sleep(delay)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=start_date - datetime.timedelta(days=3),
                end=end_date + datetime.timedelta(days=3),
                interval='1d',
                actions=False
            )
            
            if hist.empty:
                raise ValueError("No data returned")
                
            close_prices = hist[['Close']].reset_index()
            close_prices = close_prices.loc[start_date:end_date]
            
            if len(close_prices) < MIN_DATA_POINTS:
                raise ValueError(f"Only {len(close_prices)} data points")
                
            return close_prices

        except Exception as e:
            if attempt == MAX_RETRIES-1:
                st.markdown(f"""
                <div class='error-box'>
                <h4>🚨 Failed to fetch {ticker}</h4>
                <p><strong>Reason:</strong> {str(e)}</p>
                <p><strong>Solutions:</strong></p>
                <ol>
                    <li>Verify ticker on Yahoo Finance</li>
                    <li>Try smaller date range (1-3 months)</li>
                    <li>Wait 5 minutes and try again</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)
            time.sleep(2 ** attempt)
    return None

def main():
    st.title("📈 Stock Closing Price Analyzer")
    st.markdown("---")
    
    # Session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        data_source = st.radio("Data Source", ["Yahoo Finance", "CSV Upload"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
            start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        st.markdown("---")
        model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    # Data loading
    st.header("1. Data Loading")
    if st.button("Load Closing Prices"):
        if data_source == "Yahoo Finance":
            if not validate_ticker(ticker):
                st.error("Invalid ticker format")
                return
                
            if start_date >= end_date:
                st.error("Invalid date range")
                return

            with st.spinner(f"Fetching {ticker} closing prices..."):
                df = safe_fetch_close(ticker, start_date, end_date)
                if df is not None:
                    st.session_state.data = df.rename(columns={'Close': 'Price'})
                    st.success(f"Loaded {len(df)} days of closing prices")
                    st.line_chart(df.set_index('Date')['Close'])

        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Close' not in df.columns:
                        raise ValueError("Missing 'Close' column")
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.data = df[['Date', 'Close']].rename(columns={'Close': 'Price'})
                    st.success("CSV loaded successfully")
                except Exception as e:
                    st.error(f"CSV Error: {str(e)}")

    # Data processing
    if st.session_state.data is not None:
        st.header("2. Data Processing")
        df = st.session_state.data
        
        if st.button("Clean & Transform"):
            with st.spinner("Processing..."):
                # Handle missing values
                df_clean = df.dropna()
                df_clean = df_clean[df_clean['Price'] > 0]
                
                # Feature engineering
                df_clean['SMA_20'] = df_clean['Price'].rolling(20).mean()
                df_clean['SMA_50'] = df_clean['Price'].rolling(50).mean()
                df_clean['RSI'] = compute_rsi(df_clean['Price'])
                df_clean = df_clean.dropna()
                
                st.session_state.data = df_clean
                st.success(f"Processed {len(df_clean)} data points")
                st.write(df_clean.tail())

        st.header("3. Model Training")
        if st.button("Train Prediction Model"):
            if 'SMA_20' not in df.columns:
                st.error("Run data processing first")
                return
                
            X = df[['SMA_20', 'SMA_50', 'RSI']]
            y = df['Price']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False
            )
            
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100)
            
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.success("Model trained successfully")

        st.header("4. Predictions")
        if st.session_state.model:
            predictions = st.session_state.model.predict(X_scaled)
            df['Prediction'] = predictions
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name='Actual'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Prediction'], name='Predicted'))
            st.plotly_chart(fig)
            
            st.metric("RMSE", np.sqrt(mean_squared_error(y, predictions)))
            st.metric("R² Score", r2_score(y, predictions))

if __name__ == "__main__":
    main()
