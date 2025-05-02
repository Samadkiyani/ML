# app.py - Financial ML Platform with Custom Stock Input
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
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_RETRIES = 2
BASE_DELAY = 5.0
JITTER = 3.0
MIN_DATA_POINTS = 30
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def validate_ticker(ticker):
    """Validate stock ticker format"""
    pattern = r'^[A-Za-z.-]{1,10}$'
    return re.match(pattern, ticker) is not None

def compute_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_download(ticker, start_date, end_date):
    """Robust data download with error handling"""
    for attempt in range(MAX_RETRIES):
        try:
            delay = BASE_DELAY + random.uniform(0, JITTER)
            with st.spinner(f"⏳ Safety delay {delay:.1f}s..."):
                time.sleep(delay)
            
            df = yf.download(
                ticker,
                start=start_date - datetime.timedelta(days=3),
                end=end_date + datetime.timedelta(days=3),
                progress=False,
                headers={'User-Agent': random.choice(USER_AGENTS)}
            )
            
            if df.empty:
                raise ValueError("Empty response from server")
                
            return df.loc[start_date:end_date].reset_index()
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

def main():
    st.title("📈 Custom Stock Analysis Platform")
    st.markdown("---")
    
    # Session state management
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "CSV Upload"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Enter Stock Ticker:", "", 
                                 help="Example: TSLA, GOOGL, BRK.B").strip().upper()
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])
        
        st.markdown("---")
        st.header("Model Settings")
        model_type = st.selectbox("Algorithm:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Data loading section
    st.header("1. Data Acquisition")
    if st.button("📥 Load Market Data"):
        if data_source == "Yahoo Finance":
            if not ticker:
                st.error("🛑 Please enter a stock ticker!")
                return

            if not validate_ticker(ticker):
                st.markdown(f"""
                <div class='ticker-error'>
                ❌ Invalid ticker format: {ticker}<br>
                Valid format: 1-10 letters/dots (e.g.: BRK.B)
                </div>
                """, unsafe_allow_html=True)
                return

            if start_date >= end_date:
                st.error("📅 End date must be after start date!")
                return

            try:
                with st.spinner("🔍 Validating ticker..."):
                    stock_info = yf.Ticker(ticker).info
                    if not stock_info.get('regularMarketPrice'):
                        raise ValueError("Invalid or delisted ticker")

                with st.spinner("📡 Downloading data..."):
                    df = safe_download(ticker, start_date, end_date)
                    
                    if len(df) < MIN_DATA_POINTS:
                        raise ValueError(f"Only {len(df)} data points (minimum {MIN_DATA_POINTS} required)")
                    
                    st.session_state.data = df
                    st.success(f"✅ Successfully loaded {ticker} data!")
                    st.dataframe(df.head().style.format(precision=2), height=200)

            except Exception as e:
                st.error(f"""
                ❗ Failed to load {ticker} data!
                📌 Reason: {str(e)}
                🔍 Troubleshooting:
                1. Verify ticker on [Yahoo Finance](https://finance.yahoo.com)
                2. Check date range (min 3 months)
                3. Try different ticker
                """)
        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if not {'Date', 'Close'}.issubset(df.columns):
                        raise ValueError("CSV must contain 'Date' and 'Close' columns")
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.data = df.sort_values('Date')
                    st.success("✅ CSV data loaded successfully!")
                except Exception as e:
                    st.error(f"CSV Error: {str(e)}")
            else:
                st.warning("⚠️ Please upload a CSV file")

    # Data preprocessing
    if st.session_state.data is not None:
        st.header("2. Data Preparation")
        df = st.session_state.data
        
        if st.button("🧼 Clean Data"):
            with st.spinner("Processing..."):
                initial_count = len(df)
                df_clean = df.dropna()
                final_count = len(df_clean)
                
                if final_count == 0:
                    st.error("🚨 All data removed during cleaning!")
                    return
                
                st.session_state.data = df_clean
                st.success(f"Removed {initial_count - final_count} rows with missing values")

        st.subheader("Data Statistics")
        st.dataframe(df.describe().style.format(precision=2), height=300)

    # Feature engineering
    if st.session_state.data is not None and len(st.session_state.data) > 50:
        st.header("3. Feature Engineering")
        
        if st.button("⚙️ Generate Features"):
            with st.spinner("Creating technical indicators..."):
                df = st.session_state.data
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['RSI'] = compute_rsi(df['Close'])
                df = df.dropna()
                st.session_state.data = df
                
                st.plotly_chart(px.line(df, x='Date', y=['Close', 'SMA_20', 'SMA_50'], 
                                      title="Price and Moving Averages"))

    # Model training
    if st.session_state.data is not None and len(st.session_state.data) > 100:
        st.header("4. Model Training")
        
        if st.button("🤖 Train Model"):
            df = st.session_state.data
            X = df[['SMA_20', 'SMA_50', 'RSI']]
            y = df['Close']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False
            )
            
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100)
            
            with st.spinner("Training in progress..."):
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.success("🎉 Model training completed!")

    # Model evaluation
    if st.session_state.model is not None:
        st.header("5. Model Evaluation")
        
        if st.button("📈 Evaluate Performance"):
            df = st.session_state.data
            X = df[['SMA_20', 'SMA_50', 'RSI']]
            y = df['Close']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_test = X_scaled[split_idx:]
            y_test = y[split_idx:]
            
            y_pred = st.session_state.model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual'))
            fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Predicted'))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
