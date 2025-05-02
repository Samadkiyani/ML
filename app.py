# app.py - Financial Platform with Alpha Vantage Integration
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
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
    page_title="AlphaStock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; border-bottom: 2px solid #3498db;}
    .stAlert {border-left: 4px solid #3498db;}
    .api-warning {background-color: #fff3cd; padding: 15px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# Alpha Vantage Configuration
API_KEY = "BN2FYPLW1G4W7Y2G"
BASE_URL = "https://www.alphavantage.co/query"
MAX_RETRIES = 3
REQUEST_INTERVAL = 15  # Seconds between retries

def validate_ticker(ticker):
    """Validate stock ticker format"""
    return re.match(r'^[A-Za-z]{1,5}(\.[A-Za-z]{1,2})?$', ticker) is not None

def fetch_alpha_vantage_data(ticker, start_date, end_date):
    """Fetch daily time series data from Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": API_KEY,
        "outputsize": "full",
        "datatype": "json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(data["Error Message"])
            if "Note" in data:
                raise ValueError("API rate limit exceeded")

            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                raise ValueError("No time series data found")

            df = pd.DataFrame.from_dict(time_series, orient="index")
            df = df.rename(columns={
                '4. close': 'Close',
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '5. volume': 'Volume'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[['Close']].astype(float)
            
            # Filter by date range
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask]

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_INTERVAL * (attempt + 1))
                continue
            raise

    return pd.DataFrame()

def compute_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def main():
    st.title("📊 AlphaStock AI Analysis Platform")
    st.markdown("---")
    
    # Session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source", ["Alpha Vantage", "CSV Upload"])
        
        if data_source == "Alpha Vantage":
            ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
            start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        st.markdown("---")
        st.header("Model Settings")
        model_type = st.selectbox("Algorithm", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Data loading section
    st.header("1. Data Acquisition")
    if st.button("📥 Load Stock Data"):
        if data_source == "Alpha Vantage":
            if not validate_ticker(ticker):
                st.error("Invalid ticker format. Examples: AAPL, BRK.B")
                return

            if start_date >= end_date:
                st.error("End date must be after start date")
                return

            try:
                with st.spinner(f"Fetching {ticker} data from Alpha Vantage..."):
                    df = fetch_alpha_vantage_data(ticker, start_date, end_date)
                    
                    if df.empty:
                        raise ValueError("No data returned")
                        
                    st.session_state.data = df
                    st.success(f"Successfully loaded {len(df)} trading days")
                    st.line_chart(df['Close'])

            except Exception as e:
                st.error(f"""
                ❗ Data fetch failed: {str(e)}
                🔍 Troubleshooting:
                1. Verify API key status
                2. Check ticker symbol
                3. Wait 1 minute and try again
                4. Consider premium API plan for higher limits
                """)
        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Close' not in df.columns:
                        raise ValueError("CSV must contain 'Close' column")
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    st.session_state.data = df[['Close']]
                    st.success("CSV data loaded successfully")
                except Exception as e:
                    st.error(f"CSV Error: {str(e)}")
            else:
                st.warning("Please upload a CSV file")

    # Data processing
    if st.session_state.data is not None:
        st.header("2. Data Processing")
        df = st.session_state.data
        
        if st.button("🧹 Clean & Transform Data"):
            with st.spinner("Processing..."):
                # Handle missing values
                df_clean = df.dropna()
                df_clean = df_clean[df_clean['Close'] > 0]
                
                # Feature engineering
                df_clean['SMA_20'] = df_clean['Close'].rolling(20).mean()
                df_clean['SMA_50'] = df_clean['Close'].rolling(50).mean()
                df_clean['RSI'] = compute_rsi(df_clean['Close'])
                df_clean = df_clean.dropna()
                
                st.session_state.data = df_clean
                st.success(f"Processed {len(df_clean)} data points")
                st.write(df_clean.tail())

        st.header("3. Model Training")
        if st.button("🤖 Train Prediction Model"):
            df = st.session_state.data
            if 'SMA_20' not in df.columns:
                st.error("Run data processing first")
                return
                
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
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.success("Model training completed")

        st.header("4. Predictions & Analysis")
        if st.session_state.model:
            df = st.session_state.data
            X = df[['SMA_20', 'SMA_50', 'RSI']]
            X_scaled = StandardScaler().fit_transform(X)
            
            predictions = st.session_state.model.predict(X_scaled)
            df['Predicted'] = predictions
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted'))
            fig.update_layout(title="Actual vs Predicted Prices")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y, predictions)):.2f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y, predictions):.2f}")

if __name__ == "__main__":
    main()
