# app.py - Complete Secure Financial Platform
import os
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
    page_title="Secure Stock AI",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key from secure sources
API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", os.getenv("ALPHA_VANTAGE_KEY"))

if not API_KEY:
    st.error("""
    🔐 Missing API Key Configuration!
    Configure either:
    1. Add to .streamlit/secrets.toml (for deployment):
       ALPHA_VANTAGE_KEY = "BN2FYPLW1G4W7Y2G"
    2. Set environment variable (for local development):
       export ALPHA_VANTAGE_KEY='your-api-key-here'
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .secure-banner {color: #2ecc71; border: 1px solid #27ae60; padding: 10px; border-radius: 5px;}
    .api-warning {background-color: #fde8e8; padding: 15px; border-radius: 5px;}
    .data-table {max-height: 400px; overflow-y: auto;}
</style>
""", unsafe_allow_html=True)

# Alpha Vantage Configuration
BASE_URL = "https://www.alphavantage.co/query"
MAX_RETRIES = 3
REQUEST_INTERVAL = 15  # Seconds between retries

def validate_ticker(ticker):
    """Validate stock ticker format securely"""
    return re.match(r'^[A-Za-z]{1,5}(\.[A-Za-z]{1,2})?$', ticker) is not None

def fetch_secure_data(ticker, start_date, end_date):
    """Securely fetch stock data from Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": API_KEY,
        "outputsize": "compact",
        "datatype": "json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise ValueError("API Error: " + data["Error Message"])
            if "Note" in data:
                raise ValueError("API rate limit exceeded")

            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                raise ValueError("No data available for this ticker")

            df = pd.DataFrame.from_dict(time_series, orient="index")
            df = df.rename(columns={'4. close': 'Close'})[['Close']].astype(float)
            df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask]
            
            if len(filtered_df) < 30:
                raise ValueError(f"Only {len(filtered_df)} data points (minimum 30 required)")
                
            return filtered_df

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
    st.title("🔒 Secure Stock Analysis Platform")
    st.markdown("---")
    st.markdown("<div class='secure-banner'>API Key Securely Configured 🔑</div>", unsafe_allow_html=True)
    
    # Session state initialization
    session_defaults = {'data': None, 'model': None, 'features': False}
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

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
    st.header("1. Secure Data Acquisition")
    if st.button("🔑 Load Encrypted Data"):
        if data_source == "Alpha Vantage":
            if not validate_ticker(ticker):
                st.error("Invalid ticker format. Valid examples: AAPL, BRK.B")
                return

            if start_date >= end_date:
                st.error("Invalid date range selected")
                return

            try:
                with st.spinner(f"🔒 Securely fetching {ticker} data..."):
                    df = fetch_secure_data(ticker, start_date, end_date)
                    
                    st.session_state.data = df
                    st.session_state.features = False
                    st.success(f"Securely loaded {len(df)} trading days")
                    
                    # Display data summary
                    with st.expander("View Raw Data"):
                        st.dataframe(df.style.format({"Close": "{:.2f}"}), height=300)

                    st.line_chart(df['Close'])

            except Exception as e:
                st.markdown(f"""
                <div class='api-warning'>
                <h4>🔒 Secure Connection Error</h4>
                <p><strong>Reason:</strong> {str(e)}</p>
                <p><strong>Security Checks:</strong></p>
                <ol>
                    <li>Verify API key validity</li>
                    <li>Check network security</li>
                    <li>Confirm ticker permissions</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)
        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Close' not in df.columns or 'Date' not in df.columns:
                        raise ValueError("CSV must contain 'Date' and 'Close' columns")
                        
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()[['Close']]
                    
                    # Filter by date range
                    if data_source == "CSV Upload":
                        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
                        df = df.loc[mask]
                    
                    st.session_state.data = df
                    st.session_state.features = False
                    st.success("CSV data loaded securely")
                    
                    with st.expander("View CSV Data"):
                        st.dataframe(df.style.format({"Close": "{:.2f}"}), height=300)
                        
                except Exception as e:
                    st.error(f"CSV Error: {str(e)}")
            else:
                st.warning("Please upload a CSV file")

    # Data processing
    if st.session_state.data is not None:
        st.header("2. Data Processing")
        df = st.session_state.data
        
        if st.button("🧹 Clean & Transform Data"):
            with st.spinner("Securely processing data..."):
                try:
                    # Handle missing values
                    df_clean = df.dropna()
                    df_clean = df_clean[df_clean['Close'] > 0]
                    
                    # Feature engineering
                    df_clean['SMA_20'] = df_clean['Close'].rolling(20).mean()
                    df_clean['SMA_50'] = df_clean['Close'].rolling(50).mean()
                    df_clean['RSI'] = compute_rsi(df_clean['Close'])
                    df_clean = df_clean.dropna()
                    
                    st.session_state.data = df_clean
                    st.session_state.features = True
                    st.success(f"Processed {len(df_clean)} secure data points")
                    
                    # Show processed data
                    with st.expander("View Processed Data"):
                        st.dataframe(df_clean.style.format({
                            "Close": "{:.2f}",
                            "SMA_20": "{:.2f}",
                            "SMA_50": "{:.2f}",
                            "RSI": "{:.2f}"
                        }), height=300)

                    # Visualize features
                    fig = px.line(df_clean, y=['Close', 'SMA_20', 'SMA_50'], 
                                title="Price and Moving Averages")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

        # Model training section
        if st.session_state.features:
            st.header("3. Secure Model Training")
            
            if st.button("🤖 Train Predictive Model"):
                with st.spinner("Training secure model..."):
                    try:
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
                            model = RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1
                            )
                        
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        st.success("Model trained securely")
                        
                        # Immediate evaluation
                        y_pred = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col2:
                            st.metric("R² Score", f"{r2:.2f}")
                            
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")

            # Prediction and visualization
            if st.session_state.model:
                st.header("4. Secure Predictions")
                
                try:
                    df = st.session_state.data
                    X = df[['SMA_20', 'SMA_50', 'RSI']]
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    df['Predictions'] = st.session_state.model.predict(X_scaled)
                    
                    # Create visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        name='Actual Prices',
                        line=dict(color='#2c3e50')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Predictions'],
                        name='Model Predictions',
                        line=dict(color='#3498db')
                    ))
                    fig.update_layout(
                        title="Actual vs Predicted Prices",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show predictions table
                    with st.expander("View Detailed Predictions"):
                        st.dataframe(df[['Close', 'Predictions']].style.format("{:.2f}"), height=300)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
