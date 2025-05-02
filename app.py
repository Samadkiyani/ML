# app.py - Financial ML Platform with IEX Cloud Integration
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="FinML Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2a4a7c; border-bottom: 2px solid #2a4a7c;}
    h2 {color: #3b6ea5;}
    .stButton>button {background-color: #2a4a7c; color: white; border-radius: 5px;}
    .stDownloadButton>button {background-color: #4CAF50; color: white;}
    .stAlert {border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    .rate-limit {color: #e67e22; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

IEX_BASE_URL = "https://cloud.iexapis.com/stable"
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 60  # seconds

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_iex_data(symbol, token, years=5):
    endpoint = f"{IEX_BASE_URL}/stock/{symbol}/chart/5y"
    params = {
        "token": token,
        "chartCloseOnly": False,
        "includeToday": True
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                st.sidebar.markdown(f"<div class='rate-limit'>API Calls Remaining: {remaining}</div>", 
                                  unsafe_allow_html=True)
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Convert IEX format to standard OHLCV
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            
            return df.sort_values('Date').reset_index(drop=True)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                st.error(f"Rate limit exceeded. Waiting {RATE_LIMIT_WAIT} seconds...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            raise
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return pd.DataFrame()
            
    return pd.DataFrame()

def main():
    st.title("📈 FinML Pro - IEX Cloud Integration")
    st.markdown("---")
    
    # Session state
    session_defaults = {
        'data': None, 'model': None, 'api_remaining': 'N/A',
        'steps': {'loaded': False, 'processed': False, 
                 'features_created': False, 'split': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ IEX Cloud Configuration")
        iex_token = st.text_input("IEX API Token:", type="password")
        symbol = st.text_input("Stock Symbol:", "AAPL").upper()
        
        st.markdown("""
        ---
        **Free Tier Limits:**
        - 500,000 messages/month
        - 50 messages/second
        [Get API Key](https://iexcloud.io/cloud-login#/register/)
        """)
        
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reload App", on_click=lambda: st.session_state.clear())

    # Step 1: Data Acquisition
    st.header("1. Data Acquisition")
    if st.button("📡 Fetch IEX Data"):
        if not iex_token:
            st.error("IEX API Token required!")
            return
            
        if not symbol.isalpha():
            st.error("Invalid stock symbol!")
            return
            
        with st.spinner(f"Fetching 5-year historical data for {symbol}..."):
            df = fetch_iex_data(symbol, iex_token)
            
            if df.empty:
                st.error("Failed to fetch data. Check symbol and API token.")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✅ Fetched {len(df)} trading days")
            st.dataframe(df.tail(10).style.format("{:.2f}"), height=350)

    # Step 2: Data Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        
        if st.button("🧼 Clean & Validate"):
            df = st.session_state.data.copy()
            
            # Convert numeric types
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            st.session_state.data = df
            st.session_state.steps['processed'] = True
            
            st.write("### Data Summary")
            st.dataframe(df.describe().style.format("{:.2f}"), height=250)
            
            fig = px.line(df, x='Date', y='Close', title="Historical Closing Prices")
            st.plotly_chart(fig, use_container_width=True)

    # Step 3: Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚙ Generate Features"):
            df = st.session_state.data.copy()
            
            with st.spinner("Creating technical indicators..."):
                df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
                df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
                df['RSI'] = compute_rsi(df['Close'])
                df = df.dropna()
                
                st.session_state.data = df
                st.session_state.steps['features_created'] = True
                
                st.write("### Feature Correlation")
                fig = px.imshow(df.corr(), color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

    # Step 4: Data Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("✂ Split Dataset"):
            df = st.session_state.data.copy()
            features = ['SMA_20', 'SMA_50', 'RSI']
            
            X = df[features]
            y = df['Close'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            st.session_state.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            })
            st.session_state.steps['split'] = True
            
            st.write(f"Training Period: {df['Date'].iloc[0].date()} - {df['Date'].iloc[split_idx-1].date()}")
            st.write(f"Testing Period: {df['Date'].iloc[split_idx].date()} - {df['Date'].iloc[-1].date()}")

    # Step 5: Model Training
    if st.session_state.steps['split']:
        st.header("5. Model Training")
        
        if st.button("🎯 Train Model"):
            model = LinearRegression() if model_type == "Linear Regression" \
                else RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            st.success(f"{model_type} trained successfully!")
            st.balloons()

    # Step 6: Model Evaluation
    if st.session_state.steps['trained']:
        st.header("6. Model Evaluation")
        
        if st.button("📊 Evaluate"):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                                   mode='lines', name='Perfect Fit'))
            st.plotly_chart(fig, use_container_width=True)
            
            if model_type == "Random Forest":
                st.write("Feature Importance:")
                importance = pd.DataFrame({
                    'Feature': ['SMA_20', 'SMA_50', 'RSI'],
                    'Importance': model.feature_importances_
                })
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
