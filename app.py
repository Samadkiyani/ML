# app.py - Financial ML Platform with Twelve Data Integration
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

# Constants
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
HISTORICAL_YEARS = 5
MAX_REQUESTS_PER_DAY = 800

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
    .rate-counter {color: #d35400; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_twelve_data(symbol, api_key):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_YEARS*365)
    
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "apikey": api_key,
        "outputsize": 5000
    }
    
    try:
        response = requests.get(TWELVE_DATA_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data:
            st.error(f"API Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data['values'])
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df.sort_values('Date').dropna().reset_index(drop=True)
        
    except Exception as e:
        st.error(f"API Request Failed: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("📈 FinML Pro - Twelve Data Integration")
    st.markdown("---")
    
    # Session state management
    session_defaults = {
        'data': None, 'model': None, 'request_count': 0,
        'steps': {'loaded': False, 'processed': False, 
                 'features_created': False, 'split': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Twelve Data Config")
        twelve_api_key = st.text_input("API Key:", type="password")
        symbol = st.text_input("Stock Symbol:", "AAPL").upper()
        
        st.markdown(f"""
        ---
        **API Limits:**
        - {MAX_REQUESTS_PER_DAY} requests/day
        - 8 requests/minute
        [Get API Key](https://twelvedata.com/pricing)
        """)
        
        st.markdown(f"<div class='rate-counter'>Requests Used Today: {st.session_state.request_count}/{MAX_REQUESTS_PER_DAY}</div>", 
                   unsafe_allow_html=True)
        
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Algorithm:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Step 1: Data Acquisition
    st.header("1. Data Acquisition")
    if st.button("🌐 Fetch Market Data"):
        if not twelve_api_key:
            st.error("API Key Required!")
            return
            
        if st.session_state.request_count >= MAX_REQUESTS_PER_DAY:
            st.error("Daily API limit reached!")
            return
            
        with st.spinner(f"Fetching {HISTORICAL_YEARS} years of data for {symbol}..."):
            df = fetch_twelve_data(symbol, twelve_api_key)
            
            if df.empty:
                return
                
            st.session_state.data = df
            st.session_state.request_count += 1
            st.session_state.steps['loaded'] = True
            
            st.success(f"Retrieved {len(df)} trading days")
            st.dataframe(df.tail(10).style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close']))

    # Step 2: Data Preparation
    if st.session_state.steps['loaded']:
        st.header("2. Data Preparation")
        
        if st.button("🧹 Clean Data"):
            df = st.session_state.data.copy()
            
            st.write("### Data Quality Report")
            quality_report = pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Zero Values': (df == 0).sum(),
                'Data Type': df.dtypes
            })
            
            fig = px.bar(quality_report, 
                        x=['Missing Values', 'Zero Values'], 
                        y=quality_report.index,
                        barmode='group',
                        labels={'value': 'Count', 'variable': 'Metric'},
                        color_discrete_sequence=['#2a4a7c', '#3b6ea5'])
            st.plotly_chart(fig, use_container_width=True)
            
            df = df.dropna()
            st.session_state.data = df
            st.session_state.steps['processed'] = True
            st.success(f"Clean dataset: {len(df)} records")

    # Step 3: Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚡ Generate Features"):
            df = st.session_state.data.copy()
            
            with st.spinner("Creating technical indicators..."):
                df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
                df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
                df['RSI'] = compute_rsi(df['Close'])
                df = df.dropna()
                
                st.session_state.data = df
                st.session_state.steps['features_created'] = True
                
                st.write("### Feature Relationships")
                fig = px.scatter_matrix(df[['Close', 'SMA_20', 'SMA_50', 'RSI']],
                                       color='Close', height=800)
                st.plotly_chart(fig, use_container_width=True)

    # Step 4: Data Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("⏰ Time-based Split"):
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
            
            st.write("### Split Visualization")
            split_df = pd.DataFrame({
                'Dataset': ['Train', 'Test'],
                'Samples': [len(X_train), len(X_test)],
                'Start Date': [df['Date'].min().strftime("%Y-%m-%d"), 
                             df.iloc[split_idx]['Date'].strftime("%Y-%m-%d")],
                'End Date': [df.iloc[split_idx-1]['Date'].strftime("%Y-%m-%d"), 
                           df['Date'].max().strftime("%Y-%m-%d")]
            })
            
            fig = px.bar(split_df, x='Dataset', y='Samples', 
                        text='Samples', color='Dataset',
                        hover_data=['Start Date', 'End Date'])
            st.plotly_chart(fig, use_container_width=True)

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
        
        if st.button("📈 Evaluate Performance"):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = model.predict(X_test)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            with col3:
                error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                st.metric("MAPE", f"{error:.2f}%")
            
            # Prediction Visualization
            results = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Date': st.session_state.data['Date'].iloc[-len(y_test):]
            })
            
            fig = px.line(results, x='Date', y=['Actual', 'Predicted'],
                          labels={'value': 'Price', 'variable': 'Type'},
                          color_discrete_sequence=['#2a4a7c', '#4CAF50'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if model_type == "Random Forest":
                st.write("### Feature Importance")
                importance = pd.DataFrame({
                    'Feature': ['SMA_20', 'SMA_50', 'RSI'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature',
                            orientation='h', color='Importance',
                            color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
