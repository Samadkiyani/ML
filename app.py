# app.py - Complete Financial ML Platform
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
    .weekend-adjust {color: #d35400; font-weight: bold;}
    .error-list {padding-left: 20px; margin-top: 10px;}
    .countdown {color: #e67e22; font-weight: bold;}
    .csv-guide {border-left: 3px solid #2a4a7c; padding-left: 15px;}
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_RETRIES = 2
BASE_DELAY = 8.0
JITTER = 4.0
BACKUP_TICKERS = ['AAPL', 'MSFT']
MIN_DATA_POINTS = 10
RATE_LIMIT_COOLDOWN = 600 
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
]

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
    """Advanced download function with rate limit protection"""
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
                raise ValueError("Empty dataframe")
                
            df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
            return df.reset_index()
            
        except Exception as e:
            if "YFRateLimitError" in str(e):
                cooldown = RATE_LIMIT_COOLDOWN * (attempt + 1)
                st.error(f"""
                🔥 Critical Rate Limit Hit!
                ⏲️ Automatic cooldown: {cooldown//60} minutes
                """)
                with st.spinner(f"Waiting {cooldown//60} minutes..."):
                    time.sleep(cooldown)
                continue
            raise
    raise ValueError(f"Failed after {MAX_RETRIES} attempts")

def main():
    st.title("📈 FinML Pro - Financial Machine Learning Platform")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'current_ticker': None,
        'steps': {'loaded': False, 'processed': False, 
                 'features_created': False, 'split': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Ticker:", "AAPL").strip().upper()
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload Dataset:", type=["csv"])
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reload App", on_click=lambda: st.session_state.clear())

    # Step 1: Data Acquisition
    st.header("1. Data Acquisition")
    if st.button("🚀 Load Data"):
        try:
            if data_source == "Yahoo Finance":
                if start_date > end_date:
                    st.error("⛔ Start date must be before end date!")
                    return

                if not re.match(r"^[A-Za-z.-]{1,10}$", ticker):
                    st.error("❌ Invalid ticker format!")
                    return

                adjusted_end_date = end_date
                if end_date.weekday() >= 5:
                    adjusted_end_date -= datetime.timedelta(days=end_date.weekday()-4)
                    st.markdown(f"""<p class='weekend-adjust'>
                    ⚠️ Adjusted end date to {adjusted_end_date} (weekend)
                    </p>""", unsafe_allow_html=True)

                current_ticker = ticker
                all_tickers = [current_ticker] + BACKUP_TICKERS
                df = pd.DataFrame()
                failures = []

                for t in all_tickers:
                    try:
                        with st.spinner(f"🌐 Attempting {t}..."):
                            info = yf.Ticker(t).info
                            listing_date = pd.to_datetime(
                                info.get('firstTradeDateEpochUtc', pd.NaT), unit='s'
                            )
                            if pd.notna(listing_date) and start_date < listing_date.date():
                                raise ValueError(f"Start date precedes {listing_date.date()}")
                            
                            df = safe_download(t, start_date, adjusted_end_date)
                            
                            if len(df) < MIN_DATA_POINTS:
                                raise ValueError(f"Only {len(df)} data points")
                                
                            current_ticker = t
                            break
                            
                    except Exception as e:
                        failures.append(f"{t}: {str(e)}")
                        if "YFRateLimitError" in str(e):
                            st.error("""
                            🚨 Immediate Solutions:
                            1. Switch to CSV upload
                            2. Wait 15-20 minutes
                            3. Try 1-month date range
                            """)
                            return

                if df.empty:
                    st.markdown(f"""
                    ❌ All tickers failed!
                    <div class='error-list'>
                    {'<br>'.join(failures[-3:])}
                    </div>
                    🔧 Solutions:
                    1. Use CSV upload below
                    2. Try after {RATE_LIMIT_COOLDOWN//60} minutes
                    3. Reduce date range
                    """, unsafe_allow_html=True)
                    return

                if current_ticker != ticker:
                    st.warning(f"⚠️ Using backup ticker: {current_ticker}")
                    st.session_state.current_ticker = current_ticker

                st.session_state.data = df.sort_values('Date')
                st.session_state.steps['loaded'] = True
                st.success("✅ Data loaded successfully!")
                st.dataframe(df.head().style.format("{:.2f}"), height=250)

            else:
                st.markdown("""
                <div class='csv-guide'>
                📁 CSV Requirements:
                - Columns: <code>Date</code>, <code>Close</code>
                - Date Format: YYYY-MM-DD
                - Sample: <a href='https://bit.ly/finml-sample' target='_blank'>Download</a>
                </div>
                """, unsafe_allow_html=True)
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if not {'Date', 'Close'}.issubset(df.columns):
                            raise ValueError("Missing required columns")
                        df['Date'] = pd.to_datetime(df['Date'])
                        st.session_state.data = df.sort_values('Date')
                        st.session_state.steps['loaded'] = True
                        st.success("✅ CSV loaded successfully!")
                    except Exception as e:
                        st.error(f"CSV Error: {str(e)}")
                else:
                    st.warning("⚠️ Please upload a CSV file")

        except Exception as e:
            st.error(f"🚨 Critical Error: {str(e)}")
            st.markdown("""
            🔧 Troubleshooting Guide:
            1. Verify dates on [Yahoo Finance](https://finance.yahoo.com)
            2. Try smaller date range (1-3 months)
            3. Check network connection
            4. Use CSV upload
            """)

    # Step 2: Data Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Data"):
                try:
                    if st.session_state.data is None or st.session_state.data.empty:
                        st.error("⚠️ No data loaded! Complete Step 1 first.")
                        return
                        
                    df = st.session_state.data.copy()
                    
                    st.write("### Missing Values Analysis:")
                    missing = pd.DataFrame({
                        'Feature': df.columns,
                        'Missing Values': df.isnull().sum().values
                    })
                    
                    fig = px.bar(missing, x='Missing Values', y='Feature',
                                orientation='h', color='Feature',
                                color_discrete_sequence=['#2a4a7c'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.spinner("Cleaning data..."):
                        initial_count = len(df)
                        df = df.dropna().reset_index(drop=True)
                        final_count = len(df)
                    
                    if final_count == 0:
                        st.error("🔥 Critical Error: All data removed during cleaning!")
                        return
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success(f"✅ Cleaned data: {final_count} rows remaining")

                except Exception as e:
                    st.error(f"🚨 Cleaning failed: {str(e)}")

        with col2:
            if st.session_state.steps['processed']:
                try:
                    st.write("### Cleaned Data Statistics:")
                    clean_df = st.session_state.data
                    stats = clean_df.describe()
                    stats.loc['skew'] = clean_df.skew(numeric_only=True)
                    stats.loc['kurtosis'] = clean_df.kurtosis(numeric_only=True)
                    
                    st.dataframe(
                        stats.style.format("{:.2f}")
                        .highlight_null(props="background-color: #ffcccc"),
                        height=350
                    )
                    st.write(f"📅 Date Range: {clean_df['Date'].min().date()} to {clean_df['Date'].max().date()}")
                    
                except Exception as e:
                    st.error(f"📊 Stats display error: {str(e)}")

    # Step 3: Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚡ Create Features"):
            try:
                df = st.session_state.data.copy()
                
                if len(df) < 50:
                    st.error(f"❌ Insufficient data: {len(df)}/50 points required")
                    return
                    
                with st.spinner("Calculating technical indicators..."):
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['RSI'] = compute_rsi(df['Close'])
                    
                    if df[['SMA_20', 'SMA_50', 'RSI']].isnull().sum().sum() > 0:
                        st.warning("⚠️ NaN values detected after feature creation")
                        df = df.dropna().reset_index(drop=True)
                    
                    if len(df) < 30:
                        st.error("🔥 Critical Error: Too many NaN values after feature creation!")
                        return
                        
                    st.session_state.data = df
                    st.session_state.steps['features_created'] = True
                    
                    st.write("### Feature Correlation Matrix:")
                    corr_matrix = df.corr()
                    fig = px.imshow(corr_matrix, 
                                  text_auto=".2f", 
                                  color_continuous_scale='Blues',
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"🚨 Feature engineering failed: {str(e)}")

    # Step 4: Data Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("✂️ Split Dataset"):
            try:
                df = st.session_state.data.copy()
                required_features = ['SMA_20', 'SMA_50', 'RSI']
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    return
                    
                X = df[required_features]
                y = df['Close'].values  
                
                with st.spinner("Scaling features..."):
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                
                split_index = int(len(X_scaled) * (1 - test_size))
                X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
                
                st.session_state.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': scaler
                })
                st.session_state.steps['split'] = True
                
                st.write("### Dataset Split:")
                split_df = pd.DataFrame({
                    'Set': ['Train', 'Test'],
                    'Count': [len(X_train), len(X_test)],
                    'Percentage': [f"{len(X_train)/len(X_scaled):.0%}", 
                                 f"{len(X_test)/len(X_scaled):.0%}"]
                })
                fig = px.pie(split_df, values='Count', names='Set', 
                            color_discrete_sequence=['#2a4a7c', '#3b6ea5'],
                            hover_data=['Percentage'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"📅 Train Period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[split_index-1].date()}")
                st.write(f"📅 Test Period: {df['Date'].iloc[split_index].date()} to {df['Date'].iloc[-1].date()}")

            except Exception as e:
                st.error(f"🚨 Splitting failed: {str(e)}")

    # Step 5: Model Training
    if st.session_state.steps.get('split'):
        st.header("5. Model Training")
        
        if st.button("🎯 Train Model"):
            try:
                if not st.session_state.get('X_train'):
                    st.error("⚠️ Complete Step 4 first!")
                    return
                    
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        random_state=42,
                        n_jobs=-1
                    )
                
                with st.spinner(f"Training {model_type}..."):
                    progress_bar = st.progress(0)
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    progress_bar.progress(100)
                    
                    st.session_state.model = model
                    st.session_state.steps['trained'] = True
                
                st.success(f"✅ {model_type} trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"🚨 Training failed: {str(e)}")

    # Step 6: Model Evaluation
    if st.session_state.steps.get('trained'):
        st.header("6. Model Evaluation")
        
        if st.button("📊 Evaluate Performance"):
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                if model is None or X_test is None:
                    st.error("⚠️ Complete previous steps first!")
                    return
                    
                with st.spinner("Generating predictions..."):
                    y_pred = model.predict(X_test).flatten()
                    if len(y_test.shape) > 1:
                        y_test = y_test.ravel()
                    st.session_state.predictions = y_pred
                
                col1, col2 = st.columns(2)
                with col1:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    st.metric("RMSE", f"{rmse:.2f}")
                with col2:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("R² Score", f"{r2:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_test)), 
                    y=y_test, 
                    name='Actual', 
                    line=dict(color='#2a4a7c')
                ))
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_test)), 
                    y=y_pred,
                    name='Predicted', 
                    line=dict(color='#4CAF50')
                ))
                fig.update_layout(
                    title="Actual vs Predicted Prices",
                    xaxis_title="Time Index",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if model_type == "Random Forest":
                    st.write("### Feature Importance:")
                    importance = model.feature_importances_
                    features = ['SMA_20', 'SMA_50', 'RSI']
                    fig = px.bar(
                        x=features, 
                        y=importance, 
                        labels={'x': 'Features', 'y': 'Importance'},
                        color=features, 
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)

                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Predictions", 
                    csv, 
                    "predictions.csv", 
                    "text/csv"
                )

            except Exception as e:
                st.error(f"🚨 Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main()
