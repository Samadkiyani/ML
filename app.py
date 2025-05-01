# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
import pytz

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
    .stAlert {border-radius: 5px; padding: 20px;}
    .error-box {border-left: 4px solid #ff4b4b; padding: 10px; background-color: #ffe6e6;}
</style>
""", unsafe_allow_html=True)

def validate_ticker(ticker):
    """Enhanced ticker validation with multiple checks"""
    try:
        # Check historical data for multiple time periods
        for period in ["1d", "5d", "1mo"]:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                return True
        return False
    except Exception:
        return False

def main():
    st.title("📈 FinML Pro - Financial Machine Learning Platform")
    st.markdown("---")
    
    # Initialize session state
    session_defaults = {
        'data': None,
        'model': None,
        'steps': {
            'loaded': False,
            'processed': False,
            'features_created': False,
            'split': False,
            'trained': False
        },
        'predictions': None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Ticker (e.g., AAPL):", "AAPL").strip().upper()
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
        else:
            uploaded_file = st.file_uploader("Upload Dataset:", type=["csv"])
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", 
                                ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        
        st.markdown("---")
        st.header("🔗 Navigation")
        st.button("Reload App", on_click=lambda: st.session_state.clear())

    # Step 1: Load Data with Advanced Validation
    st.header("1. Data Acquisition")
    if st.button("🚀 Load Data"):
        try:
            if data_source == "Yahoo Finance":
                with st.spinner("Fetching market data..."):
                    # Validate ticker first
                    if not validate_ticker(ticker):
                        st.markdown(f"""
                        <div class='error-box'>
                        <h3>❌ Ticker Validation Failed: {ticker}</h3>
                        <p>Possible solutions:</p>
                        <ol>
                            <li>Check ticker symbol on <a href="https://finance.yahoo.com" target="_blank">Yahoo Finance</a></li>
                            <li>Try different date range</li>
                            <li>Check regional restrictions (try VPN)</li>
                            <li>Update yfinance: <code>pip install yfinance --upgrade</code></li>
                        </ol>
                        </div>
                        """, unsafe_allow_html=True)
                        return

                    # Handle timezone-aware dates
                    ny_tz = pytz.timezone("America/New_York")
                    today = datetime.datetime.now(ny_tz).date()
                    
                    # Auto-adjust future dates
                    if end_date > today:
                        st.warning(f"⚠️ Adjusted end date to today ({today})")
                        end_date = today

                    # Convert to timezone-aware datetimes
                    start_dt = ny_tz.localize(datetime.datetime.combine(start_date, datetime.time(9, 30)))
                    end_dt = ny_tz.localize(datetime.datetime.combine(end_date, datetime.time(16, 0))) + datetime.timedelta(days=1)
                    
                    # Download data with retry logic
                    try:
                        df = yf.download(ticker, start=start_dt, end=end_dt, auto_adjust=True)
                    except Exception as e:
                        st.error(f"First download attempt failed: {str(e)}. Retrying...")
                        df = yf.download(ticker, start=start_dt, end=end_dt, auto_adjust=True)

                    if df.empty:
                        st.markdown(f"""
                        <div class='error-box'>
                        <h3>🚨 No Data Found for {ticker}</h3>
                        <p>Between {start_date} and {end_date}</p>
                        <p>Possible reasons:</p>
                        <ul>
                            <li>Market closed during selected period</li>
                            <li>Delisted company</li>
                            <li>Non-trading days selected</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        return

                    df = df.reset_index()
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    st.success(f"✅ Successfully loaded {len(df)} trading days of data!")
                    st.dataframe(df.head().style.format("{:.2f}"), height=200)

            else:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    if df.empty:
                        st.error("Uploaded CSV file is empty!")
                        return
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    st.success("CSV file loaded successfully!")
                    st.dataframe(df.head().style.format("{:.2f}"), height=200)
                else:
                    st.warning("Please upload a CSV file!")
                    return

        except Exception as e:
            st.markdown(f"""
            <div class='error-box'>
            <h3>🔥 Critical Error</h3>
            <p>{str(e)}</p>
            <p>Troubleshooting steps:</p>
            <ol>
                <li>Check internet connection</li>
                <li>Try different ticker/date range</li>
                <li>Restart the application</li>
                <li>Contact support</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

    # [Rest of the code remains unchanged from previous version...]

    # Steps 2-6 (Preprocessing, Feature Engineering, etc.) remain identical
    # to the previous working version provided earlier

    st.markdown("---")
    st.markdown("Built By Samad Kiani ❤ using Streamlit")

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    main()
