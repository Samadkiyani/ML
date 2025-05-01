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
import time
import pytz
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pandas.tseries.holiday import USFederalHolidayCalendar

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
    .error-box {border-left: 4px solid #ff4b4b; padding: 10px; background-color: #ffe6e6;}
    .warning-box {border-left: 4px solid #FFA500; padding: 10px; background-color: #fff4e6;}
</style>
""", unsafe_allow_html=True)

# Configure retry strategy
retry = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504)
)

# Create custom session with headers
session = Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
})
session.mount('https://', HTTPAdapter(max_retries=retry))

# Holiday calendar setup
cal = USFederalHolidayCalendar()

def is_market_open(date):
    """Check if date was a trading day"""
    try:
        test_data = yf.download('AAPL', 
                              start=date, 
                              end=date + datetime.timedelta(days=1),
                              progress=False)
        return not test_data.empty
    except Exception:
        return False

def get_market_status():
    """Check current market status"""
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    today = now.date()
    open_time = datetime.time(9, 30)
    close_time = datetime.time(16, 0)
    
    if now.weekday() >= 5:
        next_open = today + datetime.timedelta(days=(7 - now.weekday()))
        return "closed", f"Next open: {next_open.strftime('%Y-%m-%d')} 09:30 AM ET"
    
    market_open = ny_tz.localize(datetime.datetime.combine(today, open_time))
    market_close = ny_tz.localize(datetime.datetime.combine(today, close_time))
    
    if market_open <= now <= market_close and is_market_open(today):
        time_left = market_close - now
        return "open", f"Closes in {time_left.seconds//3600}h {(time_left.seconds//60)%60}m"
    else:
        next_open = today + datetime.timedelta(days=1)
        while next_open.weekday() >= 5 or not is_market_open(next_open):
            next_open += datetime.timedelta(days=1)
        return "closed", f"Next open: {next_open.strftime('%Y-%m-%d')} 09:30 AM ET"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetch data with market validation"""
    # Adjust for non-trading days
    adjusted_end = end_date
    for _ in range(5):
        if is_market_open(adjusted_end):
            break
        adjusted_end -= datetime.timedelta(days=1)
    
    if start_date > adjusted_end:
        st.markdown(f"""
        <div class='error-box'>
        <h3>❌ Invalid Date Range</h3>
        <p>No valid trading days between {start_date} and {end_date}</p>
        </div>
        """, unsafe_allow_html=True)
        return None

    # Check holidays
    holidays = cal.holidays(start=start_date, end=adjusted_end).date.tolist()
    if holidays:
        st.markdown(f"""
        <div class='warning-box'>
        <h3>⚠️ Market Holidays Detected</h3>
        <p>Holidays in range: {', '.join([str(h) for h in holidays])}</p>
        </div>
        """, unsafe_allow_html=True)

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=adjusted_end + datetime.timedelta(days=1),
            session=session,
            auto_adjust=True
        )
        
        if data.empty:
            st.markdown(f"""
            <div class='error-box'>
            <h3>🚨 No Data Found for {ticker}</h3>
            <p>Between {start_date} and {adjusted_end}</p>
            <p>Possible reasons:</p>
            <ul>
                <li>Market closed during selected period</li>
                <li>Delisted company</li>
                <li>Data provider issues</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            return None
            
        return data.reset_index()
    
    except Exception as e:
        st.markdown(f"""
        <div class='error-box'>
        <h3>🔥 Download Failed</h3>
        <p>Error: {str(e)}</p>
        <p>Try:</p>
        <ol>
            <li>Waiting a few minutes</li>
            <li>Checking your internet connection</li>
            <li>Using a different ticker symbol</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        return None

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
        'predictions': None,
        'download_count': 0
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
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date:",
                    datetime.date(2020, 1, 2),
                    min_value=datetime.date(1990, 1, 1),
                    max_value=datetime.date.today(),
                    help="Select a market open date (Mon-Fri, non-holiday)"
                )
            with col2:
                end_date = st.date_input(
                    "End Date:",
                    datetime.date.today() - datetime.timedelta(days=1),
                    min_value=datetime.date(1990, 1, 1),
                    max_value=datetime.date.today(),
                    help="Select a market open date (Mon-Fri, non-holiday)"
                )
            
            market_status, status_msg = get_market_status()
            st.caption(f"📈 Market Status: **{market_status.upper()}** ({status_msg})")
            
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

    # Step 1: Data Acquisition
    st.header("1. Data Acquisition")
    if st.button("🚀 Load Data"):
        st.session_state.download_count += 1
        if st.session_state.download_count > 5:
            st.warning("⚠️ API request limit approaching - consider waiting before next request")
            time.sleep(1)
            
        try:
            if data_source == "Yahoo Finance":
                with st.spinner("Fetching market data..."):
                    df = fetch_yfinance_data(ticker, start_date, end_date)
                    if df is None:
                        return
                        
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExenpzeTAwcjE1dTM0YXVueGF6azl4NWVwZTZvaWt1cmZpNm1jdGdnMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LPPMTiRjzhJKXS6okK/giphy.gif", 
                           caption="Market data loaded!")
            else:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file).reset_index(drop=True)
                    if df.empty:
                        st.error("Uploaded CSV file is empty!")
                        return
                    st.success("CSV file loaded successfully!")
                else:
                    st.warning("Please upload a CSV file!")
                    return

            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.write("### Data Preview:")
            st.dataframe(df.head().style.format("{:.2f}"), height=200)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    # Step 2: Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Data"):
                try:
                    df = st.session_state.data.copy()
                    
                    st.write("### Missing Values Analysis:")
                    missing = pd.DataFrame({
                        'Feature': df.columns,
                        'Missing Values': df.isnull().sum().values
                    })
                    
                    fig = px.bar(missing, 
                                x='Missing Values', 
                                y='Feature',
                                orientation='h',
                                labels={'Feature': 'Features', 'Missing Values': 'Count'},
                                color='Feature',
                                color_discrete_sequence=['#2a4a7c'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    df = df.dropna().reset_index(drop=True)
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success("Data cleaning completed! Missing values removed.")

                except Exception as e:
                    st.error(f"Data cleaning failed: {str(e)}")

        with col2:
            if st.session_state.steps['processed']:
                try:
                    st.write("### Cleaned Data Statistics:")
                    clean_df = st.session_state.data
                    st.dataframe(clean_df.describe().style.format("{:.2f}"), height=300)
                except Exception as e:
                    st.error(f"Error displaying statistics: {str(e)}")

    # Step 3: Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚡ Create Features"):
            try:
                df = st.session_state.data.copy()
                
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['RSI'] = compute_rsi(df['Close'])
                df = df.dropna().reset_index(drop=True)
                
                st.session_state.data = df
                st.session_state.steps['features_created'] = True
                
                st.write("### Feature Correlation Matrix:")
                corr_matrix = df.corr()
                fig = px.imshow(corr_matrix, 
                              text_auto=".2f", 
                              color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Feature engineering failed: {str(e)}")

    # Step 4: Train/Test Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("✂️ Split Dataset"):
            try:
                df = st.session_state.data.copy()
                X = df[['SMA_20', 'SMA_50', 'RSI']]
                y = df['Close'].values
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, shuffle=False)
                
                st.session_state.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': scaler
                })
                st.session_state.steps['split'] = True
                
                st.write("### Dataset Split Visualization:")
                split_df = pd.DataFrame({
                    'Set': ['Train', 'Test'],
                    'Count': [len(X_train), len(X_test)]
                })
                fig = px.pie(split_df, 
                            values='Count', 
                            names='Set', 
                            color_discrete_sequence=['#2a4a7c', '#3b6ea5'])
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Data splitting failed: {str(e)}")

    # Step 5: Model Training
    if st.session_state.steps.get('split'):
        st.header("5. Model Training")
        
        if st.button("🎯 Train Model"):
            try:
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100)
                
                with st.spinner("Training in progress..."):
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = model
                    st.session_state.steps['trained'] = True
                
                st.success(f"{model_type} trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"Model training failed: {str(e)}")

    # Step 6: Model Evaluation
    if st.session_state.steps.get('trained'):
        st.header("6. Model Evaluation")
        
        if st.button("📊 Evaluate Performance"):
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                y_pred = model.predict(X_test).flatten()
                
                if len(y_test.shape) > 1:
                    y_test = y_test.ravel()
                
                st.session_state.predictions = y_pred
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                with col2:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(len(y_test)), 
                                       y=y_test, 
                                       name='Actual', 
                                       line=dict(color='#2a4a7c')))
                fig.add_trace(go.Scatter(x=np.arange(len(y_test)), 
                                       y=y_pred, 
                                       name='Predicted', 
                                       line=dict(color='#4CAF50')))
                fig.update_layout(title="Actual vs Predicted Prices",
                                xaxis_title="Index",
                                yaxis_title="Price",
                                template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                if model_type == "Random Forest":
                    st.write("### Feature Importance:")
                    importance = model.feature_importances_
                    features = ['SMA_20', 'SMA_50', 'RSI']
                    fig = px.bar(x=features, 
                               y=importance, 
                               labels={'x': 'Features', 'y': 'Importance'},
                               color=features, 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)

                results = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Predictions", csv, 
                                  "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | [GitHub Repo](#)")

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    main()
