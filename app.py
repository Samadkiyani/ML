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
    .success-box {border-left: 4px solid #4CAF50; padding: 10px; background-color: #e8f5e9;}
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
                    st.markdown(f"""
                    <div class='success-box'>
                    <h3>✅ Successfully loaded {len(df)} trading days of data!</h3>
                    <p>Ticker: {ticker} | Date Range: {start_date} to {end_date}</p>
                    </div>
                    """, unsafe_allow_html=True)
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
                    
                    # Clean data
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
                
                # Technical indicators
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
                y = df['Close'].values  # Convert to numpy array
                
                # Feature scaling
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
                
                # Ensure 1D array for predictions
                y_pred = model.predict(X_test).flatten()
                
                # Handle possible 2D arrays
                if len(y_test.shape) > 1:
                    y_test = y_test.ravel()
                
                st.session_state.predictions = y_pred
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                with col2:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                
                # Prediction plot
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
    st.markdown("Built By Samad Kiani ❤ using Streamlit")

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    main()
