# app.py - Full Implementation
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
</style>
""", unsafe_allow_html=True)

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

    # Step 1: Data Acquisition
    st.header("1. Data Acquisition")
    if st.button("🚀 Load Data"):
        try:
            if data_source == "Yahoo Finance":
                if start_date > end_date:
                    st.error("⛔ Error: Start date cannot be after end date!")
                    return
                
                with st.spinner("Verifying ticker..."):
                    ticker_check = yf.Ticker(ticker)
                    if ticker_check.history(period="1d").empty:
                        st.error(f"❌ Invalid or delisted ticker: {ticker}")
                        return
                
                with st.spinner(f"Fetching {ticker} data..."):
                    df = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date + datetime.timedelta(days=1),
                        progress=False
                    )
                    
                    if df.empty:
                        st.error(f"⚠️ No data found for {ticker}!")
                        return
                        
                    df = df.reset_index()
                    st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExenpzeTAwcjE1dTM0YXVueGF6azl4NWVwZTZvaWt1cmZpNm1jdGdnMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LPPMTiRjzhJKXS6okK/giphy.gif", 
                           caption="Market data loaded!")

            else:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    if {'Date', 'Close'}.issubset(df.columns):
                        df['Date'] = pd.to_datetime(df['Date'])
                        st.success("✅ CSV loaded successfully!")
                    else:
                        st.error("Missing required columns (Date/Close)")
                        return
                else:
                    st.warning("⚠️ Please upload a CSV file!")
                    return

            st.session_state.data = df.sort_values('Date')
            st.session_state.steps['loaded'] = True
            
            st.write(f"### Data Preview ({len(df)} rows)")
            st.dataframe(df.head().style.format("{:.2f}"), height=250)
            st.write(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        except Exception as e:
            st.error(f"🚨 Data loading failed: {str(e)}")

    # Step 2: Data Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Data"):
                try:
                    if st.session_state.data is None:
                        st.error("No data loaded! Complete Step 1 first.")
                        return
                        
                    df = st.session_state.data.copy()
                    
                    if 'Date' not in df.columns or 'Close' not in df.columns:
                        st.error("Invalid dataset structure!")
                        return

                    st.write("### Missing Values Analysis:")
                    missing = pd.DataFrame({
                        'Feature': df.columns,
                        'Missing Values': df.isnull().sum().values
                    })
                    
                    fig = px.bar(missing, x='Missing Values', y='Feature',
                                orientation='h', color='Feature',
                                color_discrete_sequence=['#2a4a7c'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    initial_count = len(df)
                    df = df.dropna().reset_index(drop=True)
                    final_count = len(df)
                    
                    if final_count == 0:
                        st.error("Data cleaning removed all records!")
                        return
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success(f"Cleaned data: {final_count} rows remaining")

                except Exception as e:
                    st.error(f"Data cleaning failed: {str(e)}")

        with col2:
            if st.session_state.steps['processed']:
                try:
                    st.write("### Cleaned Data Statistics:")
                    clean_df = st.session_state.data
                    stats = clean_df.describe()
                    stats.loc['skew'] = clean_df.skew(numeric_only=True)
                    stats.loc['kurtosis'] = clean_df.kurtosis(numeric_only=True)
                    st.dataframe(stats.style.format("{:.2f}"), height=350)
                except Exception as e:
                    st.error(f"Error displaying stats: {str(e)}")

    # Step 3: Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚡ Create Features"):
            try:
                df = st.session_state.data.copy()
                
                if len(df) < 50:
                    st.error("Need at least 50 data points!")
                    return
                    
                with st.spinner("Calculating technical indicators..."):
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['RSI'] = compute_rsi(df['Close'])
                    df = df.dropna().reset_index(drop=True)
                    
                    if len(df) < 30:
                        st.error("Too many NaN values after feature creation!")
                        return
                        
                    st.session_state.data = df
                    st.session_state.steps['features_created'] = True
                    
                    st.write("### Feature Correlation Matrix:")
                    corr_matrix = df.corr()
                    fig = px.imshow(corr_matrix, text_auto=".2f", 
                                    color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Feature engineering failed: {str(e)}")

    # Step 4: Data Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("✂️ Split Dataset"):
            try:
                df = st.session_state.data.copy()
                required_features = ['SMA_20', 'SMA_50', 'RSI']
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    st.error(f"Missing features: {', '.join(missing_features)}")
                    return
                    
                X = df[required_features]
                y = df['Close'].values  
                
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
                
                st.write("### Dataset Split Visualization:")
                split_df = pd.DataFrame({
                    'Set': ['Train', 'Test'],
                    'Count': [len(X_train), len(X_test)]
                })
                fig = px.pie(split_df, values='Count', names='Set', 
                            color_discrete_sequence=['#2a4a7c', '#3b6ea5'])
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Data splitting failed: {str(e)}")

    # Step 5: Model Training
    if st.session_state.steps.get('split'):
        st.header("5. Model Training")
        
        if st.button("🎯 Train Model"):
            try:
                if not st.session_state.get('X_train'):
                    st.error("Training data not found!")
                    return
                    
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
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
                
                if model is None or X_test is None:
                    st.error("Missing model or test data!")
                    return
                    
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
                fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, 
                            name='Actual', line=dict(color='#2a4a7c')))
                fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_pred,
                            name='Predicted', line=dict(color='#4CAF50')))
                fig.update_layout(title="Actual vs Predicted Prices",
                                xaxis_title="Index",
                                yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                
                if model_type == "Random Forest":
                    st.write("### Feature Importance:")
                    importance = model.feature_importances_
                    features = ['SMA_20', 'SMA_50', 'RSI']
                    fig = px.bar(x=features, y=importance, 
                                color=features, 
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)

                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Predictions", csv, 
                                  "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    main()
