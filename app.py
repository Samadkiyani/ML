# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="FinVision Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lottie animations
ANIMATIONS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_ysrn2iwp.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_au03ianj.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json"
}

def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# Custom CSS
st.markdown("""
<style>
    .pipeline-step {
        border-left: 4px solid #6366f1;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8fafc;
        border-radius: 8px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.session_state.setdefault('data', None)
    st.session_state.setdefault('model', None)
    st.session_state.setdefault('steps', {
        'loaded': False,
        'preprocessed': False,
        'engineered': False,
        'split': False,
        'trained': False,
        'evaluated': False,
    })

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
        data_source = st.radio("Data Source", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Symbol", "AAPL").upper()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
            with col2:
                end_date = st.date_input("End Date")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Main Content
    with st.container():
        st_lottie(load_lottie(ANIMATIONS["main"]), height=200) if not st.session_state.steps['loaded'] else st.empty()

    # Step 1: Load Data
    with st.expander("1. Load Data", expanded=True):
        if st.button("🚀 Load Data"):
            try:
                with st.spinner("Fetching data..."):
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date)
                        df = df.reset_index()
                    else:
                        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                    
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    
                    st.success("✅ Data loaded successfully!")
                    st.dataframe(df.head())
                    
                    # Initial Visualization
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close']
                    )])
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Step 2: Preprocessing
    if st.session_state.steps['loaded']:
        with st.expander("2. Data Preprocessing"):
            if st.button("🧹 Clean Data"):
                df = st.session_state.data
                initial_missing = df.isna().sum().sum()
                df = df.dropna()
                st.session_state.data = df
                st.session_state.steps['preprocessed'] = True
                
                st.success(f"✅ Removed {initial_missing} missing values")
                st.write("Missing values after cleaning:", df.isna().sum().to_frame().T)

    # Step 3: Feature Engineering
    if st.session_state.steps['preprocessed']:
        with st.expander("3. Feature Engineering"):
            if st.button("🔧 Generate Features"):
                df = st.session_state.data
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                         df['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
                st.session_state.data = df.dropna()
                st.session_state.steps['engineered'] = True
                
                st.success("✅ Features created!")
                st.write("New features:", df[['SMA_20', 'SMA_50', 'RSI']].tail())

    # Step 4: Train/Test Split
    if st.session_state.steps['engineered']:
        with st.expander("4. Train/Test Split"):
            if st.button("🎯 Split Data"):
                df = st.session_state.data
                X = df[['SMA_20', 'SMA_50', 'RSI']]
                y = df['Close']
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = \
                    train_test_split(X, y, test_size=0.2, shuffle=False)
                st.session_state.steps['split'] = True
                
                fig = px.pie(values=[len(y)-len(st.session_state.y_test), len(st.session_state.y_test)],
                             names=['Train', 'Test'],
                             title="Train-Test Split Ratio")
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Data split completed!")

    # Step 5: Model Training
    if st.session_state.steps['split']:
        with st.expander("5. Model Training"):
            if st.button("🎓 Train Model"):
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100)
                model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.model = model
                st.session_state.steps['trained'] = True
                
                st.success(f"✅ {model_type} training completed!")
                
                if model_type == "Random Forest":
                    importance = pd.DataFrame({
                        'Feature': st.session_state.X_train.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                else:
                    importance = pd.DataFrame({
                        'Feature': st.session_state.X_train.columns,
                        'Importance': model.coef_
                    }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)

    # Step 6: Evaluation
    if st.session_state.steps['trained']:
        with st.expander("6. Model Evaluation"):
            if st.button("📊 Evaluate Model"):
                model = st.session_state.model
                y_pred = model.predict(st.session_state.X_test)
                rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                r2 = r2_score(st.session_state.y_test, y_pred)
                st.session_state.steps['evaluated'] = True

                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("R² Score", f"{r2:.2f}")

                df_result = pd.DataFrame({
                    "Actual": st.session_state.y_test,
                    "Predicted": y_pred
                }).reset_index(drop=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_result["Actual"], name="Actual"))
                fig.add_trace(go.Scatter(y=df_result["Predicted"], name="Predicted"))
                fig.update_layout(title="Actual vs Predicted Closing Prices")
                st.plotly_chart(fig, use_container_width=True)

    # Step 7: Prediction
    if st.session_state.steps['evaluated']:
        with st.expander("7. Make Prediction"):
            st.markdown("🔮 Enter new feature values to predict future closing price:")
            sma_20 = st.number_input("SMA_20")
            sma_50 = st.number_input("SMA_50")
            rsi = st.number_input("RSI")

            if st.button("📈 Predict Closing Price"):
                input_data = np.array([[sma_20, sma_50, rsi]])
                prediction = st.session_state.model.predict(input_data)[0]
                st.success(f"📌 Predicted Closing Price: ${prediction:.2f}")

if __name__ == "__main__":
    main()
