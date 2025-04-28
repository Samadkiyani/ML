# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# ======================
# 1. ANIMATION SETTINGS
# ======================
LOTTIE_URLS = {
    "main": "https://lottie.host/bfd0d47e-6d7a-4504-8f3d-1a60d1d58f3b/5Qx4p3XqQH.json",
    "loading": "https://lottie.host/8d66c9a5-161c-41d8-8a35-7986d307b9e9/9EiZmAOQEi.json",
    "success": "https://lottie.host/8c9c5449-0d5a-4b72-8c3d-03a6c3c6d6e5/1XvzZzZz7y.json",
    "chart": "https://lottie.host/4e127d57-14d2-4d5e-b58a-3a0a8b0a7d3a/4Xm6E6Z6wD.json"
}

@st.cache_data
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ======================
# 2. APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="AAPL Stock Analyst",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
<style>
    .main {background: #000000;}
    h1 {color: #ffffff; border-bottom: 3px solid #3498db;}
    .stButton>button {
        background: #3498db !important;
        color: white !important;
        border-radius: 25px;
        padding: 12px 24px;
        transition: transform 0.3s;
    }
    .stButton>button:hover {transform: scale(1.05);}
    .step-card {background: rgba(255,255,255,0.1); border-radius: 15px; padding: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ======================
# 3. SESSION STATE SETUP
# ======================
if 'aapl_data' not in st.session_state:
    st.session_state.aapl_data = None
if 'steps' not in st.session_state:
    st.session_state.steps = {'loaded': False, 'processed': False, 'trained': False}

# ======================
# 4. MAIN APP FUNCTIONALITY
# ======================
def main():
    st.title("AAPL Stock Analysis Suite")
    
    # ------------------
    # A. Data Loading Section
    # ------------------
    with st.expander("STEP 1: Load AAPL Data", expanded=True):
        if st.button("🚀 Load AAPL Data"):
            with st.spinner('Fetching AAPL data from Yahoo Finance...'):
                try:
                    # Load animation
                    st_lottie(load_lottie(LOTTIE_URLS["loading"]), height=100)
                    
                    # Fetch AAPL data
                    df = yf.download("AAPL", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
                    df = df.reset_index()
                    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
                    
                    # Store in session state
                    st.session_state.aapl_data = df
                    st.session_state.steps['loaded'] = True
                    
                    # Show success
                    st_lottie(load_lottie(LOTTIE_URLS["success"]), height=100)
                    st.success("AAPL Data Loaded Successfully!")
                    
                    # Show data preview
                    st.dataframe(df.head().style.format({"Close": "${:.2f}"}), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Failed to load AAPL data: {str(e)}")

    # ------------------
    # B. Data Processing (Visible after loading)
    # ------------------
    if st.session_state.steps['loaded']:
        with st.expander("STEP 2: Clean Data", expanded=True):
            if st.button("✨ Clean AAPL Data"):
                with st.spinner('Cleaning dataset...'):
                    df = st.session_state.aapl_data
                    df = df.dropna()
                    df = df[df['Volume'] > 0]
                    st.session_state.aapl_data = df
                    st.session_state.steps['processed'] = True
                    st.success("Data Cleaning Complete!")
                    st_lottie(load_lottie(LOTTIE_URLS["success"]), height=80)

    # ------------------
    # C. Feature Engineering
    # ------------------
    if st.session_state.steps['processed']:
        with st.expander("STEP 3: Create Features", expanded=True):
            if st.button("🔧 Generate Features"):
                with st.spinner('Creating technical indicators...'):
                    df = st.session_state.aapl_data
                    
                    # Calculate technical indicators
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    
                    # RSI Calculation (fixed)
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.rolling(14).mean()
                    avg_loss = loss.rolling(14).mean()
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    st.session_state.aapl_data = df.dropna()
                    st.session_state.steps['trained'] = True
                    st.success("Feature Engineering Complete!")
                    st_lottie(load_lottie(LOTTIE_URLS["success"]), height=80)

    # ------------------
    # D. Model Training
    # ------------------
    if st.session_state.steps['trained']:
        with st.expander("STEP 4: Train Model", expanded=True):
            if st.button("🎓 Train Prediction Model"):
                with st.spinner('Training model...'):
                    df = st.session_state.aapl_data
                    features = ['SMA_20', 'SMA_50', 'RSI']
                    target = 'Close'
                    
                    X = df[features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Store model and show results
                    st.session_state.model = model
                    y_pred = model.predict(X_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    with col2:
                        st.metric("R² Score", f"{r2_score(y_test, y_pred)*100:.1f}%")
                    
                    st_lottie(load_lottie(LOTTIE_URLS["chart"]), height=150)
                    st.success("Model Training Complete!")

    # ------------------
    # E. Predictions
    # ------------------
    if st.session_state.get('model'):
        with st.expander("STEP 5: View Predictions", expanded=True):
            df = st.session_state.aapl_data.copy()
            df['Prediction'] = st.session_state.model.predict(df[['SMA_20', 'SMA_50', 'RSI']])
            
            fig = px.line(df, x='Date', y=['Close', 'Prediction'],
                         title="AAPL Price Predictions",
                         color_discrete_sequence=['#3498db', '#e74c3c'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
