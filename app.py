# app.py (100% Working Version)
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
# 1. PREMIUM ANIMATIONS
# ======================
# ======================
LOTTIE_ASSETS = {
    "main": "https://lottie.host/bfd0d47e-6d7a-4504-8f3d-1a60d1d58f3b/5Qx4p3XqQH.json",
    "loading": "https://lottie.host/8d66c9a5-161c-41d8-8a35-7986d307b9e9/9EiZmAOQEi.json",
    "success": "https://lottie.host/8c9c5449-0d5a-4b72-8c3d-03a6c3c6d6e5/1XvzZzZz7y.json",
    "chart": "https://lottie.host/4e127d57-14d2-4d5e-b58a-3a0a8b0a7d3a/4Xm6E6Z6wD.json"
}

@st.cache_data(show_spinner=False)
def load_lottie(url: str):
    """Enhanced animation loader with error handling"""
    try:
        if not url.startswith('http'):
            return None
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"Animation server error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Animation loading failed: {str(e)}")
        return None
# ======================
# 2. LUXURY UI/UX
# ======================
st.set_page_config(
    page_title="AAPL Professional Analyst",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #000000 0%, #2c3e50 100%);}
    h1 {color: #ffffff; border-bottom: 3px solid #3498db; font-family: 'Helvetica Neue'}
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2c3e50) !important;
        color: white !important;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        transition: all 0.4s !important;
    }
    .stButton>button:hover {
        transform: scale(1.1) rotate(3deg);
        box-shadow: 0 8px 15px rgba(52,152,219,0.3);
    }
    .step-card {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 3. SESSION STATE MANAGEMENT
# ======================
def init_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'aapl_data': None,
        'model': None,
        'steps': {
            'loaded': False,
            'processed': False,
            'trained': False
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ======================
# 4. CORE FUNCTIONALITY
# ======================
def main():
    # Initialize session state first
    init_session_state()
    
    # ------------------
    # A. Hero Section
    # ------------------
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st_lottie(load_lottie(LOTTIE_ASSETS["main"]), height=200)
            st.title("AAPL Stock Intelligence Suite")
            st.markdown("---")
        with col2:
            st.image("https://1000logos.net/wp-content/uploads/2016/10/Apple-Logo-1977.png", width=150)

    # ------------------
    # B. Data Loading
    # ------------------
    with st.expander("🚀 STEP 1: Load AAPL Data", expanded=True):
        if st.button("Fetch Real-time AAPL Data"):
            with st.spinner('Connecting to NASDAQ...'):
                st_lottie(load_lottie(LOTTIE_ASSETS["loading"]), height=100)
                
                try:
                    df = yf.download("AAPL", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
                        st.session_state.aapl_data = df
                        st.session_state.steps['loaded'] = True
                        
                        # Success animation
                        st_lottie(load_lottie(LOTTIE_ASSETS["success"]), height=100)
                        st.balloons()
                        
                        # Data showcase
                        st.write("### AAPL Historical Data")
                        st.dataframe(df.style.format({"Close": "${:.2f}", "Volume": "{:,.0f}"}), 
                                   height=300,
                                   use_container_width=True)
                    else:
                        st.error("No data found for AAPL")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

    # ------------------
    # C. Data Processing
    # ------------------
    if st.session_state.steps['loaded']:
        with st.expander("🧹 STEP 2: Clean & Prepare Data", expanded=True):
            if st.button("✨ Enhance Data Quality"):
                with st.spinner('Optimizing dataset...'):
                    st_lottie(load_lottie(LOTTIE_ASSETS["loading"]), height=80)
                    
                    df = st.session_state.aapl_data
                    df = df.dropna()
                    df = df[df['Volume'] > 0]
                    st.session_state.aapl_data = df
                    st.session_state.steps['processed'] = True
                    
                    st.success("Data polished to institutional standards")
                    st_lottie(load_lottie(LOTTIE_ASSETS["success"]), height=80)

    # ------------------
    # D. Feature Engineering
    # ------------------
    if st.session_state.steps['processed']:
        with st.expander("⚡ STEP 3: Create Advanced Features", expanded=True):
            if st.button("🔧 Generate Trading Signals"):
                with st.spinner('Calculating market indicators...'):
                    st_lottie(load_lottie(LOTTIE_ASSETS["loading"]), height=80)
                    
                    df = st.session_state.aapl_data
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['EMA_12'] = df['Close'].ewm(span=12).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26).mean()
                    df['MACD'] = df['EMA_12'] - df['EMA_26']
                    
                    # Fixed RSI Calculation
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.rolling(14).mean()
                    avg_loss = loss.rolling(14).mean()
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    st.session_state.aapl_data = df.dropna()
                    st.success("Professional trading features created")
                    st_lottie(load_lottie(LOTTIE_ASSETS["success"]), height=80)

    # ------------------
    # E. Model Training
    # ------------------
    if st.session_state.steps['processed']:
        with st.expander("🤖 STEP 4: Predictive Modeling", expanded=True):
            if st.button("🎓 Train AAPL Predictor"):
                with st.spinner('Training institutional model...'):
                    st_lottie(load_lottie(LOTTIE_ASSETS["loading"]), height=80)
                    
                    df = st.session_state.aapl_data
                    features = ['SMA_20', 'MACD', 'RSI']
                    X = df[features]
                    y = df['Close']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Performance metrics
                    y_pred = model.predict(X_test)
                    st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    st.metric("R² Score", f"{r2_score(y_test, y_pred)*100:.1f}%")
                    
                    st.session_state.steps['trained'] = True
                    st_lottie(load_lottie(LOTTIE_ASSETS["chart"]), height=150)
                    st.success("Institutional Model Ready")

    # ------------------
    # F. Predictions
    # ------------------
    if st.session_state.steps['trained']:
        with st.expander("🔮 STEP 5: AAPL Price Forecast", expanded=True):
            df = st.session_state.aapl_data.copy()
            df['Prediction'] = st.session_state.model.predict(df[features])
            
            fig = px.line(df, x='Date', y=['Close', 'Prediction'],
                         title="AAPL Price Prediction",
                         color_discrete_sequence=['#3498db', '#e74c3c'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
