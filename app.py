# app.py (AAPL-Specific Professional Edition)
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
LOTTIE_ASSETS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_mDnmhAgZkb.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_6wutsrox.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_jbrw3hq5.json",
    "chart": "https://assets1.lottiefiles.com/packages/lf20_0skurerf.json"
}

@st.cache_data
def load_lottie(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
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
# 3. CORE FUNCTIONALITY
# ======================
def main():
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

    # Initialize session state
    if 'aapl_data' not in st.session_state:
        st.session_state.aapl_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False, 'trained': False}

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
                    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                           df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
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
