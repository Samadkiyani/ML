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

# Configure page
st.set_page_config(
    page_title="FinVision Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lottie animations with fallback
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

ANIMATIONS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_ysrn2iwp.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_au03ianj.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json",
    "chart": "https://assets1.lottiefiles.com/packages/lf20_ujvx3qxj.json"
}

def show_animation(url: str, height=200):
    anim = load_lottie(url)
    if anim:
        st_lottie(anim, height=height, key=f"anim_{url}")
    else:
        st.empty()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: #4f46e5 !important;
        transition: all 0.3s ease !important;
    }
</style>
""", unsafe_allow_html=True)

def validate_dates(start: datetime, end: datetime):
    if start >= end:
        st.error("End date must be after start date")
        return False
    return True

def main():
    st.session_state.setdefault('data', None)
    st.session_state.setdefault('steps', {'loaded': False, 'processed': False})

    # Animated Header
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            show_animation(ANIMATIONS["main"], height=250)
        with col2:
            st.title("📈 FinVision Pro")
            st.markdown("---")

    # Sidebar Configuration
    with st.sidebar:
        show_animation(ANIMATIONS["chart"], height=120)
        st.header("Settings")
        data_source = st.radio("Data Source", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Symbol", "AAPL").upper()
            start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date")
            if not validate_dates(start_date, end_date):
                return
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Data Loading
    if not st.session_state.steps['loaded']:
        with st.container():
            if st.button("🚀 Load Data"):
                try:
                    with st.spinner("Loading..."):
                        show_animation(ANIMATIONS["loading"], height=100)
                        
                        if data_source == "Yahoo Finance":
                            df = yf.download(ticker, start=start_date, end=end_date)
                            df = df.reset_index()  # Ensure Date becomes a column
                        else:
                            df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                            df = df.set_index('Date')
                        
                        st.session_state.data = df.dropna()
                        st.session_state.steps['loaded'] = True
                        show_animation(ANIMATIONS["success"], height=100)
                        st.dataframe(df.head())
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Data Processing Pipeline
    if st.session_state.steps['loaded']:
        df = st.session_state.data
        
        # Feature Engineering
        if not st.session_state.steps['processed']:
            with st.container():
                if st.button("🔧 Generate Features"):
                    with st.spinner("Creating features..."):
                        df['SMA_20'] = df['Close'].rolling(20).mean()
                        df['SMA_50'] = df['Close'].rolling(50).mean()
                        st.session_state.data = df.dropna()
                        st.session_state.steps['processed'] = True
                        show_animation(ANIMATIONS["success"], height=100)
                        st.dataframe(df.tail())

        # Model Training & Prediction
        if st.session_state.steps['processed']:
            with st.container():
                if st.button("🤖 Run Analysis"):
                    X = df[['SMA_20', 'SMA_50']]
                    y = df['Close']
                    
                    # Data splitting with index handling
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False
                    )
                    
                    # Model training
                    model = LinearRegression().fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Ensure 1D arrays
                    y_pred = y_pred.ravel()
                    y_test = y_test.ravel()
                    
                    # Get dates correctly
                    try:
                        dates = df.loc[X_test.index, 'Date'].values
                    except KeyError:
                        dates = X_test.index.to_numpy()

                    # Create comparison DataFrame with strict 1D arrays
                    comparison_df = pd.DataFrame({
                        'Date': dates.ravel(),    # Force 1D
                        'Actual': y_test.ravel(),
                        'Predicted': y_pred.ravel()
                    }).melt(id_vars='Date', var_name='Type', value_name='Price')

                    # Visualization
                    fig = px.line(
                        comparison_df,
                        x='Date',
                        y='Price',
                        color='Type',
                        title="Price Predictions",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>MSE</h3>
                            <h2>{mean_squared_error(y_test, y_pred):.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>R² Score</h3>
                            <h2>{r2_score(y_test, y_pred):.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    show_animation(ANIMATIONS["success"], height=100)

if __name__ == "__main__":
    main()
