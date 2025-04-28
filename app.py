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

# Configure page
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide"
)

# Load Lottie animations with enhanced error handling
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"⚠️ Animation load error: {str(e)}")
        return None

# Reliable animation URLs from lottiefiles.com's public CDN
LOTTIE_URLS = {
    "finance": "https://assets8.lottiefiles.com/packages/lf20_2znx3l3i.json",
    "success": "https://assets2.lottiefiles.com/packages/lf20_au03ianj.json",
    "loading": "https://assets9.lottiefiles.com/packages/lf20_raiw2hpe.json"
}

# Custom CSS with professional animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; animation: fadeIn 1s;}
    .stButton>button {
        background-color: #3498db !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-card {
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        animation: fadeIn 0.8s;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function with proper error handling"""
    
    # Initialize session state
    session_defaults = {
        'data': None,
        'model': None,
        'steps': {'loaded': False, 'processed': False}
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # --- Header Section ---
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            lottie_finance = load_lottieurl(LOTTIE_URLS["finance"])
            if lottie_finance:
                st_lottie(lottie_finance, height=250, key="header_anim")
            else:
                st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", 
                        width=300)
        with col2:
            st.title("💰 Financial ML Analyzer")
            st.markdown("---")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Symbol (e.g., AAPL):", "AAPL")
            start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date:")
        else:
            uploaded_file = st.file_uploader("Choose CSV:", type=["csv"])

    # --- Data Loading Section ---
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("📥 Step 1: Load Data")
        
        if st.button("🚀 Load Dataset"):
            with st.spinner("Fetching data..."):
                try:
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date)
                        df = df.reset_index()
                    else:
                        df = pd.read_csv(uploaded_file)
                        
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    
                    # Success animation
                    success_anim = load_lottieurl(LOTTIE_URLS["success"])
                    if success_anim:
                        st_lottie(success_anim, height=100, key="load_success")
                    
                    st.success("✅ Data loaded successfully!")
                    st.dataframe(df.head().style.set_properties(**{
                        'background-color': '#f8f9fa',
                        'color': '#2c3e50'
                    }))

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Subsequent Steps ---
    if st.session_state.steps['loaded']:
        # Data Cleaning Section
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.header("🧹 Step 2: Data Cleaning")
            
            if st.button("✨ Clean Data"):
                with st.spinner("Processing..."):
                    df = st.session_state.data
                    
                    # Data cleaning logic
                    st.write("Missing Values:", df.isnull().sum())
                    df = df.dropna()
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    
                    st.success("✅ Cleaning completed!")
                    st_lottie(load_lottieurl(LOTTIE_URLS["success"]), height=100)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature Engineering
        if st.session_state.steps.get('processed'):
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("⚙️ Step 3: Feature Engineering")
                
                if st.button("🔧 Generate Features"):
                    df = st.session_state.data
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    st.session_state.data = df.dropna()
                    
                    st.success("✅ Features created!")
                    st.dataframe(df.tail())
                st.markdown('</div>', unsafe_allow_html=True)

            # Model Training Section
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("🤖 Step 4: Model Training")
                
                if st.button("🎓 Train Model"):
                    df = st.session_state.data
                    X = df[['SMA_20', 'SMA_50']]
                    y = df['Close']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False)
                    
                    model = LinearRegression().fit(X_train, y_train)
                    st.session_state.model = model
                    
                    st.success("✅ Model trained successfully!")
                    st_lottie(load_lottieurl(LOTTIE_URLS["success"]), height=100)
                st.markdown('</div>', unsafe_allow_html=True)

            # Model Evaluation
            if st.session_state.model:
                with st.container():
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.header("📈 Step 5: Evaluation")
                    
                    if st.button("📊 Show Metrics"):
                        y_pred = st.session_state.model.predict(X_test)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                        with col2:
                            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                        
                        fig = px.line(
                            x=X_test.index, y=[y_test, y_pred],
                            labels={'value': 'Price'},
                            color_discrete_sequence=['#3498db', '#e74c3c']
                        )
                        st.plotly_chart(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
