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
    page_title="Financial ML Pro",
    page_icon="💹",
    layout="wide"
)

# Load Lottie animations safely
def load_lottie(url: str):
    """Load Lottie animation with error handling"""
    try:
        if url.startswith("http"):
            r = requests.get(url, timeout=10)
            return r.json() if r.status_code == 200 else None
        return None
    except Exception:
        return None

# Verified animation URLs
ANIMATIONS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_ysrn2iwp.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_au03ianj.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json"
}

# Custom CSS with smooth animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main {background: #f8fafc;}
    h1 {color: #1e3a8a; animation: fadeIn 1s;}
    .stButton>button {
        background: #3b82f6 !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(59,130,246,0.3) !important;
    }
    .section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        animation: fadeIn 0.8s;
    }
    .success-anim {animation: fadeIn 0.8s;}
</style>
""", unsafe_allow_html=True)

def show_animation(url: str, height=200):
    """Safe animation display with fallback"""
    anim = load_lottie(url)
    if anim:
        st_lottie(anim, height=height)
    else:
        st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", width=300)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False}

    # --- Header Section ---
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            show_animation(ANIMATIONS["main"], height=300)
        with col2:
            st.title("Financial ML Analyzer")
            st.markdown("---")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("📈 Stock Symbol:", "AAPL")
            start_date = st.date_input("📅 Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("📅 End Date:")
        else:
            uploaded_file = st.file_uploader("📤 Upload CSV:", type=["csv"])

    # --- Data Loading Section ---
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("📥 Step 1: Load Data")
        
        if st.button("🚀 Load Data"):
            try:
                with st.spinner("Loading data..."):
                    show_animation(ANIMATIONS["loading"], height=100)
                    
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date).reset_index()
                    else:
                        df = pd.read_csv(uploaded_file)
                    
                    st.session_state.data = df
                    st.session_state.steps = {'loaded': True, 'processed': False}
                    
                    st.success("✅ Data loaded successfully!")
                    show_animation(ANIMATIONS["success"], height=100)
                    st.dataframe(df.head().style.set_properties(**{
                        'background-color': '#f8fafc',
                        'color': '#1e293b'
                    }))

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Data Processing Flow ---
    if st.session_state.steps['loaded']:
        with st.container():
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.header("🧹 Step 2: Clean Data")
            
            if st.button("✨ Clean Data"):
                with st.spinner("Cleaning..."):
                    df = st.session_state.data.dropna()
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    
                    st.success("✅ Cleaning completed!")
                    show_animation(ANIMATIONS["success"], height=100)
                    st.write("Missing values:", df.isna().sum())
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.steps['processed']:
            # Feature Engineering
            with st.container():
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.header("⚙️ Step 3: Feature Engineering")
                
                if st.button("🔧 Generate Features"):
                    df = st.session_state.data
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    st.session_state.data = df.dropna()
                    
                    st.success("✅ Features created!")
                    show_animation(ANIMATIONS["success"], height=100)
                    st.dataframe(df.tail())
                st.markdown('</div>', unsafe_allow_html=True)

            # Model Training & Evaluation
            with st.container():
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.header("🤖 Step 4: Model Training")
                
                if st.button("🎓 Train Model"):
                    df = st.session_state.data
                    X = df[['SMA_20', 'SMA_50']]
                    y = df['Close']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False)
                    
                    model = LinearRegression().fit(X_train, y_train)
                    st.session_state.model = model
                    
                    st.success("✅ Model trained!")
                    show_animation(ANIMATIONS["success"], height=100)
                    
                    # Evaluation
                    y_pred = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                    with col2:
                        st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                    
                    fig = px.line(
                        x=X_test.index, y=[y_test, y_pred],
                        labels={'value': 'Price', 'variable': 'Legend'},
                        color_discrete_sequence=['#3b82f6', '#ef4444']
                    )
                    st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
