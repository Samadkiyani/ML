# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from streamlit_lottie import st_lottie
import json
import requests
from io import StringIO

# Configure page
st.set_page_config(
    page_title="FinML Analyst",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved Lottie animation handling with unique keys
def load_lottie(url: str, fallback_path: str = None):
    """Load Lottie animation with error handling and fallback"""
    try:
        if url.startswith("http"):
            r = requests.get(url)
            return r.json() if r.status_code == 200 else None
        elif fallback_path:
            with open(fallback_path) as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Animation error: {str(e)}")
        return None

# Reliable animation sources with fallbacks
LOTTIE_URLS = {
    "loading": "https://assets2.lottiefiles.com/packages/lf20_Stt1R6.json",
    "success": "https://assets9.lottiefiles.com/packages/lf20_auiqr3if.json"
}

# Custom CSS with proper scoping
st.markdown("""
<style>
    .main {background: #f0f2f6;}
    h1 {color: #2c3e50; border-bottom: 3px solid #3498db;}
    .stButton>button {border-radius: 8px; transition: transform 0.2s;}
    .stButton>button:hover {transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state with proper keys
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {
            'loaded': False,
            'processed': False,
            'features_created': False,
            'split': False,
            'trained': False
        }

    # Welcome section with unique keys
    st.title("📈 FinTech Machine Learning Analyst")
    st.markdown("---")
    
    # Animation with unique key per instance
    loading_anim = load_lottie(LOTTIE_URLS["loading"])
    if loading_anim:
        st_lottie(loading_anim, speed=1, height=200, key="loading_anim")
    else:
        st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", 
                width=300, use_column_width=False)

    # Sidebar with proper key management
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", 
                             ["Yahoo Finance", "Upload CSV"],
                             key="data_source_radio")
        
        model_type = st.selectbox(
            "Select Model:",
            ["Linear Regression", "Random Forest", "K-Means Clustering"],
            key="model_selectbox"
        )
        
        # Reset button with proper state management
        if st.button("🔄 Reset All", key="reset_button"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()

    # Data loading section with unique keys
    with st.expander("📂 Step 1: Load Data", expanded=True):
        if st.button("🚀 Load Data", key="load_data_button"):
            try:
                # Data loading logic...
                # (Keep your existing data loading code here)
                
                # Success animation with unique key
                success_anim = load_lottie(LOTTIE_URLS["success"])
                if success_anim:
                    st_lottie(success_anim, speed=1, height=150, key="success_anim_1")
                else:
                    st.image("https://media.giphy.com/media/3ohzdIuqJoo8QdKmlW/giphy.gif",
                            width=200, use_column_width=False)
                
                st.session_state.steps['loaded'] = True
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Remaining steps with proper key management...
    # (Maintain this pattern of unique keys for all interactive elements)

if __name__ == "__main__":
    main()
