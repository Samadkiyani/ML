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
import json
import requests

# Configure page
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide"
)

# Load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_loading = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_raiw2hpe.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_auwiessx.json")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #F5F5F5;}
    h1 {color: #003366;}
    .stButton>button {background-color: #004488; color: white; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.05);}
    .stSuccess {background-color: #DFF2BF;}
</style>
""", unsafe_allow_html=True)

def main():
    # Welcome Interface with animation
    st.title("Financial Machine Learning Application")
    st.markdown("---")
    
    if lottie_loading:
        st_lottie(lottie_loading, speed=1, height=200, key="main_anim")
    else:
        st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", 
                width=300, use_container_width=True)
    
    # Initialize session state with reset capability
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False}

    # Sidebar Configuration with reset button
    with st.sidebar:
        st.header("Data Configuration")
        data_source = st.radio("Select Data Source:", 
                             ["Yahoo Finance", "Upload Dataset"],
                             key="data_source")
        
        if st.button("🔄 Reset All"):
            st.session_state.clear()
            st.experimental_rerun()
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL",
                                 help="Enter valid Yahoo Finance ticker symbol")
            start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date:")
        else:
            uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"],
                                           help="Upload your financial dataset in CSV format")
    
    # Step 1: Load Data with enhanced error handling
    st.header("Step 1: Load Data")
    if st.button("Load Data", key="load_data"):
        try:
            with st.spinner('Loading data...'):
                if data_source == "Yahoo Finance":
                    df = yf.download(ticker, start=start_date, end=end_date)
                    if df.empty:
                        st.error(f"No data found for {ticker} in selected date range")
                        return
                    df = df.reset_index()
                else:
                    if not uploaded_file:
                        st.error("Please upload a CSV file first")
                        return
                    df = pd.read_csv(uploaded_file)
                    if df.empty:
                        st.error("Uploaded file is empty")
                        return
                
                st.session_state.data = df
                st.session_state.steps['loaded'] = True
                
                if lottie_success:
                    st_lottie(lottie_success, speed=1, height=100, key="load_success")
                st.success("Data loaded successfully!")
                
                st.write("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.steps['loaded'] = False

    # Rest of the code remains similar but add these fixes:
    # 1. Add key parameters to all buttons
    # 2. Use use_container_width instead of use_column_width
    # 3. Add proper error handling for all steps

if __name__ == "__main__":
    main()
