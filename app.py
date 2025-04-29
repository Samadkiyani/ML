# app.py
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
            ticker = st.text_input("Stock Ticker (e.g., AAPL):", "AAPL")
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

    # Step 1: Load Data
    st.header("1. Data Acquisition")
    if st.button("🚀 Load Data"):
        try:
            if data_source == "Yahoo Finance":
                with st.spinner("Fetching market data..."):
                    df = yf.download(ticker, start=start_date, end=end_date)
                    if df.empty:
                        st.error("Invalid ticker or date range!")
                        return
                    df = df.reset_index()
                    st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", 
                           caption="Market data loaded!")
            else:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file).reset_index(drop=True)
                    st.success("CSV file loaded successfully!")
                else:
                    st.warning("Please upload a CSV file!")
                    return

            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.write("### Data Preview:")
            st.dataframe(df.head().style.format("{:.2f}"), height=200)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    # Step 2: Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Data"):
                try:
                    df = st.session_state.data.copy()
                    
                    st.write("### Missing Values Analysis:")
                    missing = pd.DataFrame({
                        'Feature': df.columns,
                        'Missing Values': df.isnull().sum().values
                    })
                    
                    fig = px.bar(missing, 
                                x='Missing Values', 
                                y='Feature',
                                orientation='h',
                                labels={'Feature': 'Features', 'Missing Values': 'Count'},
                                color='Feature',
                                color_discrete_sequence=['#2a4a7c'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Clean data
                    df = df.dropna().reset_index(drop=True)
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success("Data cleaning completed! Missing values removed.")

                except Exception as e:
                    st.error(f"Data cleaning failed: {str(e)}")

        with col2:
            if st.session_state.steps['processed']:
                try:
                    st.write("### Cleaned Data Statistics:")
                    # Fixed UnboundLocalError by using session_state data
                    clean_df = st.session_state.data
                    st.dataframe(clean_df.describe().style.format("{:.2f}"), height=300)
                except AttributeError:
                    st.warning("No data available for display")
                except Exception as e:
                    st.error(f"Error displaying statistics: {str(e)}")

    # [Rest of the code remains identical to previous version...]

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    main()
