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
    page_title="Multi-Stock ML Analyst",
    page_icon="💹",
    layout="wide"
)

# Enhanced Lottie animations
LOTTIE_URLS = {
    "loading": "https://assets1.lottiefiles.com/packages/lf20_5njp3vgg.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_auwiessx.json",
    "finance": "https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json"
}

def load_lottie(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);}
    h1 {color: #2b5876; border-bottom: 3px solid #4e4376;}
    .stButton>button {
        background: linear-gradient(45deg, #4e4376, #2b5876);
        color: white;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {transform: scale(1.05);}
    .stock-card {border-radius: 10px; padding: 20px; margin: 10px 0; 
                background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

def main():
    # Animated header
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 Multi-Stock ML Analyst")
            st.markdown("---")
        with col2:
            finance_anim = load_lottie(LOTTIE_URLS["finance"])
            if finance_anim:
                st_lottie(finance_anim, height=150, key="header_anim")

    # Initialize session state
    if 'stocks' not in st.session_state:
        st.session_state.stocks = {}
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", 
                             ["Yahoo Finance", "Upload CSV"],
                             key="data_source")
        
        if st.button("🔄 Reset All", help="Clear all loaded data"):
            st.session_state.clear()
            st.experimental_rerun()

        if data_source == "Yahoo Finance":
            st.subheader("Stock Selection")
            tickers = st.text_input("Enter Tickers (comma separated):", 
                                  "AAPL, MSFT, GOOGL",
                                  help="E.g., AAPL, MSFT, TSLA")
            start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date:")
        else:
            uploaded_files = st.file_uploader("Upload CSV Files:", 
                                            type=["csv"],
                                            accept_multiple_files=True)

    # Data loading section
    with st.expander("📥 Step 1: Load Data", expanded=True):
        if st.button("🚀 Load Data", key="load_data"):
            try:
                with st.spinner('Fetching data...'):
                    if data_source == "Yahoo Finance":
                        ticker_list = [t.strip() for t in tickers.split(',')]
                        df = yf.download(ticker_list, start=start_date, end=end_date)
                        
                        if df.empty:
                            st.error("No data found for these tickers")
                            return
                            
                        # Process multi-index data
                        for ticker in ticker_list:
                            if ('Close', ticker) in df.columns:
                                st.session_state.stocks[ticker] = {
                                    'data': df.xs(ticker, axis=1, level=1).reset_index()
                                }
                                
                    else:
                        for uploaded_file in uploaded_files:
                            df = pd.read_csv(uploaded_file)
                            ticker = uploaded_file.name.split('.')[0]
                            st.session_state.stocks[ticker] = {'data': df}
                    
                    if lottie_success:
                        st_lottie(load_lottie(LOTTIE_URLS["success"]), 
                                height=100, key="load_success")
                    st.success(f"Loaded {len(st.session_state.stocks)} stocks!")
                    
                    # Display loaded stocks
                    cols = st.columns(3)
                    for idx, (ticker, data) in enumerate(st.session_state.stocks.items()):
                        with cols[idx % 3]:
                            with st.container():
                                st.markdown(f"""
                                <div class='stock-card'>
                                    <h4>{ticker}</h4>
                                    <p>Rows: {len(data['data'])}</p>
                                    <p>Columns: {len(data['data'].columns)}</p>
                                </div>
                                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Only show processing if data exists
    if st.session_state.stocks:
        # Data preprocessing
        with st.expander("🧹 Step 2: Preprocess Data"):
            if st.button("✨ Clean All Data"):
                with st.spinner('Cleaning...'):
                    for ticker in st.session_state.stocks:
                        df = st.session_state.stocks[ticker]['data']
                        df = df.dropna()
                        st.session_state.stocks[ticker]['data'] = df
                    st.success("Data cleaned across all stocks!")

        # Feature engineering
        with st.expander("⚡ Step 3: Create Features"):
            if st.button("🔧 Generate Features"):
                with st.spinner('Creating features...'):
                    for ticker in st.session_state.stocks:
                        df = st.session_state.stocks[ticker]['data']
                        df['SMA_20'] = df['Close'].rolling(20).mean()
                        df['SMA_50'] = df['Close'].rolling(50).mean()
                        df['Returns'] = df['Close'].pct_change()
                        st.session_state.stocks[ticker]['data'] = df.dropna()
                    st.success("Features created for all stocks!")

        # Model training
        with st.expander("🤖 Step 4: Train Model"):
            if st.button("🎓 Train Model"):
                try:
                    # Combine all stock data
                    combined_df = pd.concat(
                        [stock['data'] for stock in st.session_state.stocks.values()]
                    )
                    
                    # Prepare features
                    X = combined_df[['SMA_20', 'SMA_50']]
                    y = combined_df['Close']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False)
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Show metrics
                    y_pred = model.predict(X_test)
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                    
                    # Success animation
                    st_lottie(load_lottie(LOTTIE_URLS["success"]), height=100)
                    st.success("Model trained on all stocks!")

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        # Prediction visualization
        if st.session_state.model:
            with st.expander("🔮 Step 5: Predictions"):
                selected_ticker = st.selectbox("Select Stock:", list(st.session_state.stocks.keys()))
                
                df = st.session_state.stocks[selected_ticker]['data']
                df['Prediction'] = st.session_state.model.predict(df[['SMA_20', 'SMA_50']])
                
                fig = px.line(df, x='Date', y=['Close', 'Prediction'],
                            title=f"{selected_ticker} Predictions",
                            color_discrete_sequence=['#4e4376', '#2b5876'])
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
