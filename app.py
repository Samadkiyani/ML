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
import datetime

# ======================
# 1. PROFESSIONAL ANIMATIONS
# ======================
LOTTIE_ASSETS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_ubzh7row.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_jbrw3hq5.json",
    "analytics": "https://assets1.lottiefiles.com/packages/lf20_uzjfd3gd.json"
}

@st.cache_data
def load_lottie(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ======================
# 2. PROFESSIONAL UI/UX
# ======================
st.set_page_config(
    page_title="Pro Stock Analyst",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);}
    h1 {color: #2b5876; border-bottom: 3px solid #4e4376; font-family: 'Helvetica Neue'}
    .stButton>button {
        background: linear-gradient(45deg, #4e4376, #2b5876) !important;
        color: white !important;
        border-radius: 8px;
        transition: transform 0.3s, box-shadow 0.3s !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ticker-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s;
    }
    .ticker-card:hover {transform: translateY(-3px)}
</style>
""", unsafe_allow_html=True)

# ======================
# 3. CORE FUNCTIONALITY
# ======================
def main():
    # ------------------
    # A. Animated Header
    # ------------------
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("📈 Professional Stock Analysis Suite")
            st.markdown("---")
        with col2:
            header_anim = load_lottie(LOTTIE_ASSETS["main"])
            if header_anim:
                st_lottie(header_anim, height=150, key="header_anim")

    # Initialize session state
    if 'stocks' not in st.session_state:
        st.session_state.stocks = {}
    if 'model' not in st.session_state:
        st.session_state.model = None

    # ------------------
    # B. Sidebar Config
    # ------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. Data Source
        data_source = st.radio("Data Source:", 
                             ["Yahoo Finance", "Upload CSV"],
                             key="data_source")
        
        # 2. Ticker Input (Professional Format)
        if data_source == "Yahoo Finance":
            tickers = st.text_input("Enter Stock Symbols (comma separated):",
                                  "AAPL, NDAQ, MSFT",
                                  help="Enter NASDAQ symbols like NDAQ, INTC, AMZN")
            start_date = st.date_input("Start Date:", 
                                     datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:")
        else:
            uploaded_files = st.file_uploader("Upload Stock Data:", 
                                           type=["csv"],
                                           accept_multiple_files=True)

        # 3. Advanced Options
        with st.expander("Advanced Options"):
            st.checkbox("Enable Daily Updates", value=False)
            st.slider("Test Size (%)", 10, 40, 20)

        if st.button("🔄 Full Reset", type="primary"):
            st.session_state.clear()
            st.experimental_rerun()

    # ------------------
    # C. Data Loading (Professional Grade)
    # ------------------
    with st.expander("📥 STEP 1: Load Market Data", expanded=True):
        if st.button("🚀 Load & Process Data", key="load_data"):
            try:
                with st.spinner('Fetching market data...'):
                    anim = load_lottie(LOTTIE_ASSETS["loading"])
                    if anim:
                        st_lottie(anim, height=100, key="load_anim")

                    # Clean previous data
                    st.session_state.stocks.clear()

                    if data_source == "Yahoo Finance":
                        ticker_list = [t.strip() for t in tickers.split(',')]
                        
                        # Fetch all tickers at once
                        df = yf.download(ticker_list, 
                                       start=start_date,
                                       end=end_date,
                                       group_by='ticker')
                        
                        # Process each ticker
                        for ticker in ticker_list:
                            try:
                                if ticker in df.columns.levels[1]:
                                    stock_df = df.xs(ticker, axis=1, level=1)
                                    stock_df = stock_df.reset_index()
                                    stock_df.columns = [col.strftime('%Y-%m-%d') 
                                                      if isinstance(col, pd.Timestamp) 
                                                      else col 
                                                      for col in stock_df.columns]
                                    st.session_state.stocks[ticker] = stock_df
                            except Exception as e:
                                st.error(f"Error processing {ticker}: {str(e)}")

                    else:
                        for file in uploaded_files:
                            try:
                                ticker = file.name.split('.')[0].upper()
                                st.session_state.stocks[ticker] = pd.read_csv(file)
                            except Exception as e:
                                st.error(f"Error reading {file.name}: {str(e)}")

                    # Success Animation
                    success_anim = load_lottie(LOTTIE_ASSETS["success"])
                    if success_anim:
                        st_lottie(success_anim, height=100, key="success_anim")
                    
                    # Display Loaded Tickers
                    st.success(f"Successfully loaded {len(st.session_state.stocks)} stocks")
                    cols = st.columns(3)
                    for idx, (ticker, data) in enumerate(st.session_state.stocks.items()):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class='ticker-card'>
                                <h4>{ticker}</h4>
                                <p>Period: {data.iloc[0]['Date']} to {data.iloc[-1]['Date']}</p>
                                <p>Rows: {len(data):,}</p>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Fatal loading error: {str(e)}")

    # ------------------
    # D. Data Processing
    # ------------------
    if st.session_state.stocks:
        # 1. Preprocessing
        with st.expander("🧹 STEP 2: Clean & Prepare Data"):
            if st.button("✨ Professional Cleaning"):
                with st.spinner('Optimizing datasets...'):
                    for ticker in st.session_state.stocks:
                        df = st.session_state.stocks[ticker]
                        # Advanced cleaning
                        df = df.dropna().drop_duplicates()
                        df = df[df['Volume'] > 0]  # Filter zero-volume days
                        st.session_state.stocks[ticker] = df
                    st.success("Institutional-grade cleaning complete!")

        # 2. Feature Engineering
        with st.expander("⚡ STEP 3: Advanced Features"):
            if st.button("🔧 Generate Pro Features"):
                with st.spinner('Creating institutional features...'):
                    for ticker in st.session_state.stocks:
                        df = st.session_state.stocks[ticker]
                        # Professional technical indicators
                        df['SMA_20'] = df['Close'].rolling(20).mean()
                        df['SMA_50'] = df['Close'].rolling(50).mean()
                        df['EMA_12'] = df['Close'].ewm(span=12).mean()
                        df['EMA_26'] = df['Close'].ewm(span=26).mean()
                        df['MACD'] = df['EMA_12'] - df['EMA_26']
                        df['RSI'] = 100 - (100 / (1 + (
                            df['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
                            df['Close'].diff(1).clip(upper=0).abs().rolling(14).mean()
                        )))
                        st.session_state.stocks[ticker] = df.dropna()
                    st.success("Hedge-fund grade features created!")

        # 3. Model Training
        with st.expander("🤖 STEP 4: Institutional Modeling"):
            if st.button("🎓 Train Professional Model"):
                try:
                    # Combine all stocks
                    combined_df = pd.concat([
                        df.assign(Ticker=ticker) 
                        for ticker, df in st.session_state.stocks.items()
                    ])
                    
                    # Feature selection
                    features = ['SMA_20', 'SMA_50', 'MACD', 'RSI']
                    target = 'Close'
                    
                    X = combined_df[features]
                    y = combined_df[target]
                    
                    # Professional split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False)
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Institutional-grade metrics
                    y_pred = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    with col2:
                        st.metric("R² Score", f"{r2_score(y_test, y_pred)*100:.1f}%")
                    
                    # Success display
                    st_lottie(load_lottie(LOTTIE_ASSETS["analytics"]), 
                             height=150, key="model_anim")
                    st.success("Portfolio model trained successfully!")

                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")

        # 4. Predictions & Analytics
        if st.session_state.model:
            with st.expander("🔮 STEP 5: Professional Analytics"):
                selected_ticker = st.selectbox("Select Stock:", 
                                             list(st.session_state.stocks.keys()))
                
                df = st.session_state.stocks[selected_ticker]
                df['Prediction'] = st.session_state.model.predict(df[features])
                
                # Professional chart
                fig = px.line(df, x='Date', y=['Close', 'Prediction'],
                            title=f"{selected_ticker} Institutional Forecast",
                            color_discrete_map={
                                'Close': '#4e4376',
                                'Prediction': '#2b5876'
                            })
                fig.update_layout(
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
