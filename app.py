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
from datetime import date

# ======================
# 1. PROFESSIONAL ANIMATIONS
# ======================
LOTTIE_URLS = {
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
    h1 {color: #2b5876; border-bottom: 3px solid #4e4376;}
    .stButton>button {
        background: linear-gradient(45deg, #4e4376, #2b5876);
        color: white;
        border-radius: 8px;
        transition: transform 0.3s;
    }
    .stButton>button:hover {transform: scale(1.05);}
    .ticker-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .step-container {border-left: 4px solid #4e4376; padding-left: 1.5rem;}
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
            header_anim = load_lottie(LOTTIE_URLS["main"])
            if header_anim:
                st_lottie(header_anim, height=150, key="header_anim")

    # Initialize session state
    if 'stocks' not in st.session_state:
        st.session_state.stocks = {}
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'cleaned': False, 'features': False, 'trained': False}

    # ------------------
    # B. Sidebar Config
    # ------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            tickers = st.text_input("Enter Stock Symbols (comma separated):", "AAPL, NDAQ, MSFT")
            start_date = st.date_input("Start Date:", date(2020, 1, 1))
            end_date = st.date_input("End Date:", date.today())
        else:
            uploaded_files = st.file_uploader("Upload CSV Files:", type=["csv"], accept_multiple_files=True)

        if st.button("🔄 Full Reset"):
            st.session_state.clear()
            st.rerun()

    # ------------------
    # C. Data Loading with Animations
    # ------------------
    with st.expander("📥 STEP 1: Load Market Data", expanded=True):
        if st.button("🚀 Load & Process Data"):
            with st.spinner('Fetching institutional data...'):
                st_lottie(load_lottie(LOTTIE_URLS["loading"]), height=100, key="load_anim")
                
                st.session_state.stocks.clear()
                current_date = date.today()
                
                if data_source == "Yahoo Finance":
                    if end_date > current_date:
                        st.error("End date cannot be in the future")
                        return
                        
                    ticker_list = [t.strip().upper() for t in tickers.split(',')]
                    for ticker in ticker_list:
                        try:
                            df = yf.download(ticker, start=start_date, end=end_date)
                            if not df.empty:
                                df = df.reset_index()
                                df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) 
                                            else col for col in df.columns]
                                st.session_state.stocks[ticker] = df
                                st.success(f"✅ {ticker} loaded")
                            else:
                                st.error(f"❌ No data for {ticker}")
                        except Exception as e:
                            st.error(f"❌ Failed to load {ticker}: {str(e)}")
                
                if st.session_state.stocks:
                    st.session_state.steps['loaded'] = True
                    st_lottie(load_lottie(LOTTIE_URLS["success"]), height=100, key="success_anim")
                    
                    # Display loaded stocks
                    cols = st.columns(3)
                    for idx, (ticker, data) in enumerate(st.session_state.stocks.items()):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class='ticker-card'>
                                <h4>{ticker}</h4>
                                <p>From: {data.iloc[0]['Date']}</p>
                                <p>To: {data.iloc[-1]['Date']}</p>
                                <p>Data Points: {len(data):,}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    st.rerun()

    # ------------------
    # D. Step-by-Step Process
    # ------------------
    if st.session_state.steps['loaded']:
        with st.container():
            st.markdown("""<div class='step-container'>""", unsafe_allow_html=True)
            
            # STEP 2: Data Cleaning
            with st.expander("🧹 STEP 2: Institutional Cleaning", expanded=True):
                if st.button("✨ Clean Data"):
                    with st.spinner('Professional cleaning...'):
                        for ticker in st.session_state.stocks:
                            df = st.session_state.stocks[ticker]
                            df = df.dropna().drop_duplicates()
                            df = df[df['Volume'] > 0]
                            st.session_state.stocks[ticker] = df
                        st.session_state.steps['cleaned'] = True
                        st.rerun()
            
            if st.session_state.steps['cleaned']:
                # STEP 3: Feature Engineering
                with st.expander("⚡ STEP 3: Advanced Features", expanded=True):
                    if st.button("🔧 Create Features"):
                        with st.spinner('Generating hedge-fund features...'):
                            for ticker in st.session_state.stocks:
                                df = st.session_state.stocks[ticker]
                                df['SMA_20'] = df['Close'].rolling(20).mean()
                                df['SMA_50'] = df['Close'].rolling(50).mean()
                                df['EMA_12'] = df['Close'].ewm(span=12).mean()
                                df['EMA_26'] = df['Close'].ewm(span=26).mean()
                                df['MACD'] = df['EMA_12'] - df['EMA_26']
                                df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                                      df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
                                st.session_state.stocks[ticker] = df.dropna()
                            st.session_state.steps['features'] = True
                            st.rerun()
                
                if st.session_state.steps['features']:
                    # STEP 4: Model Training
                    with st.expander("🤖 STEP 4: Portfolio Modeling", expanded=True):
                        if st.button("🎓 Train Model"):
                            with st.spinner('Training institutional model...'):
                                combined_df = pd.concat([df.assign(Ticker=ticker) 
                                                        for ticker, df in st.session_state.stocks.items()])
                                features = ['SMA_20', 'SMA_50', 'MACD', 'RSI']
                                X = combined_df[features]
                                y = combined_df['Close']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                                
                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                st.session_state.model = model
                                
                                y_pred = model.predict(X_test)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                                with col2:
                                    st.metric("R² Score", f"{r2_score(y_test, y_pred)*100:.1f}%")
                                
                                st.session_state.steps['trained'] = True
                                st_lottie(load_lottie(LOTTIE_URLS["analytics"]), height=150)
                                st.rerun()
                    
                    if st.session_state.steps['trained']:
                        # STEP 5: Predictions
                        with st.expander("🔮 STEP 5: Institutional Analytics", expanded=True):
                            selected_ticker = st.selectbox("Select Stock:", list(st.session_state.stocks.keys()))
                            df = st.session_state.stocks[selected_ticker]
                            df['Prediction'] = st.session_state.model.predict(df[features])
                            
                            fig = px.line(df, x='Date', y=['Close', 'Prediction'],
                                        title=f"{selected_ticker} Professional Forecast",
                                        color_discrete_map={'Close': '#4e4376', 'Prediction': '#2b5876'})
                            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
