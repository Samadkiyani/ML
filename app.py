# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Lottie animations
ANIMATIONS = {
    "main": "https://assets1.lottiefiles.com/packages/lf20_ysrn2iwp.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_au03ianj.json",
    "loading": "https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json"
}

def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# Professional CSS enhancements
st.markdown("""
<style>
    .metric-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        background: #6366f1 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(99,102,241,0.3) !important;
    }
    .section-title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.session_state.setdefault('data', None)
    st.session_state.setdefault('model', None)
    st.session_state.setdefault('steps', {'loaded': False, 'processed': False})

    # Animated Header
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            anim = load_lottie(ANIMATIONS["main"])
            if anim:
                st_lottie(anim, height=200, key="main-anim")
        with col2:
            st.title("FinVision Pro")
            st.markdown("""
            <div style="border-left: 4px solid #6366f1; padding-left: 1rem; margin: 1rem 0;">
                <p style="color: #4b5563; margin: 0;">AI-Powered Financial Analytics Platform</p>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Symbol", "AAPL").upper()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
            with col2:
                end_date = st.date_input("End Date")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Data Loading and Processing
    if not st.session_state.steps['loaded']:
        if st.button("🚀 Load Market Data"):
            try:
                with st.spinner("Fetching financial data..."):
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date)
                        df = df.reset_index()
                    else:
                        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                        df = df.set_index('Date')
                    
                    st.session_state.data = df.dropna()
                    st.session_state.steps['loaded'] = True
                    
                    # Display initial analysis
                    with st.container():
                        st.subheader("📊 Initial Market Analysis")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df['Date'],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Market Data'
                        ))
                        fig.update_layout(
                            height=500,
                            xaxis_rangeslider_visible=False,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Data loading error: {str(e)}")

    # Advanced Analytics
    if st.session_state.steps['loaded']:
        df = st.session_state.data
        
        with st.expander("🔍 Advanced Analytics"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Technical Indicators")
                if st.button("Generate Moving Averages"):
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    st.session_state.data = df.dropna()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20-Day SMA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA'))
                    fig.update_layout(
                        height=400,
                        title="Price Trend Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            with col2:
                st.markdown("### Risk Metrics")
                if st.checkbox("Show Volatility Analysis"):
                    df['Daily Return'] = df['Close'].pct_change()
                    df['Volatility'] = df['Daily Return'].rolling(20).std() * np.sqrt(252)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Volatility'))
                    fig.update_layout(
                        height=400,
                        title="Annualized Volatility (20-Day Rolling)",
                        xaxis_title="Date",
                        yaxis_title="Volatility",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Predictive Analytics
        if st.button("📈 Run Predictive Analysis"):
            try:
                X = df[['SMA_20', 'SMA_50']]
                y = df['Close']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Ensure 1D arrays
                y_pred = np.asarray(y_pred).ravel()
                y_test = np.asarray(y_test).ravel()
                
                # Create results dataframe
                results = pd.DataFrame({
                    'Date': X_test.index.to_numpy().ravel(),
                    'Actual': y_test,
                    'Predicted': y_pred
                }).melt(id_vars='Date', var_name='Metric', value_name='Value')
                
                # Performance Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2>{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2>{r2_score(y_test, y_pred):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Error Margin</h3>
                        <h2>{np.mean(np.abs(y_test - y_pred)):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction Visualization
                fig = px.line(
                    results,
                    x='Date',
                    y='Value',
                    color='Metric',
                    title="Price Prediction Performance",
                    height=500,
                    template='plotly_white'
                )
                fig.update_layout(
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download feature
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name='market_predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
