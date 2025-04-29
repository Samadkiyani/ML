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
    "loading": "https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json",
    "chart": "https://assets1.lottiefiles.com/packages/lf20_ujvx3qxj.json"
}

# Custom CSS with modern design
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main {background: #f8fafc;}
    h1 {color: #1e3a8a; animation: fadeIn 1s; font-family: 'Arial Rounded MT Bold'}
    .stButton>button {
        background: #4f46e5 !important;
        color: white !important;
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(79,70,229,0.3) !important;
    }
    .section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        animation: fadeIn 0.8s;
    }
    .metric-card {
        background: #f1f5f9;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def show_animation(url: str, height=200):
    """Safe animation display with fallback"""
    anim = load_lottie(url)
    if anim:
        st_lottie(anim, height=height, key=f"anim_{url}")
    else:
        st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", width=300)

def validate_dates(start: datetime, end: datetime) -> bool:
    """Validate date range inputs"""
    if start >= end:
        st.error("❌ End date must be after start date")
        return False
    if start.year < 2000:
        st.error("❌ Start date must be after 2000")
        return False
    return True

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False}

    # --- Animated Header ---
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            show_animation(ANIMATIONS["main"], height=300)
        with col2:
            st.title("FinVision Pro")
            st.markdown("---")
            st.caption("AI-Powered Financial Analysis Platform")

    # --- Sidebar Configuration ---
    with st.sidebar:
        show_animation(ANIMATIONS["chart"], height=150)
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("📈 Stock Symbol:", "AAPL").upper()
            start_date = st.date_input("📅 Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("📅 End Date:")
            if not validate_dates(start_date, end_date):
                return
        else:
            uploaded_file = st.file_uploader("📤 Upload CSV:", type=["csv"])

    # --- Data Loading Section ---
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("📥 Step 1: Load Data")
        
        if st.button("🚀 Load Data", key="load_btn"):
            try:
                with st.spinner("Loading data..."):
                    show_animation(ANIMATIONS["loading"], height=100)
                    
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date).reset_index()
                    else:
                        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                        df = df.set_index('Date')
                    
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
            
            if st.button("✨ Clean Data", key="clean_btn"):
                with st.spinner("Cleaning..."):
                    df = st.session_state.data.dropna()
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    
                    st.success("✅ Cleaning completed!")
                    show_animation(ANIMATIONS["success"], height=100)
                    st.write("Missing values:", df.isna().sum().to_frame().T)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.steps['processed']:
            # Feature Engineering
            with st.container():
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.header("⚙️ Step 3: Feature Engineering")
                
                if st.button("🔧 Generate Features", key="feature_btn"):
                    df = st.session_state.data
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    st.session_state.data = df.dropna()
                    
                    st.success("✅ Features created!")
                    show_animation(ANIMATIONS["success"], height=100)
                    st.dataframe(df.tail())
                st.markdown('</div>', unsafe_allow_html=True)

            # Model Training & Evaluation (Fixed Section)
            with st.container():
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.header("🤖 Step 4: Model Analysis")
                
                if st.button("🎓 Run Analysis", key="model_btn"):
                    df = st.session_state.data
                    
                    # Validate required features
                    required_cols = ['SMA_20', 'SMA_50', 'Close']
                    if not all(col in df.columns for col in required_cols):
                        st.error("❌ Required columns missing. Run Feature Engineering first!")
                        return

                    X = df[['SMA_20', 'SMA_50']]
                    y = df['Close']
                    
                    # Split data with index preservation
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False
                    )
                    
                    # Train model
                    model = LinearRegression().fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Generate and verify predictions
                    y_pred = model.predict(X_test)
                    
                    # Ensure 1D arrays
                    y_pred = np.asarray(y_pred).reshape(-1)
                    y_test_vals = np.asarray(y_test).reshape(-1)
                    
                    # Dimension validation
                    if y_pred.ndim != 1 or y_test_vals.ndim != 1:
                        st.error(f"Dimension Error - Pred: {y_pred.shape} | Test: {y_test_vals.shape}")
                        return

                    # Get aligned dates from original data
                    try:
                        if 'Date' in df.columns:
                            dates = df.loc[X_test.index, 'Date'].values
                        else:
                            dates = X_test.index.to_numpy()
                    except KeyError:
                        dates = X_test.index.to_numpy()

                    # Create comparison dataframe
                    comparison_df = pd.DataFrame({
                        'Date': dates,          # Verified 1D
                        'Actual': y_test_vals,  # Verified 1D
                        'Predicted': y_pred     # Verified 1D
                    }).melt(id_vars='Date', var_name='Type', value_name='Price')

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📉 MSE</h3>
                            <h2>{mean_squared_error(y_test_vals, y_pred):.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 R² Score</h3>
                            <h2>{r2_score(y_test_vals, y_pred):.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # Create interactive plot
                    fig = px.line(
                        comparison_df,
                        x='Date',
                        y='Price',
                        color='Type',
                        color_discrete_sequence=['#4f46e5', '#ef4444'],
                        title="Actual vs Predicted Prices",
                        labels={'Price': 'USD Value'},
                        height=500
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Analysis complete!")
                    show_animation(ANIMATIONS["success"], height=100)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
