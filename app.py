# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_lottie import st_lottie

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Financial ML App", page_icon="📈", layout="wide")

# ----------------- ANIMATION LOADER -----------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_finance = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2znx3l3i.json")
lottie_success = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_au03ianj.json")

# ----------------- CSS -----------------
st.markdown("""
<style>
    @keyframes fadeIn { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
    .main {background-color: #F5F5F5;}
    h1, h2, h3 {color: #003366; animation: fadeIn 1s ease-out;}
    .section-card {
        padding: 2rem; margin: 1rem 0;
        border-radius: 15px; background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    .stButton>button {
        background-color: #004488; color: white;
        transition: 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION INIT -----------------
for key in ['data', 'model', 'X_train', 'X_test', 'y_train', 'y_test', 'steps']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state['steps'] is None:
    st.session_state['steps'] = {'loaded': False, 'processed': False}

# ----------------- HEADER -----------------
col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(lottie_finance, height=300, key="finance")
with col2:
    st.title("Financial Machine Learning App")
    st.markdown("---")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Data Configuration")
data_source = st.sidebar.radio("Select Data Source:", ["Yahoo Finance", "Upload Dataset"])

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("📈 Stock Ticker:", "AAPL")
    start_date = st.sidebar.date_input("📅 Start Date:", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("📅 End Date:")
else:
    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV:", type=["csv"])

# ----------------- LOAD DATA -----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("📥 Step 1: Load Data")

if st.button("🚀 Load Data"):
    with st.spinner("Loading..."):
        try:
            if data_source == "Yahoo Finance":
                df = yf.download(ticker, start=start_date, end=end_date).reset_index()
            else:
                df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success("✅ Data Loaded Successfully!")
            st_lottie(lottie_success, height=100, key="load_success")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------- DATA PREPROCESSING -----------------
if st.session_state.steps['loaded']:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🧹 Step 2: Data Preprocessing")

    if st.button("✨ Clean Data"):
        df = st.session_state.data
        st.write("Missing values before cleaning:")
        st.dataframe(df.isnull().sum())
        df = df.dropna()
        st.write("Missing values after cleaning:")
        st.dataframe(df.isnull().sum())
        st.session_state.data = df
        st.session_state.steps['processed'] = True
        st.success("✅ Data Cleaned!")
        st_lottie(lottie_success, height=100, key="clean_success")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FEATURE ENGINEERING -----------------
if st.session_state.steps.get('processed'):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("⚙️ Step 3: Feature Engineering")

    if st.button("🔧 Create Features"):
        df = st.session_state.data
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df.dropna(inplace=True)
        st.session_state.data = df
        st.success("✅ Features Created!")
        st_lottie(lottie_success, height=100, key="feature_success")
        st.dataframe(df.tail())

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TRAIN/TEST SPLIT -----------------
if st.session_state.steps.get('processed'):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📊 Step 4: Train/Test Split")

    if st.button("✂️ Split Data"):
        df = st.session_state.data
        X = df[['SMA_20', 'SMA_50']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        fig = px.pie(values=[len(X_train), len(X_test)], names=['Training', 'Testing'],
                     color_discrete_sequence=['#004488', '#4CAF50'])
        st.plotly_chart(fig)
        st.success("✅ Data Split Done!")
        st_lottie(lottie_success, height=100, key="split_success")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- MODEL TRAINING -----------------
if st.session_state.X_train is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🤖 Step 5: Model Training")

    if st.button("🎓 Train Model"):
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.success("✅ Model Trained Successfully!")
        st_lottie(lottie_success, height=100, key="train_success")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- MODEL EVALUATION -----------------
if st.session_state.model is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📈 Step 6: Model Evaluation")

    if st.button("📝 Evaluate Model"):
        y_pred = st.session_state.model.predict(st.session_state.X_test)

        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)

        st.subheader("📋 Metrics")
        st.metric(label="MSE", value=f"{mse:.2f}")
        st.metric(label="R²", value=f"{r2:.2f}")

        fig = px.line()
        fig.add_scatter(x=st.session_state.X_test.index, y=st.session_state.y_test, mode='lines', name="Actual")
        fig.add_scatter(x=st.session_state.X_test.index, y=y_pred, mode='lines', name="Predicted")
        fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Index", yaxis_title="Price")
        st.plotly_chart(fig)

        st.success("✅ Evaluation Complete!")
        st_lottie(lottie_success, height=100, key="evaluate_success")

    st.markdown('</div>', unsafe_allow_html=True)
