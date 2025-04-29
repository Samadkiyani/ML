import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import base64

# --------------------- Helper Functions ---------------------

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
         <style>
         .stApp {{
             background-image: url("data:image/gif;base64,{encoded}");
             background-size: cover;
         }}
         </style>
         """, unsafe_allow_html=True)

def download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# ---------------------- App Layout -----------------------

st.set_page_config(layout="wide")
st.title("📈 Stock Prediction Dashboard")
add_bg_from_local("welcome.gif")  # Place your gif in the same directory

st.sidebar.header("📊 Data Input Options")
data_source = st.sidebar.radio("Select Data Source", ["Yahoo Finance", "Upload CSV"])

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start_date, end=end_date)
        st.success("✅ Data fetched successfully from Yahoo Finance.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully.")

# -------------------- Data Preprocessing --------------------

if 'df' in locals():
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.tail())
    st.info(f"Data contains {df.isnull().sum().sum()} missing values.")
    df.dropna(inplace=True)
    st.success("✅ Missing values dropped.")

    # ------------------ Feature Engineering ------------------
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    st.success("✅ Features Engineered (SMA_5, SMA_20, Target).")

    # ----------------- Feature Selection ---------------------
    st.subheader("🎯 Feature Importance (Linear Regression)")
    X = df[["SMA_5", "SMA_20"]]
    y = df["Target"]
    linreg = LinearRegression().fit(X, y)
    coef_df = pd.DataFrame({'Feature': X.columns, 'Importance': linreg.coef_})
    st.dataframe(coef_df)

    # ----------------- Train/Test Split ---------------------
    st.subheader("🧪 Train/Test Split")
    test_size = st.slider("Test Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Pie chart for visual
    split_fig = go.Figure(data=[go.Pie(labels=["Train", "Test"], values=[len(X_train), len(X_test)])])
    st.plotly_chart(split_fig)

    # ---------------- Model Training & Selection ----------------
    st.subheader("⚙️ Model Training")
    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])
    if st.button("Train Model"):
        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("✅ Model trained successfully.")

        # ---------------- Evaluation -------------------
        st.subheader("📈 Evaluation Metrics")
        st.metric("R² Score", round(r2_score(y_test, y_pred), 4))
        st.metric("MSE", round(mean_squared_error(y_test, y_pred), 4))

        # ---------------- Result Visualization -------------------
        st.subheader("📊 Prediction Chart")
        pred_df = pd.DataFrame({
            "Date": X_test.index,
            "Actual": y_test,
            "Predicted": y_pred
        }).set_index("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predicted"], name="Predicted"))
        st.plotly_chart(fig)

        # ---------------- Download Option -------------------
        st.subheader("⬇️ Download Results")
        st.markdown(download_link(pred_df.reset_index(), "predictions.csv", "📥 Download Predictions CSV"), unsafe_allow_html=True)

        # ---------------- Completion GIF -------------------
        add_bg_from_local("success.gif")
        st.balloons()
