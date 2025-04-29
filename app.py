import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        .stButton > button {background-color: #007ACC; color: white; border-radius: 5px;}
        .stDataFrame {border: 1px solid #ccc; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)  # use unsafe_allow_html to apply CSS&#8203;:contentReference[oaicite:15]{index=15}

st.title("Stock Price Prediction App")

# Sidebar inputs: data source selection
source = st.sidebar.radio("Select Data Source", ["Yahoo Finance", "CSV Upload"])
if source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-12-31"))
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    ticker = None

# Button to load data
if st.button("Load Data"):
    if source == "Yahoo Finance" and ticker:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        st.success(f"Fetched data for {ticker} from {start_date} to {end_date}")
    elif source == "CSV Upload" and uploaded_file is not None:
        # Read uploaded CSV (streamlit gives a BytesIO)
        data = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        st.success("CSV data loaded successfully")
    else:
        st.error("Please select a valid data source and input.")
        st.stop()

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(data)

    # Clean data: drop missing values
    df = data.dropna()
    st.write("Data after dropping missing values:")
    st.dataframe(df)

    # Feature engineering: compute 20-day and 50-day SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()  # drop initial rows with NaN SMAs
    st.write("Data with SMA_20 and SMA_50:")
    st.dataframe(df[['Close', 'SMA_20', 'SMA_50']].head(50))

    # Split into train/test sets (80/20 split, no shuffle)&#8203;:contentReference[oaicite:16]{index=16}
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    X_train = train_df[['SMA_20', 'SMA_50']]
    y_train = train_df['Close']
    X_test = test_df[['SMA_20', 'SMA_50']]
    y_test = test_df['Close']

    # Train Linear Regression model&#8203;:contentReference[oaicite:17]{index=17}
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set and flatten predictions to 1D array&#8203;:contentReference[oaicite:18]{index=18}
    y_pred = model.predict(X_test).flatten()

    # Prepare DataFrame for plotting actual vs predicted
    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }, index=y_test.index)

    # Plot Actual vs Predicted line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Actual'], name="Actual"))
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Predicted'], name="Predicted"))
    fig.update_layout(title="Actual vs Predicted Close Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Show Lottie animation for success (example)
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json"
    lottie_json = requests.get(lottie_url).json()
    st_lottie(lottie_json, height=200, key="success")  # e.g. a checkmark animation
