import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set Streamlit layout
st.set_page_config(layout="wide")
st.title("📈 Stock Market Analysis Dashboard")

# Function to print section headers
def print_heading(text):
    st.markdown(f"### 📌 {text}")

# Step 1: Select Stocks and Date Range
print_heading("Step 1: Select Stocks and Date Range")

stocks = st.multiselect("Select stock tickers:", ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"], default=["AAPL", "GOOGL", "TSLA", "AMZN"])
start_date = st.date_input("Start date", pd.to_datetime("2023-10-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-04-01"))

# Step 2: Fetch Stock Market Data
print_heading("Step 2: Fetching Stock Market Data")

if stocks:
    try:
        stock_data = yf.download(stocks, start=start_date, end=end_date)

        if stock_data.empty:
            st.error("🚨 No stock data retrieved! Please check tickers and date range.")
            st.stop()

        st.subheader("Raw Stock Data")
        st.dataframe(stock_data.head())
        
        # Step 3: Extract Close Prices
        print_heading("Step 3: Extracting Close Prices")
        if isinstance(stock_data.columns, pd.MultiIndex):
            adj_close = stock_data.xs("Close", axis=1, level=0)
        else:
            adj_close = stock_data.get("Close", pd.DataFrame())

        if adj_close.empty:
            st.error("🚨 No 'Close' price data found!")
            st.stop()

        st.dataframe(adj_close.head())

        # Step 4: Line Plot of Stock Prices
        print_heading("Step 4: Stock Price Trends Over Time")
        plt.figure(figsize=(12,6))
        for stock in stocks:
            sns.lineplot(data=adj_close, x=adj_close.index, y=stock, label=stock)
        plt.title("Stock Price Trends")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

        # Step 5: Histogram of Prices
        print_heading("Step 5: Stock Price Volatility")
        plt.figure(figsize=(10,6))
        for stock in stocks:
            sns.histplot(adj_close[stock], kde=True, label=stock, alpha=0.5)
        plt.title("Stock Price Distribution & Volatility")
        plt.xlabel("Price (USD)")
        plt.legend()
        st.pyplot(plt.gcf())

        # Step 6: Correlation Matrix
        print_heading("Step 6: Stock Correlation Matrix")
        corr_matrix = adj_close.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        st.pyplot(plt.gcf())

        # Step 7: Daily Returns
        print_heading("Step 7: Daily Returns")
        daily_returns = adj_close.pct_change()
        st.subheader("Daily Returns Data")
        st.dataframe(daily_returns.head())

        plt.figure(figsize=(12,6))
        for stock in stocks:
            sns.lineplot(data=daily_returns, x=daily_returns.index, y=stock, label=stock)
        plt.title("Daily Stock Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

        # Step 8: Financial Dashboard
        print_heading("Step 8: Financial Dashboard")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.lineplot(data=adj_close, x=adj_close.index, y=stocks[0], ax=axes[0, 0])
        axes[0, 0].set_title(f"{stocks[0]} Price Trend")

        sns.histplot(adj_close[stocks[1]], kde=True, color="purple", ax=axes[0, 1])
        axes[0, 1].set_title(f"{stocks[1]} Price Distribution")

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=axes[1, 0])
        axes[1, 0].set_title("Correlation Matrix")

        sns.lineplot(data=daily_returns, x=daily_returns.index, y=stocks[2], ax=axes[1, 1])
        axes[1, 1].set_title(f"{stocks[2]} Daily Returns")

        plt.tight_layout()
        st.pyplot(fig)

        st.success("✅ Analysis Complete!")

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
else:
    st.warning("⚠️ Please select at least one stock ticker.")
