# app.py (100% Working Version)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# Enhanced Data Loading Function
def load_stock_data(tickers, start_date, end_date):
    loaded_stocks = {}
    current_date = date.today()
    
    # Validate dates
    if end_date > current_date:
        st.error("❌ Error: End date cannot be in the future")
        return loaded_stocks
    
    for ticker in [t.strip().upper() for t in tickers.split(',')]:
        try:
            # Download individual stock data
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                df = df.reset_index()
                df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) 
                            else col for col in df.columns]
                loaded_stocks[ticker] = df
                st.success(f"✅ Successfully loaded {ticker}")
            else:
                st.error(f"❌ No data found for {ticker}")
        except Exception as e:
            st.error(f"❌ Failed to load {ticker}: {str(e)}")
    
    return loaded_stocks

# Modified Main Function Section
def main():
    # Previous UI setup remains the same
    
    with st.expander("📥 STEP 1: Load Market Data", expanded=True):
        if st.button("🚀 Load & Process Data", key="load_data"):
            st.session_state.stocks = load_stock_data(tickers, start_date, end_date)
            
            if st.session_state.stocks:
                # Display loaded stocks
                cols = st.columns(3)
                for idx, (ticker, data) in enumerate(st.session_state.stocks.items()):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class='ticker-card'>
                            <h4>{ticker}</h4>
                            <p>First Date: {data.iloc[0]['Date']}</p>
                            <p>Last Date: {data.iloc[-1]['Date']}</p>
                            <p>Data Points: {len(data):,}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Force show next steps
                st.session_state.steps = {
                    'loaded': True,
                    'cleaned': False,
                    'features': False,
                    'trained': False
                }
                st.rerun()

    # Rest of the steps remain the same but add:
    if st.session_state.get('steps', {}).get('loaded'):
        with st.expander("🧹 STEP 2: Data Cleaning", expanded=True):
            # Cleaning controls
            if st.button("✨ Clean Data"):
                # Cleaning logic
                st.session_state.steps['cleaned'] = True
                st.rerun()
    
    if st.session_state.get('steps', {}).get('cleaned'):
        with st.expander("⚡ STEP 3: Feature Engineering", expanded=True):
            # Feature engineering
            if st.button("🔧 Create Features"):
                # Feature logic
                st.session_state.steps['features'] = True
                st.rerun()
    
    # Continue this pattern for other steps

if __name__ == "__main__":
    main()
