# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide"
)

# Enhanced Professional CSS with Animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .main {background-color: #F5F5F5;}
    h1 {color: #003366; animation: fadeIn 1s ease-out;}
    .stButton>button {
        background-color: #004488; 
        color: white; 
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSuccess {
        background-color: #DFF2BF; 
        animation: slideInRight 0.5s ease-out;
        border-left: 4px solid #4CAF50;
    }
    .dataframe {
        animation: fadeIn 0.8s ease-out;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    .plotly-chart {
        animation: fadeIn 1s ease-out;
        border-radius: 12px;
        overflow: hidden;
        background: white;
        padding: 15px;
    }
    [data-testid="stImage"] {
        animation: pulse 2s ease-in-out infinite;
        border-radius: 12px;
        overflow: hidden;
    }
    .stAlert {
        animation: fadeIn 0.6s ease-out;
    }
    .section-header {
        animation: fadeIn 0.8s ease-out;
        border-left: 4px solid #004488;
        padding-left: 1rem;
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    .metric-card h3 {
        color: #004488;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Welcome Interface with Enhanced Animations
    st.title("Financial Machine Learning Application")
    st.markdown("---")
    
    # Animated Finance GIF
    st.image("https://media.giphy.com/media/3ohhwgr4HoUu0k3buw/giphy.gif", width=300)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False}

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.header("Data Configuration")
        data_source = st.radio("Select Data Source:", ["Yahoo Finance", "Upload Dataset"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
            start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date:")
        else:
            uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 1: Load Data
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.header("Step 1: Load Data")
    if st.button("Load Data"):
        try:
            if data_source == "Yahoo Finance":
                df = yf.download(ticker, start=start_date, end=end_date)
                df = df.reset_index()
                st.session_state.data = df
            else:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
            
            st.session_state.steps['loaded'] = True
            st.success("Data loaded successfully!")
            st.write("Data Preview:")
            st.dataframe(st.session_state.data.head().style.set_properties(**{
                'background-color': '#f8f9fa',
                'color': '#003366',
                'border': '1px solid #dee2e6'
            }))
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Only show subsequent steps if data is loaded
    if st.session_state.steps['loaded']:
        # Step 2: Preprocessing
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.header("Step 2: Data Preprocessing")
        if st.button("Clean Data"):
            df = st.session_state.data
            
            # Handle missing values
            missing = df.isnull().sum()
            st.write("Missing Values Before Cleaning:")
            st.write(missing)
            
            df = df.dropna()
            
            st.write("Missing Values After Cleaning:")
            st.write(df.isnull().sum())
            
            st.session_state.data = df
            st.session_state.steps['processed'] = True
            st.success("Data cleaning completed!")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.steps.get('processed'):
            # Step 3: Feature Engineering
            st.markdown('<div class="section-header">', unsafe_allow_html=True)
            st.header("Step 3: Feature Engineering")
            if st.button("Create Features"):
                df = st.session_state.data
                
                # Create features for stock data
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df = df.dropna()
                
                st.session_state.data = df
                st.success("Features created!")
                st.write("Updated Data:")
                st.dataframe(df.tail().style.set_properties(**{
                    'background-color': '#f8f9fa',
                    'color': '#003366',
                    'border': '1px solid #dee2e6'
                }))
            st.markdown('</div>', unsafe_allow_html=True)

            # Step 4: Train/Test Split
            st.markdown('<div class="section-header">', unsafe_allow_html=True)
            st.header("Step 4: Train/Test Split")
            if st.button("Split Data"):
                df = st.session_state.data
                X = df[['SMA_20', 'SMA_50']]
                y = df['Close']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False)
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Visualize split
                split_counts = pd.Series({
                    'Training': len(X_train),
                    'Testing': len(X_test)
                })
                fig = px.pie(split_counts, values=split_counts, names=split_counts.index)
                st.plotly_chart(fig)
                st.success("Data split completed!")
            st.markdown('</div>', unsafe_allow_html=True)

                # Step 5: Model Training
                st.markdown('<div class="section-header">', unsafe_allow_html=True)

                st.header("Step 5: Model Training")
                if st.button("Train Model"):
                    model = LinearRegression()
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = model
                    st.success("Model training completed!")
                st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.model:
                    # Step 6: Evaluation
                    st.markdown('<div class="section-header">', unsafe_allow_html=True)
                    st.header("Step 6: Model Evaluation")
                    if st.button("Evaluate Model"):
                        y_pred = st.session_state.model.predict(st.session_state.X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        r2 = r2_score(st.session_state.y_test, y_pred)
                        
                        # Display metrics in animated cards
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="metric-card">'
                                       '<h3>Mean Squared Error</h3>'
                                       f'<h2 style="color:#004488;">{mse:.2f}</h2>'
                                       '</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">'
                                       '<h3>R² Score</h3>'
                                       f'<h2 style="color:#004488;">{r2:.2f}</h2>'
                                       '</div>', unsafe_allow_html=True)
                        
                        # Plot predictions with enhanced style
                        fig = px.line(
                            title="Actual vs Predicted Prices",
                            x=st.session_state.X_test.index,
                            y=[st.session_state.y_test, y_pred],
                            labels={'value': 'Price', 'variable': 'Legend'},
                            color_discrete_sequence=['#004488', '#4CAF50']
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Date",
                            yaxis_title="Price",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
