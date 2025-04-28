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

# Configure page
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide"
)

# Load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json")
lottie_success = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_au03ianj.json")
lottie_finance = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2znx3l3i.json")

# Custom CSS with animations
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
    
    .main {background-color: #F5F5F5;}
    h1, h2, h3 {color: #003366; animation: fadeIn 1s ease-out;}
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
    .stSuccess {background-color: #DFF2BF; animation: slideInRight 0.5s ease-out;}
    .dataframe {animation: fadeIn 0.8s ease-out; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .plotly-chart {animation: fadeIn 1s ease-out;}
    .stAlert {animation: fadeIn 0.6s ease-out;}
    .section-card {
        padding: 2rem;
        margin: 1rem 0;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Welcome Interface
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(lottie_finance, height=300, key="welcome")
    with col2:
        st.title("Financial Machine Learning Application")
        st.markdown("---")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {'loaded': False, 'processed': False}

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Data Configuration")
        data_source = st.radio("Select Data Source:", ["Yahoo Finance", "Upload Dataset"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("📈 Enter Stock Ticker (e.g., AAPL):", "AAPL")
            start_date = st.date_input("📅 Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("📅 End Date:")
        else:
            uploaded_file = st.file_uploader("📤 Upload CSV File:", type=["csv"])
    
    # Step 1: Load Data
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("📥 Step 1: Load Data")
        if st.button("🚀 Load Data"):
            with st.spinner("Loading data..."):
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
                    st.success("✅ Data loaded successfully!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st_lottie(lottie_success, height=100, key="success1")
                    with col2:
                        st.write("Data Preview:")
                        st.dataframe(st.session_state.data.head().style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'color': '#003366',
                            'border': '1px solid #dee2e6'
                        }))
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Subsequent steps
    if st.session_state.steps['loaded']:
        # Step 2: Preprocessing
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.header("🧹 Step 2: Data Preprocessing")
            if st.button("✨ Clean Data"):
                with st.spinner("Cleaning data..."):
                    df = st.session_state.data
                    
                    # Handle missing values
                    missing = df.isnull().sum()
                    st.write("Missing Values Before Cleaning:")
                    st.dataframe(missing)
                    
                    df = df.dropna()
                    
                    st.write("Missing Values After Cleaning:")
                    st.dataframe(df.isnull().sum())
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success("✅ Data cleaning completed!")
                    st_lottie(lottie_success, height=100, key="success2")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.steps.get('processed'):
            # Step 3: Feature Engineering
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("⚙️ Step 3: Feature Engineering")
                if st.button("🔧 Create Features"):
                    with st.spinner("Creating features..."):
                        df = st.session_state.data
                        
                        # Create features
                        df['SMA_20'] = df['Close'].rolling(window=20).mean()
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        df = df.dropna()
                        
                        st.session_state.data = df
                        st.success("✅ Features created!")
                        st_lottie(lottie_success, height=100, key="success3")
                        st.write("Updated Data:")
                        st.dataframe(df.tail().style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'color': '#003366',
                            'border': '1px solid #dee2e6'
                        }))
                st.markdown('</div>', unsafe_allow_html=True)

            # Step 4: Train/Test Split
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("📊 Step 4: Train/Test Split")
                if st.button("✂️ Split Data"):
                    with st.spinner("Splitting data..."):
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
                        fig = px.pie(split_counts, values=split_counts, names=split_counts.index,
                                    color_discrete_sequence=['#004488', '#4CAF50'])
                        st.plotly_chart(fig)
                        st.success("✅ Data split completed!")
                        st_lottie(lottie_success, height=100, key="success4")
                st.markdown('</div>', unsafe_allow_html=True)

                # Step 5: Model Training
                with st.container():
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.header("🤖 Step 5: Model Training")
                    if st.button("🎓 Train Model"):
                        with st.spinner("Training model..."):
                            model = LinearRegression()
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            st.session_state.model = model
                            st.success("✅ Model training completed!")
                            st_lottie(lottie_success, height=100, key="success5")
                    st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.model:
                    # Step 6: Evaluation
                    with st.container():
                        st.markdown('<div class="section-card">', unsafe_allow_html=True)
                        st.header("📈 Step 6: Model Evaluation")
                        if st.button("📝 Evaluate Model"):
                            with st.spinner("Evaluating model..."):
                                y_pred = st.session_state.model.predict(st.session_state.X_test)
                                
                                # Metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("""
                                    <div style='background: #f8f9fa; padding: 2rem; border-radius: 10px;'>
                                        <h3 style='color: #004488;'>Model Metrics</h3>
                                        <h4 style='color: #4CAF50;'>MSE: {mse:.2f}</h4>
                                        <h4 style='color: #4CAF50;'>R²: {r2:.2f}</h4>
                                    </div>
                                    """.format(mse=mean_squared_error(st.session_state.y_test, y_pred),
                                    unsafe_allow_html=True)
                                
                                # Plot
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
                                st_lottie(lottie_success, height=100, key="success6")
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
