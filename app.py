# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from streamlit_lottie import st_lottie
import json
import requests
from io import StringIO

# Configure page
st.set_page_config(
    page_title="FinML Analyst",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_raiw2hpe.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_auwiessx.json")

# Custom CSS
st.markdown("""
<style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 1s ease-out;
    }
    
    .main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
    h1 {color: #2c3e50; border-bottom: 3px solid #3498db;}
    .stButton>button {
        background: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #2980b9;
        transform: scale(1.05);
    }
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

def main():
    # Welcome Interface
    st.title("📈 FinTech Machine Learning Analyst")
    st.markdown("---")
    
    # Animated header
    with st.container():
        if lottie_loading:
            st_lottie(lottie_loading, speed=1, height=200, key="welcome")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {
            'loaded': False,
            'processed': False,
            'features_created': False,
            'split': False,
            'trained': False
        }

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Yahoo Finance", "Upload CSV"])
        
        model_type = st.selectbox(
            "Select Model:",
            ["Linear Regression", "Random Forest", "K-Means Clustering"],
            help="Choose your machine learning model"
        )
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Ticker:", "AAPL")
            start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
            end_date = st.date_input("End Date:")
        else:
            uploaded_file = st.file_uploader("Upload Dataset:", type=["csv"])
        
        st.markdown("---")
        if st.button("🔄 Reset All Steps"):
            st.session_state.steps = {k: False for k in st.session_state.steps}
            st.experimental_rerun()

    # Step 1: Load Data
    with st.container():
        st.header("📂 Step 1: Load Data", anchor="step1")
        if st.button("🚀 Load Data", key="load"):
            try:
                with st.spinner('Fetching data...'):
                    if data_source == "Yahoo Finance":
                        df = yf.download(ticker, start=start_date, end=end_date)
                        df = df.reset_index()
                        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
                    else:
                        if uploaded_file:
                            df = pd.read_csv(uploaded_file)
                    
                    st.session_state.data = df
                    st.session_state.steps['loaded'] = True
                    
                    # Animated success
                    st_lottie(lottie_success, speed=1, height=150, key="success1")
                    st.balloons()
                    
                    st.write("### Data Preview")
                    st.dataframe(df.head().style.highlight_max(axis=0, color="#3498db"))

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")

    # Only show subsequent steps if data is loaded
    if st.session_state.steps['loaded']:
        # Step 2: Preprocessing
        with st.container():
            st.header("🧹 Step 2: Data Preprocessing", anchor="step2")
            if st.button("✨ Clean Data", key="clean"):
                df = st.session_state.data.copy()
                
                # Animated preprocessing
                with st.spinner('Cleaning data...'):
                    progress = st.progress(0)
                    
                    # Handle missing values
                    missing = df.isnull().sum()
                    st.write("### Missing Values Report")
                    fig = px.bar(missing, title="Missing Values per Column")
                    st.plotly_chart(fig)
                    
                    df = df.dropna()
                    progress.progress(50)
                    
                    # Remove duplicates
                    df = df.drop_duplicates()
                    progress.progress(100)
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    
                    # Success animation
                    st_lottie(lottie_success, speed=1, height=150, key="success2")
                    st.success("Data cleaning completed!")

        if st.session_state.steps.get('processed'):
            # Step 3: Feature Engineering
            with st.container():
                st.header("⚡ Step 3: Feature Engineering", anchor="step3")
                if st.button("🔧 Create Features", key="features"):
                    df = st.session_state.data.copy()
                    
                    # Create technical indicators
                    with st.spinner('Engineering features...'):
                        df['SMA_20'] = df['Close'].rolling(window=20).mean()
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        df['Returns'] = df['Close'].pct_change()
                        df = df.dropna()
                        
                        st.session_state.data = df
                        st.session_state.steps['features_created'] = True
                        
                        st.write("### Feature Correlation Matrix")
                        corr = df.corr()
                        fig = px.imshow(corr, color_continuous_scale='Blues')
                        st.plotly_chart(fig)

            if st.session_state.steps.get('features_created'):
                # Step 4: Train/Test Split
                with st.container():
                    st.header("✂️ Step 4: Train/Test Split", anchor="step4")
                    if st.button("🔀 Split Data", key="split"):
                        df = st.session_state.data
                        
                        if model_type != "K-Means Clustering":
                            X = df[['SMA_20', 'SMA_50', 'Returns']]
                            y = df['Close']
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, shuffle=False)
                            
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                        else:
                            X = df[['SMA_20', 'SMA_50', 'Returns']]
                            st.session_state.X = X
                        
                        # Visualize split
                        st.write("### Data Split Visualization")
                        split_counts = pd.Series({
                            'Training': len(X_train) if model_type != "K-Means" else len(X),
                            'Testing': len(X_test) if model_type != "K-Means" else 0
                        })
                        fig = px.pie(split_counts, values=split_counts, 
                                   names=split_counts.index,
                                   color=split_counts.index,
                                   color_discrete_sequence=['#3498db', '#e74c3c'])
                        st.plotly_chart(fig)
                        
                        st.session_state.steps['split'] = True

                if st.session_state.steps.get('split'):
                    # Step 5: Model Training
                    with st.container():
                        st.header("🤖 Step 5: Model Training", anchor="step5")
                        if st.button("🎓 Train Model", key="train"):
                            with st.spinner('Training in progress...'):
                                progress = st.progress(0)
                                
                                if model_type == "Linear Regression":
                                    model = LinearRegression()
                                elif model_type == "Random Forest":
                                    model = RandomForestRegressor()
                                else:
                                    model = KMeans(n_clusters=3)
                                
                                if model_type != "K-Means Clustering":
                                    model.fit(st.session_state.X_train, st.session_state.y_train)
                                else:
                                    model.fit(st.session_state.X)
                                
                                st.session_state.model = model
                                progress.progress(100)
                                
                                st_lottie(lottie_success, speed=1, height=150, key="success3")
                                st.balloons()
                                st.success(f"{model_type} training completed!")

                    if st.session_state.model:
                        # Step 6: Evaluation
                        with st.container():
                            st.header("📊 Step 6: Model Evaluation", anchor="step6")
                            if st.button("🧪 Evaluate Model", key="eval"):
                                if model_type != "K-Means Clustering":
                                    y_pred = st.session_state.model.predict(st.session_state.X_test)
                                    
                                    # Metrics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)):.2f}")
                                    with col2:
                                        st.metric("R² Score", f"{r2_score(st.session_state.y_test, y_pred):.2f}")
                                    
                                    # Prediction plot
                                    fig = px.line(
                                        x=st.session_state.X_test.index,
                                        y=[st.session_state.y_test, y_pred],
                                        labels={'value': 'Price', 'variable': 'Legend'},
                                        color_discrete_sequence=['#3498db', '#e74c3c']
                                    )
                                    fig.update_layout(title="Actual vs Predicted Prices")
                                    st.plotly_chart(fig)
                                else:
                                    labels = st.session_state.model.labels_
                                    st.metric("Silhouette Score", f"{silhouette_score(st.session_state.X, labels):.2f}")
                                    
                                    fig = px.scatter_3d(
                                        st.session_state.X,
                                        x='SMA_20',
                                        y='SMA_50',
                                        z='Returns',
                                        color=labels.astype(str),
                                        color_discrete_sequence=px.colors.qualitative.Set1
                                    )
                                    st.plotly_chart(fig)

                                # Download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name='financial_analysis.csv',
                                    mime='text/csv'
                                )

if __name__ == "__main__":
    main()
