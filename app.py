# app.py - Complete Financial ML Platform with Dynamic Data Handling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="FinML Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2a4a7c; border-bottom: 2px solid #2a4a7c;}
    h2 {color: #3b6ea5;}
    .stButton>button {background-color: #2a4a7c; color: white; border-radius: 5px;}
    .stDownloadButton>button {background-color: #4CAF50; color: white;}
    .stAlert {border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    .csv-guide {border-left: 3px solid #2a4a7c; padding-left: 15px;}
</style>
""", unsafe_allow_html=True)

def compute_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def main():
    st.title("📈 FinML Pro - Financial Machine Learning Platform")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None,
        'steps': {'loaded': False, 'processed': False, 
                 'features_created': False, 'split': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("""
        **Dataset Source:**  
        [Kaggle Finance Data](https://www.kaggle.com/datasets/nitindatta/finance-data)
        """)
        
        uploaded_file = st.file_uploader("Upload Dataset:", type=["csv"])
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reload App", on_click=lambda: st.session_state.clear())

    # Step 1: Data Acquisition with Auto-Cleaning
    st.header("1. Data Acquisition")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate dataset structure
            required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                st.error(f"❌ Missing columns: {', '.join(missing)}")
                return
                
            # Clean numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(
                pd.to_numeric, errors='coerce'
            )
            df = df.dropna(subset=numeric_cols)
            
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.data = df.sort_values('Date')
            st.session_state.steps['loaded'] = True
            st.success("✅ CSV loaded successfully!")
            
            if not df.empty:
                st.dataframe(
                    df.head().style.format("{:.2f}", subset=numeric_cols),
                    height=250
                )
            
        except Exception as e:
            st.error(f"CSV Error: {str(e)}")
    else:
        st.markdown("""
        <div class='csv-guide'>
        📁 **How to Use:**
        1. Download dataset from <a href="https://www.kaggle.com/datasets/nitindatta/finance-data" target="_blank">Kaggle</a>
        2. Upload CSV file using the sidebar uploader
        3. Ensure columns include: Date, Open, High, Low, Close, Volume
        </div>
        """, unsafe_allow_html=True)

    # Step 2: Data Preprocessing
    if st.session_state.steps['loaded']:
        st.header("2. Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Data"):
                try:
                    df = st.session_state.data.copy()
                    
                    st.write("### Data Quality Report:")
                    missing = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Values': df.isnull().sum(),
                        'Data Type': df.dtypes
                    })
                    
                    fig = px.bar(missing, x='Missing Values', y='Column',
                                orientation='h', color='Data Type',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.spinner("Finalizing dataset..."):
                        df = df.dropna().reset_index(drop=True)
                    
                    st.session_state.data = df
                    st.session_state.steps['processed'] = True
                    st.success(f"✅ Final dataset: {len(df)} rows")

                except Exception as e:
                    st.error(f"🚨 Cleaning failed: {str(e)}")

        with col2:
            if st.session_state.steps['processed']:
                try:
                    clean_df = st.session_state.data
                    st.write("### Dataset Overview:")
                    stats = clean_df.describe().T
                    stats['skew'] = clean_df.skew(numeric_only=True)
                    stats['kurtosis'] = clean_df.kurtosis(numeric_only=True)
                    
                    st.dataframe(
                        stats.style.format("{:.2f}")
                        .background_gradient(cmap='Blues'),
                        height=400
                    )
                    
                except Exception as e:
                    st.error(f"📊 Stats error: {str(e)}")

    # Step 3: Adaptive Feature Engineering
    if st.session_state.steps['processed']:
        st.header("3. Feature Engineering")
        
        if st.button("⚡ Create Features"):
            try:
                df = st.session_state.data.copy()
                
                with st.spinner("Building financial features..."):
                    # Rolling features with dynamic window handling
                    df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
                    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
                    df['RSI'] = compute_rsi(df['Close'])
                    
                    # Auto-handle remaining NaNs
                    df = df.dropna().reset_index(drop=True)
                        
                    st.session_state.data = df
                    st.session_state.steps['features_created'] = True
                    
                    st.write("### Feature Relationships:")
                    fig = px.scatter_matrix(df[['Close', 'SMA_20', 'SMA_50', 'RSI']],
                                           dimensions=['Close', 'SMA_20', 'SMA_50', 'RSI'],
                                           color='Close', height=800)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"🚨 Feature creation error: {str(e)}")

    # Step 4: Intelligent Data Split
    if st.session_state.steps['features_created']:
        st.header("4. Data Split")
        
        if st.button("✂️ Split Dataset"):
            try:
                df = st.session_state.data.copy()
                features = ['SMA_20', 'SMA_50', 'RSI']
                
                X = df[features]
                y = df['Close'].values  
                
                with st.spinner("Preprocessing data..."):
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Time-based split
                    split_idx = int(len(X_scaled) * (1 - test_size))
                    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    st.session_state.update({
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'scaler': scaler
                    })
                    st.session_state.steps['split'] = True
                
                st.write("### Data Partition Visualization:")
                split_df = pd.DataFrame({
                    'Type': ['Training', 'Testing'],
                    'Samples': [len(X_train), len(X_test)],
                    'Time Period': [
                        f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[split_idx-1].date()}",
                        f"{df['Date'].iloc[split_idx].date()} to {df['Date'].iloc[-1].date()}"
                    ]
                })
                
                fig = px.bar(split_df, x='Type', y='Samples', text='Time Period',
                            color='Type', color_discrete_sequence=['#2a4a7c', '#3b6ea5'])
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"🚨 Splitting error: {str(e)}")

    # Step 5: Model Training
    if st.session_state.steps.get('split'):
        st.header("5. Model Training")
        
        if st.button("🎯 Train Model"):
            try:
                model = LinearRegression() if model_type == "Linear Regression" \
                    else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                
                with st.spinner(f"Training {model_type}..."):
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = model
                    st.session_state.steps['trained'] = True
                
                st.success(f"✅ {model_type} trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"🚨 Training error: {str(e)}")

    # Step 6: Comprehensive Model Evaluation
    if st.session_state.steps.get('trained'):
        st.header("6. Model Evaluation")
        
        if st.button("📊 Evaluate Performance"):
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                y_pred = model.predict(X_test)
                
                st.write("### Performance Metrics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                with col2:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                with col3:
                    st.metric("Error Range", 
                            f"±{np.abs(y_test - y_pred).mean():.2f}",
                            help="Average absolute prediction error")
                
                st.write("### Prediction Analysis:")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                       marker=dict(color='#2a4a7c', size=8),
                                       name='Predictions'))
                fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], 
                                       y=[min(y_test), max(y_test)],
                                       mode='lines', 
                                       line=dict(color='#4CAF50', dash='dash'),
                                       name='Perfect Fit'))
                fig.update_layout(
                    title="Actual vs Predicted Values",
                    xaxis_title="Actual Prices",
                    yaxis_title="Predicted Prices",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if model_type == "Random Forest":
                    st.write("### Feature Importance Analysis:")
                    importance = pd.DataFrame({
                        'Feature': ['SMA_20', 'SMA_50', 'RSI'],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance, x='Importance', y='Feature',
                                orientation='h', color='Importance',
                                color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

                # Prediction download
                results = pd.DataFrame({
                    'Date': st.session_state.data['Date'].iloc[-len(y_test):],
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Full Predictions", 
                    csv, 
                    "predictions.csv", 
                    "text/csv"
                )

            except Exception as e:
                st.error(f"🚨 Evaluation error: {str(e)}")

if __name__ == "__main__":
    main()
