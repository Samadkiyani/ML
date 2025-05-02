# app.py - Universal ML Platform for Any Dataset
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
    page_title="ML PRO MADE BY SAMAD KIANI",
    page_icon="https://tse2.mm.bing.net/th?id=OIP.Fkdoyke5qijSDVWyGKJB9QHaHk&pid=Api&P=0&h=220",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Background image & styling
st.markdown("""
<style>
    body, .stApp {
        background-image: url('https://static.vecteezy.com/system/resources/previews/022/144/287/large_2x/growth-strategy-business-graph-analysis-concept-on-finance-chart-data-diagram-3d-background-wallpaper-ai-generated-photo.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #ffffff;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
        color: #333333;
    }
    h1, h2 {
        color: #1c3d5a;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #1c3d5a;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }
    .data-warning {
        color: #c0392b;
        font-weight: bold;
    }
    .feature-selector {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        color: #333;
    }
    .st-expanderContent {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("📊 Universal ML Analysis Platform")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header("1. Data Upload & Selection")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("Dataset needs at least 2 numeric columns for analysis")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            st.write("### Dataset Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🔍 Select Features & Target"):
                st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
                all_cols = df.columns.tolist()
                target = st.selectbox("Select Target Variable:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("Select Features:", numeric_cols, default=default_features)
                
                if st.button("Confirm Selection"):
                    if len(features) < 1:
                        st.error("Please select at least one feature")
                    elif target in features:
                        st.error("Target variable cannot be a feature")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        st.success("Features and target confirmed!")
                st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.markdown("""
        <div class='feature-selector'>
        📁 **How to Use:**
        1. Upload any CSV or Excel file with numeric data  
        2. Select target variable (what you want to predict)  
        3. Choose features (variables used for prediction)  
        4. The system will automatically handle the rest  
        </div>
        """, unsafe_allow_html=True)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.header("2. Data Analysis")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Feature-Target Relationships")
            selected_feature = st.selectbox("Select feature to plot:", features)
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Correlation Matrix")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Blues', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Proceed to Model Training"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Model Training")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(f"Training {model_type}..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            st.success("Model trained successfully!")
            st.balloons()

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Model Evaluation")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        with col2:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        
        st.write("### Actual vs Predicted Values")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', mode='markers', marker=dict(color='#2a4a7c')))
        fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', mode='markers', marker=dict(color='#4CAF50')))
        fig.update_layout(xaxis_title="Sample Index", yaxis_title="Value", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write("### Feature Importance")
            importance = pd.DataFrame({'Feature': st.session_state.features, 'Importance': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
