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
    page_title="ML Pro",
    page_icon="📊",
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
    .data-warning {color: #d35400; font-weight: bold;}
    .feature-selector {background-color: #e8f4f8; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

def main():
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

    # Step 1: Data Acquisition
    st.header("1. Data Upload & Selection")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Auto-detect numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Dataset needs at least 2 numeric columns for analysis")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Show data summary
            st.write("### Dataset Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            # Feature and target selection
            with st.expander("🔍 Select Features & Target"):
                st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
                all_cols = df.columns.tolist()
                
                # Target selection
                target = st.selectbox("Select Target Variable:", 
                                    numeric_cols,
                                    index=len(numeric_cols)-1 if len(numeric_cols) > 0 else 0)
                
                # Feature selection (multi-select)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("Select Features:", 
                                         numeric_cols,
                                         default=default_features)
                
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

    # Step 2: Data Preprocessing
    if st.session_state.steps['processed']:
        st.header("2. Data Analysis")
        
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Feature-Target Relationships")
            selected_feature = st.selectbox("Select feature to plot:", features)
            fig = px.scatter(df, x=selected_feature, y=target, 
                            trendline="ols", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Correlation Matrix")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", 
                           color_continuous_scale='Blues', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Proceed to Model Training"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Model Training")
        
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        with st.spinner(f"Training {model_type}..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }
            
            st.success("Model trained successfully!")
            st.balloons()

    # Step 4: Model Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Model Evaluation")
        
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        with col2:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        
        # Actual vs Predicted plot
        st.write("### Actual vs Predicted Values")
        results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Actual'],
            name='Actual',
            mode='markers',
            marker=dict(color='#2a4a7c')
        ))
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Predicted'],
            name='Predicted',
            mode='markers',
            marker=dict(color='#4CAF50')
        ))
        fig.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="Value",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance for Random Forest
        if model_type == "Random Forest":
            st.write("### Feature Importance")
            importance = pd.DataFrame({
                'Feature': st.session_state.features,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance, x='Importance', y='Feature',
                         orientation='h', color='Importance',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Predictions", 
            csv, 
            "predictions.csv", 
            "text/csv"
        )

if __name__ == "__main__":
    main()
