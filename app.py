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
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Configurations
THEMES = {
    "🧟 Zombie Apocalypse": {
        "emojis": ["🧟", "🔪", "🩸", "☠️", "🦇"],
        "background": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHB6YTZzdjM3anpzOTdtdXg5NXBpM2VlbDNmNzlvenBmMnpicXJ6eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/13bD1lqmU6fRbq/giphy.gif",
        "primary_color": "#ff0000",
        "secondary_color": "#8b0000",
        "header_color": "#ffffff",
        "text_color": "#000000",
        "sidebar_bg": "rgba(255, 255, 255, 0.9)",
        "sidebar_text": "#000000",
        "font_family": "'Creepster', cursive",
        "chart_theme": "plotly_dark"
    },
    "🚀 Futuristic Voyage": {
        "emojis": ["🚀", "🌌", "🤖", "👽", "💫"],
        "background": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGxrZmJwbjYxNHB1aHhuNGZnb3I2ejhiZ3JrNnd5dnBmdzI0dWRkayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26FPKhUtNG3TW74f6/giphy.gif",
        "primary_color": "#00f3ff",
        "secondary_color": "#0066ff",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(0, 0, 0, 0.8)",
        "sidebar_text": "#ffffff",
        "font_family": "'Orbitron', sans-serif",
        "chart_theme": "plotly_white"
    },
    "🏰 Game of Kingdoms": {
        "emojis": ["🏰", "⚔️", "🛡️", "👑", "🐉"],
        "background": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExenptbXcwMWc5NWlxNDE0c2cyODNkZHQwcTNkYWJnNmVla21kbGY2biZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kJ3KXdQzBr69y/giphy.gif",
        "primary_color": "#ffcc00",
        "secondary_color": "#ff6600",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(255, 215, 0, 0.8)",
        "sidebar_text": "#000000",
        "font_family": "'Cinzel', serif",
        "chart_theme": "plotly"
    },
    "🎯 Pubg Battlefield": {
        "emojis": ["🎯", "🔫", "🕹️", "🎖️", "⚔️"],
        "background": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmp4cm53dm5odGFpMXV6ZnZmbjFsaWZlYWtzbnI4ZGowYTR1a3p6ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vVwv7I87uB9gZ7avim/giphy.gif",
        "primary_color": "#39ff14",
        "secondary_color": "#228B22",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(0, 0, 0, 0.8)",
        "sidebar_text": "#ffffff",
        "font_family": "'Press Start 2P', cursive",
        "chart_theme": "plotly_dark"
    }
}

def apply_theme(theme):
    emoji = theme["emojis"][0]
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        body, .stApp {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-attachment: fixed;
            color: {theme["text_color"]};
            font-family: {theme["font_family"]};
        }}
        
        .main-container {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 0 20px {theme["primary_color"]};
            margin: 1rem;
        }}
        
        h1, h2, h3 {{
            color: {theme["primary_color"]} !important;
            text-shadow: 2px 2px 4px {theme["secondary_color"]};
        }}
        
        .stButton>button {{
            background: linear-gradient(45deg, {theme["primary_color"]}, {theme["secondary_color"]});
            border: none;
            color: {theme["header_color"]};
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 15px {theme["primary_color"]};
        }}
        
        .sidebar .sidebar-content {{
            background: {theme["sidebar_bg"]};
            color: {theme["sidebar_text"]};
            border-right: 3px solid {theme["primary_color"]};
        }}
        
        .dataframe {{
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px;
            box-shadow: 0 0 10px {theme["primary_color"]};
        }}
        
        .stMetric {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1rem;
            border-left: 5px solid {theme["primary_color"]};
        }}
        
        .stExpander {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            border: 2px solid {theme["primary_color"]};
        }}
        
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-20px); }}
            100% {{ transform: translateY(0px); }}
        }}
        
        .theme-emoji {{
            font-size: 2.5rem;
            animation: float 3s ease-in-out infinite;
            text-shadow: 0 0 10px {theme["primary_color"]};
        }}
    </style>
    """, unsafe_allow_html=True)

# Main Application
def main():
    # Theme Selection
    st.sidebar.header(f"🎨 THEME CUSTOMIZATION")
    theme_name = st.sidebar.selectbox("Select Theme", list(THEMES.keys()))
    theme = THEMES[theme_name]
    apply_theme(theme)
    
    # Dynamic Emoji Display
    emoji = theme["emojis"][0]
    st.sidebar.markdown(f"""
    <div style="text-align:center; margin: 2rem 0;">
        <span class="theme-emoji">{''.join(theme["emojis"][:3])}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content Container
    st.markdown(f'<div class="main-container">', unsafe_allow_html=True)
    
    # Dynamic Header with Theme Emojis
    st.markdown(f"""
    <h1 style="text-align:center;">
        {emoji} Universal ML Platform {emoji}
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Session State Management
    session_defaults = {'data': None, 'model': None, 'features': [], 'target': None, 'steps': {'loaded': False, 'processed': False, 'trained': False}, 'predictions': None}
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header(f"{theme['emojis'][1]} CONFIGURATION")
        uploaded_file = st.file_uploader(f"{theme['emojis'][2]} Upload Dataset:", type=["csv", "xlsx"])
        model_type = st.selectbox(f"{theme['emojis'][3]} Model Type:", ["Linear Regression", "Random Forest"])
        test_size = st.slider(f"{theme['emojis'][4]} Test Size:", 0.1, 0.5, 0.2)
        st.button("🔄 Reset Session", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header(f"{theme['emojis'][1]} Step 1: Data Upload")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error(f"{theme['emojis'][-1]} Dataset needs at least 2 numeric columns!")
                return
            
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"{theme['emojis'][2]} Successfully loaded {len(df)} records")
            
            # Data Preview
            st.write("### Dataset Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander(f"{theme['emojis'][3]} Select Features & Target"):
                all_cols = df.columns.tolist()
                target = st.selectbox("🎯 Select Target:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("📊 Select Features:", numeric_cols, default=default_features)
                
                if st.button(f"{theme['emojis'][1]} Confirm Selection"):
                    if len(features) < 1:
                        st.error(f"{theme['emojis'][4]} Please select at least one feature")
                    elif target in features:
                        st.error(f"{theme['emojis'][0]} Target variable cannot be a feature")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        st.success(f"{theme['emojis'][2]} Features and target confirmed!")
                        st.session_state.steps['step_2_done'] = False
                st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.markdown(f"""
        <div class='feature-selector'>
        📁 **How to Use:**
        1. Upload any CSV or Excel file with numeric data  
        2. Select target variable (what you want to predict)  
        3. Choose features (variables used for prediction)  
        4. The system will automatically handle the rest  
        </div>
        """, unsafe_allow_html=True)

    # Step 2: Data Analysis
    if st.session_state.steps['processed'] and not st.session_state.get('step_2_done'):
        st.header(f"{theme['emojis'][1]} Step 2: Data Analysis")
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
        
        if st.button(f"🚀 Proceed to Model Training"):
            st.session_state.steps['step_2_done'] = True
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header(f"{theme['emojis'][1]} Step 3: Model Training")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
            st.success(f"{theme['emojis'][2]} Model trained successfully!")
            st.balloons()

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header(f"{theme['emojis'][1]} Step 4: Model Evaluation")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        with col2:
            st.metric("📈 R² Score", f"{r2_score(y_test, y_pred):.2f}")
        
        st.write("### Actual vs Predicted Values")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', mode='markers', marker=dict(color='#2a4a7c')))
        fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', mode='markers', marker=dict(color='#4CAF50')))
        fig.update_layout(xaxis_title="Sample Index", yaxis_title="Value", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write(f"{theme['emojis'][4]} Feature Importance")
            importance = pd.DataFrame({'Feature': st.session_state.features, 'Importance': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(f"💾 Download Predictions", csv, "predictions.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
