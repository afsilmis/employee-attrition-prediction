import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ========================
# ENHANCED CUSTOM CSS
# ========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        min-height: 100vh;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 3rem;
        color: #000000;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
    }

    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .input-section {
        background: white;
        padding: 0;
        margin: 2rem 0;
        border-radius: 15px;
    }

    .input-group {
        margin-top: 1.5rem;
        transition: all 0.3s ease;
    }

    .input-group h4 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header {
        text-align: center;
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
        position: relative;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }

    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e1e8f7;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
    }

    .stNumberInput > div > div > input {
        background: white;
        border: 2px solid #e1e8f7;
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .prediction-card-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }

    .prediction-card-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .recommendation-card:hover {
        transform: translateX(5px);

    .recommendation-card h4 {
        margin-top: 0;
        font-size: 1.1rem;
        color: #2d3748;
        font-weight: 600;
    }

    .recommendation-card p {
        margin: 0;
        line-height: 1.6;
        font-size: 0.95rem;
        color: #4a5568;
    }

    .main .block-container {
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .main-container {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .input-group {
            padding: 1rem;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# Load model & transformer
# ========================
@st.cache_resource
def load_models():
    try:
        model = CatBoostClassifier()
        model.load_model("models/catboost_simplified_model.cbm")
        transformer = joblib.load('models/transformer.pkl')
        ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')
        return model, transformer, ordinal_encoder
    except:
        return None, None, None

model, transformer, ordinal_encoder = load_models()


# ========================
# Data Configuration
# ========================
selected_num_cols = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'AvgWorkHours', 'OverTime'
]

selected_ordinal_cols = [
    'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction'
]

selected_nominal_cols = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus'
]

expected_ohe_columns = [
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

# Default values and options
median_dict = {
    "Age": 36.0, "DistanceFromHome": 7.0, "MonthlyIncome": 48980.0,
    "NumCompaniesWorked": 2.0, "PercentSalaryHike": 14.0,
    "TotalWorkingYears": 10.0, "TrainingTimesLastYear": 3.0,
    "YearsAtCompany": 5.0, "YearsSinceLastPromotion": 1.0,
    "YearsWithCurrManager": 3.0, "AvgWorkHours": 7.4, "OverTime": 0.0
}

modus_dict = {
    "BusinessTravel": "Travel_Rarely", "Department": "Research & Development",
    "EducationField": "Life Sciences", "Gender": "Male",
    "JobRole": "Sales Executive", "MaritalStatus": "Married",
    "WorkLifeBalance": 3.0, "JobSatisfaction": 4.0, "EnvironmentSatisfaction": 3.0
}

bounds = {
    'TotalWorkingYears': [-2.47, 2.54],
    'TrainingTimesLastYear': [-1.79, 1.41],
    'YearsAtCompany': [-2.43, 2.49],
}

nominal_options = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Research & Development", "Sales", "Human Resources"],
    "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
    "Gender": ["Male", "Female"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Research Director", "Human Resources", "Sales Representative"],
    "MaritalStatus": ["Single", "Married", "Divorced"]
}

ordinal_options = {
    "WorkLifeBalance": [1, 2, 3, 4],
    "JobSatisfaction": [1, 2, 3, 4],
    "EnvironmentSatisfaction": [1, 2, 3, 4]
}

# ========================
# APP LAYOUT
# ========================

# Header
st.markdown('<h1 class="main-header">HR Analytics & Resignation Prediction</h1>', unsafe_allow_html=True)

# Check if models are loaded
if not all([model, transformer, ordinal_encoder]):
    st.markdown("""
    <div class="prediction-card-high">
        <h3>⚠️ Model Loading Error</h3>
        <p>Terjadi kesalahan saat memuat model. Silakan periksa file model di folder 'models/'</p>
        <p>File yang dibutuhkan:</p>
        <ul>
            <li>models/catboost_simplified_model.cbm</li>
            <li>models/transformer.pkl</li>
            <li>models/ordinal_encoder.pkl</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Input form section
st.markdown("""
    <div class="prediction-container">
        <h3 class="results-header">
            Employee Information
        </h3>
    </div>
    """, unsafe_allow_html=True)

# Create input form
input_data = {}

# Personal Information
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-user"></i>Personal Information</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_data['Age'] = st.number_input("Age", min_value=18, max_value=65, value=36)
    with col2:
        input_data['Gender'] = st.selectbox("Gender", nominal_options['Gender'])
    with col3:
        input_data['MaritalStatus'] = st.selectbox("Marital Status", nominal_options['MaritalStatus'])
    with col4:
        input_data['DistanceFromHome'] = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=7)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Work Information
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-briefcase"></i>Work Information</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_data['Department'] = st.selectbox("Department", nominal_options['Department'])
    with col2:
        input_data['JobRole'] = st.selectbox("Job Role", nominal_options['JobRole'])
    with col3:
        input_data['EducationField'] = st.selectbox("Education Field", nominal_options['EducationField'])
    with col4:
        input_data['BusinessTravel'] = st.selectbox("Business Travel", nominal_options['BusinessTravel'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Experience & Career
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-chart-simple"></i>Experience & Career</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['TotalWorkingYears'] = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
        input_data['YearsWithCurrManager'] = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
    with col2:
        input_data['YearsAtCompany'] = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        input_data['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
    with col3:
        input_data['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=3)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Compensation & Work
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-money-check-dollar"></i>Compensation & Work</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=0, max_value=200000, value=48980)
    with col2:
        input_data['PercentSalaryHike'] = st.number_input("Percent Salary Hike (%)", min_value=0, max_value=50, value=14)
    with col3:
        input_data['AvgWorkHours'] = st.number_input("Average Work Hours per Day", min_value=1.0, max_value=12.0, value=7.4, step=0.1)
    with col4:
        input_data['OverTime'] = st.number_input("Over Time Days", min_value=0, max_value=365, value=0, step=1)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Satisfaction Levels
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-face-smile"></i>Satisfaction Levels</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['WorkLifeBalance'] = st.selectbox("Work Life Balance (1-4)", ordinal_options['WorkLifeBalance'])
    with col2:
        input_data['JobSatisfaction'] = st.selectbox("Job Satisfaction (1-4)", ordinal_options['JobSatisfaction'])
    with col3:
        input_data['EnvironmentSatisfaction'] = st.selectbox("Environment Satisfaction (1-4)", ordinal_options['EnvironmentSatisfaction'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Predict Resignation Risk", key="predict_btn")

st.markdown('</div>', unsafe_allow_html=True)

# Add some space at the bottom
st.markdown('<hr class="divider-custom">', unsafe_allow_html=True)

# Enhanced Prediction Results Section
# Add this CSS for additional styling
st.markdown("""
<style>
.prediction-container {
    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    border-radius: 25px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid rgba(102,126,234,0.2);
    position: relative;
    overflow: hidden;
}

.prediction-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.results-header {
    text-align: center;
    color: #2d3748;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 3rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}

.gauge-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    position: relative;
}

.risk-badge-high {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    color: white;
    padding: 12px 24px;
    border-radius: 25px;
    font-weight: 700;
    font-size: 18px;
    display: inline-block;
    margin-top: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.risk-badge-low {
    background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
    color: white;
    padding: 12px 24px;
    border-radius: 25px;
    font-weight: 700;
    font-size: 18px;
    display: inline-block;
    margin-top: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.shap-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    margin-left: 1rem;
}

.shap-title {
    text-align: center;
    color: #2d3748;
    font-weight: 600;
    font-size: 1.4rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.probability-label {
    text-align: center;
    margin-top: -20px;
    color: #64748b;
    font-size: 16px;
    font-weight: 500;
}

.recommendation-section {
    background: linear-gradient(135deg, #f1f5f9 0%, #ffffff 100%);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid #e2e8f0;
}

.recommendation-header {
    text-align: center;
    color: #2d3748;
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.enhanced-recommendation-card {
    background: white;
    padding: 2rem;
    margin: 1rem 0;
    border-radius: 15px;
    border-left: 5px solid #667eea;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.enhanced-recommendation-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(102,126,234,0.02) 0%, rgba(118,75,162,0.02) 100%);
    pointer-events: none;
}

.enhanced-recommendation-card:hover {
    transform: translateY(-5px);
    border-left-color: #764ba2;
}

.recommendation-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.recommendation-subtitle {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}           
.recommendation-content {
    color: #4a5568;
    line-height: 1.7;
    font-size: 15px;
}

.low-risk-message {
    background: white;
    border-radius: 20px;
    padding: 3rem;
    border: 1px solid #e2e8f0;
    margin-left: 1rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.success-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.success-title {
    color: #2d3748;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.success-message {
    color: #64748b;
    font-size: 16px;
    line-height: 1.6;
    max-width: 400px;
}

.divider-custom {
    height: 2px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 1px;
    margin: 3rem 0;
    opacity: 0.3;
}

.footer-section {
    text-align: center;
    color: #64748b;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    border-radius: 20px;
    margin: 2rem 0;
    border: 1px solid #e2e8f0;
}

.footer-content {
    font-size: 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Main content
if not predict_button:
    # Enhanced welcome message
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); 
                border-radius: 25px; margin: 3rem 0; border: 1px solid rgba(102,126,234,0.2);">
        <h3 style="color: #667eea; margin-bottom: 1.5rem; font-weight: 700; font-size: 2rem;">
            HR Analytics System for Resignation Risk Prediction
        </h3>
        <p style="color: #64748b; font-size: 1.2rem; line-height: 1.8; max-width: 700px; margin: 0 auto;">
            Complete the employee information form above and click <strong>Predict Resignation Risk</strong> 
            to receive comprehensive analysis with actionable insights and strategic recommendations.
        </p>
        <div style="margin-top: 2rem; padding: 1rem; background: rgba(102,126,234,0.05); 
                    border-radius: 15px; border: 1px solid rgba(102,126,234,0.1);">
            <p style="color: #667eea; font-weight: 600; margin: 0;">
                <i class="fa-solid fa-globe"></i> Advanced machine learning  •  <i class="fa-solid fa-square-poll-vertical"></i> SHAP analysis  •  <i class="fa-solid fa-users-viewfinder"></i> Personalized recommendations
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Process prediction with enhanced loading
    with st.spinner('Analyzing employee data'):
        # Your existing prediction processing code here
        df = pd.DataFrame([input_data])
        
        # Convert numeric columns
        for col in selected_num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        for col in selected_num_cols:
            df[col].fillna(median_dict[col], inplace=True)
        for col in selected_ordinal_cols + selected_nominal_cols:
            df[col].fillna(modus_dict[col], inplace=True)
        
        # Apply bounds
        for col in bounds.keys():
            lower, upper = bounds[col]
            df[col] = np.clip(df[col], lower, upper)
        
        # Transform features
        X_num = transformer.transform(df[selected_num_cols])
        X_ord = ordinal_encoder.transform(df[selected_ordinal_cols])
        X_nom = pd.get_dummies(df[selected_nominal_cols])
        for col in expected_ohe_columns:
            if col not in X_nom.columns:
                X_nom[col] = 0
        X_nom = X_nom[expected_ohe_columns]
        
        X_all = np.concatenate([X_num, X_ord, X_nom.values], axis=1)
        
        # Make prediction
        probabilities = model.predict_proba(X_all)[0]
        proba_resign = probabilities[1]
        threshold = 0.73
        prediction = 1 if proba_resign >= threshold else 0

    # Enhanced Results Container
    st.markdown("""
    <div class="prediction-container">
        <h3 class="results-header">
            Prediction Results
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # SHAP Analysis for high risk cases
    if prediction == 1:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_all)
        
        feature_names = selected_num_cols + selected_ordinal_cols + expected_ohe_columns
        shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_pos.flatten() 
        })

        top3_df = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(3)
        
        col_gauge, div_col, col_chart = st.columns([1, 0.1, 1])

        with col_gauge:
            # Enhanced gauge chart
            probability_percent = round(proba_resign * 100, 1)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': ""},
                number = {'font': {'size': 42, 'color': '#2d3748', 'family': 'Inter'}, 'suffix': '%'},
                gauge = {
                    'axis': {
                        'range': [None, 100], 
                        'tickwidth': 2, 
                        'tickcolor': "#e2e8f0",
                        'tickfont': {'size': 12, 'color': '#64748b', 'family': 'Inter'}
                    },
                    'bar': {'color': "#ff6b6b", 'thickness': 0.35},
                    'bgcolor': "#f8fafc",
                    'borderwidth': 4,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 4},
                        'thickness': 0.9,
                        'value': 73
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=150,
                margin=dict(l=20, r=20, t=20, b=20),
                font={'color': "#2d3748", 'family': 'Inter'},
                paper_bgcolor="white"
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Enhanced status badge
            st.markdown(f"""
            <div class="probability-label">
                <p style="font-size: 16px;">Resignation Probability</p>
                <div class="risk-badge-high" style="margin-top: -5px;">
                    <i class="fa-solid fa-triangle-exclamation"></i> HIGH RISK
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart:
            with st.container(border=True):
                # Enhanced SHAP title
                st.markdown("""
                <h5 class="shap-title">Top Risk Factors</h5>
                """, unsafe_allow_html=True)
                
                # Sort by SHAP value (descending for highest impact first)
                top3_df = top3_df.sort_values('shap_value', ascending=False)
                
                # Display top 3 factors as simple text
                for i, (_, row) in enumerate(top3_df.iterrows(), 1):
                    st.markdown(f"""
                    <div style="margin-left: 20px; margin-bottom: 10px;">
                        <strong>{i}. {row['feature']}</strong> - Impact Score: {row['shap_value']:.2f}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced divider
        st.markdown('<hr class="divider-custom">', unsafe_allow_html=True)

        # Enhanced Recommendations section
        st.markdown("""<div class="prediction-container"><h3 class="results-header">Strategic Recommendations</h3></div>""",unsafe_allow_html=True)    
        col1, col2 = st.columns(2)
        
        recommendations = []
        for _, row in top3_df.iterrows():
            feat = row['feature']
            if any(x in feat for x in ['MonthlyIncome', 'PercentSalaryHike']):
                recommendations.append(("<i class='fa-solid fa-wallet'></i> Compensation Strategy", "Offer Competitive (Market-Rate) Salaries", "A significant increase in income or benefits is considered \"very important\" by 80%+ employees. Ensuring pay matches market rates is key to attract and retain talent (<a href='https://www.gallup.com/workplace/389807/top-things-employees-next-job.aspx'>Source</a>)"))
            elif any(x in feat for x in ['OverTime', 'WorkLifeBalance', 'AvgWorkHours', 'DistanceFromHome']):
                recommendations.append(("<i class='fa-solid fa-scale-balanced'></i> Work-Life Balance", "Implement Flexible Work Policies", "Around 50% of employees would leave for better work-life balance and flexible arrangements (<a href='https://www.ey.com/en_sg/newsroom/2022/07/employee-influence-in-singapore-grows-51-percentage-set-to-quit-jobs-for-better-pay-career-opportunities-and-flexibility'>Source</a>)"))
            elif any(x in feat for x in ['TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'NumCompaniesWorked', 'TrainingTimesLastYear']):
                recommendations.append(("<i class='fa-solid fa-rocket'></i> Career Development", "Provide Development & Training Programs", "94% say they’d stay longer if their company invested in career growth and training (<a href='https://www.linkedin.com/posts/linkedinlearning_94-of-employees-say-that-they-would-stay-activity-6372632932300963840-k9hI'>Source</a>)"))
            elif any(x in feat for x in ['EnvironmentSatisfaction', 'YearsWithCurrManager', 'JobSatisfaction']):
                recommendations.append(("<i class='fa-solid fa-handshake'></i> Workplace Culture", "Conduct Stay Interviews", "Stay interviews help identify satisfaction drivers and fix issues before people leave (<a href='https://hr.nih.gov/sites/default/files/public/documents/2024-03/stay-interview-guide.pdf'>Source</a>)"))
            elif any(x in feat for x in ['JobRole', 'Department', 'EducationField']):
                recommendations.append(("<i class='fa-solid fa-people-arrows'></i> Role Optimization", "Strategic Job Rotation", "Job rotation can increase retention by up to 23% and boost satisfaction by 15% (<a href='https://blogs.psico-smart.com/blog-maximizing-the-effectiveness-of-employee-rotation-programs-best-practices-and-strategies-12164'>Source</a>)"))
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec[0] not in seen:
                unique_recommendations.append(rec)
                seen.add(rec[0])
        
        for i, (title, subtitle, content) in enumerate(unique_recommendations):
            if i % 2 == 0:
                with col1:
                    st.markdown(f"""
                    <div class="enhanced-recommendation-card">
                        <div class="recommendation-title">{title}</div>
                        <div class="recommendation-subtitle">{subtitle}</div>
                        <div class="recommendation-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown(f"""
                    <div class="enhanced-recommendation-card">
                        <div class="recommendation-title">{title}</div>
                        <div class="recommendation-subtitle">{subtitle}</div>
                        <div class="recommendation-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Low risk scenario
        col_gauge, col_message = st.columns([1, 1.5])

        with col_gauge:
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            
            probability_percent = round(proba_resign * 100, 1)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': ""},
                number = {'font': {'size': 60, 'color': '#2d3748', 'family': 'Inter'}, 'suffix': '%'},
                gauge = {
                    'axis': {
                        'range': [None, 100], 
                        'tickwidth': 2, 
                        'tickcolor': "#e2e8f0",
                        'tickfont': {'size': 12, 'color': '#64748b', 'family': 'Inter'}
                    },
                    'bar': {'color': "#51cf66", 'thickness': 0.35},
                    'bgcolor': "#f8fafc",
                    'borderwidth': 4,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "#16a34a", 'width': 4},
                        'thickness': 0.9,
                        'value': 30
                    }
                }
            ))

            fig_gauge.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=-0, b=0),
                font={'color': "#2d3748", 'family': 'Inter'},
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div class="probability-label" style="margin-top: -50px;">
                <p style="font-size: 16px;">Resignation Probability</p>
                <div class="risk-badge-low" style="margin-top: -5px;">
                    <i class="fa-solid fa-shield"></i> LOW RISK
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col_message:
            st.markdown("""
            <div class="low-risk-message">
                <div class="success-icon">🎉</div>
                <h6 class="success-title">Excellent Retention Outlook</h6>
                <p class="success-message">
                    This employee shows strong indicators for staying with the company. 
                    Continue current engagement strategies and maintain regular check-ins 
                    to ensure continued satisfaction and growth.
                </p>
                <div style="margin-top: 2rem; padding: 1rem; background: #f0fdf4; 
                            border-radius: 10px; border: 1px solid #bbf7d0;">
                    <p style="color: #16a34a; font-weight: 600; margin: 0; font-size: 14px;">
                        💡 Recommendation: Focus on career development and recognition programs
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Enhanced divider
    st.markdown('<hr class="divider-custom">', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer-section">
    <div class="footer-content">
        <p style="margin: 0; font-size: 18px; font-weight: 600; color: #2d3748;">
            <i class="fa-solid fa-chart-simple"></i> HR Analytics Dashboard v2.0
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #64748b;">
            Powered by Machine Learning • Built with ❤️ using Streamlit & Plotly
        </p>
    </div>
</div>
""", unsafe_allow_html=True)