# ========================
# IMPORT LIBRARIES
# ========================
import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import shap
import plotly.graph_objects as go
from datetime import datetime

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide"
)

# Load Font Awesome for icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ========================
# ENHANCED CUSTOM CSS STYLING
# ========================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font family */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 1rem 2rem;
        min-height: 100vh;
    }

    /* Main header styling */
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

    /* Main container card styling */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Input section styling */
    .input-section {
        background: white;
        padding: 0;
        margin: 2rem 0;
        border-radius: 15px;
    }

    /* Input group styling */
    .input-group {
        margin-top: 1.5rem;
        transition: all 0.3s ease;
    }

    /* Input group header styling */
    .input-group h4 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Section header styling */
    .section-header {
        text-align: center;
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
        position: relative;
    }

    /* Section header underline effect */
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

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e1e8f7;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    /* Selectbox focus styling */
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
    }

    /* Number input styling */
    .stNumberInput > div > div > input {
        background: white;
        border: 2px solid #e1e8f7;
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    /* Number input focus styling */
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
    }

    /* Button styling */
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

    /* Button hover effect */
    .stButton > button:hover {
        transform: translateY(-3px);
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%);
    }

    /* Button active effect */
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }

    /* Metric card hover effect */
    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* High risk prediction card styling */
    .prediction-card-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }

    /* Low risk prediction card styling */
    .prediction-card-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }

    /* Recommendation card styling */
    .recommendation-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    /* Recommendation card hover effect */
    .recommendation-card:hover {
        transform: translateX(5px);
    }

    /* Recommendation card header styling */
    .recommendation-card h4 {
        margin-top: 0;
        font-size: 1.1rem;
        color: #2d3748;
        font-weight: 600;
    }

    /* Recommendation card paragraph styling */
    .recommendation-card p {
        margin: 0;
        line-height: 1.6;
        font-size: 0.95rem;
        color: #4a5568;
    }

    /* Main block container max width */
    .main .block-container {
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Loading animation spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    /* Spinner animation keyframes */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive design for mobile */
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

    /* Custom scrollbar track */
    ::-webkit-scrollbar {
        width: 8px;
    }

    /* Custom scrollbar track background */
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    /* Custom scrollbar thumb */
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Custom scrollbar thumb hover effect */
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%);
    }
    
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

# ========================
# MODEL LOADING FUNCTION
# ========================
@st.cache_resource
def load_models():
    """
    Load the trained CatBoost model and preprocessing transformers.
    
    Returns:
        tuple: (model, transformer, ordinal_encoder) if successful, (None, None, None) if failed
    """
    try:
        # Load CatBoost model
        model = CatBoostClassifier()
        model.load_model("models/catboost_simplified_model.cbm")
        
        # Load preprocessing transformers
        transformer = joblib.load('models/transformer.pkl')
        ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')
        
        return model, transformer, ordinal_encoder
    except:
        return None, None, None

# Load models and transformers
model, transformer, ordinal_encoder = load_models()

# ========================
# DATA CONFIGURATION
# ========================

# Selected numerical columns for model input
selected_num_cols = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'AvgWorkHours', 'OverTime'
]

# Selected ordinal columns for model input
selected_ordinal_cols = [
    'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction'
]

# Selected nominal/categorical columns for model input
selected_nominal_cols = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus'
]

# Expected one-hot encoded columns after preprocessing
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

# Default median values for numerical features
median_dict = {
    "Age": 36.0, "DistanceFromHome": 7.0, "MonthlyIncome": 48980.0,
    "NumCompaniesWorked": 2.0, "PercentSalaryHike": 14.0,
    "TotalWorkingYears": 10.0, "TrainingTimesLastYear": 3.0,
    "YearsAtCompany": 5.0, "YearsSinceLastPromotion": 1.0,
    "YearsWithCurrManager": 3.0, "AvgWorkHours": 7.4, "OverTime": 0.0
}

# Default mode values for categorical and ordinal features
modus_dict = {
    "BusinessTravel": "Travel_Rarely", "Department": "Research & Development",
    "EducationField": "Life Sciences", "Gender": "Male",
    "JobRole": "Sales Executive", "MaritalStatus": "Married",
    "WorkLifeBalance": 3.0, "JobSatisfaction": 4.0, "EnvironmentSatisfaction": 3.0
}

# Bounds for certain features (likely used for validation or scaling)
bounds = {
    'TotalWorkingYears': [-2.47, 2.54],
    'TrainingTimesLastYear': [-1.79, 1.41],
    'YearsAtCompany': [-2.43, 2.49],
}

# Available options for nominal/categorical features
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

# Available options for ordinal features (satisfaction scales)
ordinal_options = {
    "WorkLifeBalance": [1, 2, 3, 4],
    "JobSatisfaction": [1, 2, 3, 4],
    "EnvironmentSatisfaction": [1, 2, 3, 4]
}

def beautify_feature_name(raw_name):
    if '_' in raw_name:
        prefix, value = raw_name.split('_', 1)
        value = value.replace('_', ' ').title()
        prefix = re.sub(r'([a-z])([A-Z])', r'\1 \2', prefix).title()
        return f"{prefix}: {value}"
    else:
        pretty_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_name)
        return pretty_name.title()

# ========================
# STREAMLIT UI COMPONENTS
# ========================

# Header
# Create the main title and subtitle for the resignation prediction application
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Individual Resignation Prediction</h1>
    <p style="color: black; margin: 0; opacity: 0.9; text-align: center;">Personalized Attrition Risk Assessment — Predict an Employee's Likelihood to Resign</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Check if models are loaded
# Verify that all required machine learning models are properly loaded before proceeding
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
# Create the main container for employee information input
st.markdown("""
    <div class="prediction-container">
        <h3 class="results-header">
            Employee Information
        </h3>
    </div>
    """, unsafe_allow_html=True)

# Create input form
# Initialize dictionary to store all input data from the user
input_data = {}

# Personal Information
# Section for collecting basic personal details of the employee
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-user"></i>Personal Information</h4>
    """, unsafe_allow_html=True)
    
    # Create 4 columns for personal information inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Age input with validation range
        input_data['Age'] = st.number_input("Age", min_value=18, max_value=65, value=36)
    with col2:
        # Gender selection from predefined options
        input_data['Gender'] = st.selectbox("Gender", nominal_options['Gender'])
    with col3:
        # Marital status selection
        input_data['MaritalStatus'] = st.selectbox("Marital Status", nominal_options['MaritalStatus'])
    with col4:
        # Distance from home in kilometers
        input_data['DistanceFromHome'] = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=7)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Work Information
# Section for collecting work-related information
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-briefcase"></i>Work Information</h4>
    """, unsafe_allow_html=True)
    
    # Create 4 columns for work information inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Department selection
        input_data['Department'] = st.selectbox("Department", nominal_options['Department'])
    with col2:
        # Job role selection
        input_data['JobRole'] = st.selectbox("Job Role", nominal_options['JobRole'])
    with col3:
        # Education field selection
        input_data['EducationField'] = st.selectbox("Education Field", nominal_options['EducationField'])
    with col4:
        # Business travel frequency
        input_data['BusinessTravel'] = st.selectbox("Business Travel", nominal_options['BusinessTravel'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Experience & Career
# Section for collecting career progression and experience data
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-chart-simple"></i>Experience & Career</h4>
    """, unsafe_allow_html=True)
    
    # Create 3 columns for experience inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        # Total working years across all companies
        input_data['TotalWorkingYears'] = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
        # Years with current manager
        input_data['YearsWithCurrManager'] = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
    with col2:
        # Years at current company
        input_data['YearsAtCompany'] = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        # Number of companies worked at previously
        input_data['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
    with col3:
        # Years since last promotion
        input_data['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        # Training frequency in the last year
        input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=3)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Compensation & Work
# Section for collecting salary and work-related metrics
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-money-check-dollar"></i>Compensation & Work</h4>
    """, unsafe_allow_html=True)
    
    # Create 4 columns for compensation inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Monthly income in local currency
        input_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=0, max_value=200000, value=48980)
    with col2:
        # Percentage salary increase
        input_data['PercentSalaryHike'] = st.number_input("Percent Salary Hike (%)", min_value=0, max_value=50, value=14)
    with col3:
        # Average daily work hours
        input_data['AvgWorkHours'] = st.number_input("Average Work Hours per Day", min_value=1.0, max_value=12.0, value=7.4, step=0.1)
    with col4:
        # Number of overtime days per year
        input_data['OverTime'] = st.number_input("Over Time Days", min_value=0, max_value=365, value=0, step=1)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Satisfaction Levels
# Section for collecting employee satisfaction ratings
with st.container():
    st.markdown("""
    <div class="input-group">
        <h4><i class="fa-solid fa-face-smile"></i>Satisfaction Levels</h4>
    """, unsafe_allow_html=True)
    
    # Create 3 columns for satisfaction inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        # Work-life balance rating (1-4 scale)
        input_data['WorkLifeBalance'] = st.selectbox("Work Life Balance (1-4)", ordinal_options['WorkLifeBalance'])
    with col2:
        # Job satisfaction rating (1-4 scale)
        input_data['JobSatisfaction'] = st.selectbox("Job Satisfaction (1-4)", ordinal_options['JobSatisfaction'])
    with col3:
        # Environment satisfaction rating (1-4 scale)
        input_data['EnvironmentSatisfaction'] = st.selectbox("Environment Satisfaction (1-4)", ordinal_options['EnvironmentSatisfaction'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Predict button
# Create the prediction button centered on the page
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Main prediction button that triggers the model inference
    predict_button = st.button("Predict Resignation Risk", key="predict_btn")

st.markdown('</div>', unsafe_allow_html=True)

# Add some space at the bottom
# Add visual separator at the bottom of the form
st.markdown('<hr class="divider-custom">', unsafe_allow_html=True)

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
        
        col_gauge, _, col_status = st.columns([1, 0.2, 1])

        with col_gauge:
            probability_percent = round(proba_resign * 100, 1)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={'font': {'size': 58, 'color': '#2d3748', 'family': 'Inter'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff6b6b", 'thickness': 0.35},
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 4},
                        'thickness': 0.9,
                        'value': probability_percent
                    }
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_status:
            today = datetime.today().strftime('%d %B %Y')
            st.markdown(f"""
            <div style="
                background-color:#dc2626;
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                color: white;
                height: 170px;
                font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 8px;"><i class="fa-solid fa-triangle-exclamation"></i> HIGH RISK</h4>
                <p style="margin: 0; font-size: 16px;">Employee has a <strong>{probability_percent:.1f}%</strong> probability of resignation</p>
                <p style="margin-top: 10px; font-size: 14px;">Last Updated: {today}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<h4 style="margin-bottom: 10px;">Top Risk Factors</h4>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        top3_df = top3_df.sort_values('shap_value', ascending=False).reset_index(drop=True)

        for i, col in enumerate([col1, col2, col3]):
            if i < len(top3_df):
                raw_feature = top3_df.loc[i, 'feature']
                feature = beautify_feature_name(raw_feature)
                shap_val = top3_df.loc[i, 'shap_value']

                with col:
                    col.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); text-align: center;
                                padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; color: white;">
                        <h5 style="font-size: 18px; margin-top: 10px; margin-bottom: 2px;">{feature}</h5>
                        <p style="font-size: 14px;">Impact Score: <strong>{shap_val:.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)


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
        col_gauge, col, col_message = st.columns([1, 0.2, 1.5])

        with col_gauge:
            probability_percent = round(proba_resign * 100, 1)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={'font': {'size': 58, 'color': '#2d3748', 'family': 'Inter'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff6b6b", 'thickness': 0.35},
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 4},
                        'thickness': 0.9,
                        'value': probability_percent
                    }
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_message:
            today = datetime.today().strftime('%d %B %Y')
            st.markdown(f"""
            <div style="
                background-color:#4CAF50;
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                color: white;
                height: 170px;
                font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 8px;"><i class="fa-solid fa-circle-check"></i> LOW RISK</h4>
                <p style="padding-right: 10px; padding-left: 10px; font-size: 16px;">Employee shows strong retention potential. Continue current engagement and regular check-ins to support satisfaction and growth.</p>
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