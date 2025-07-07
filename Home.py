# Import required libraries
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import random

# Configure Streamlit page settings
st.set_page_config(page_title="Welcome - 3Sigma Squad", layout="wide")

# Load Font Awesome CSS for icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# Define custom CSS styles for section headers
st.markdown("""
<style>
    .section-header {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define feature columns for the ML model
# Numerical features
selected_num_cols = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'AvgWorkHours', 'OverTime'
]

# Ordinal features (ordered categorical)
selected_ordinal_cols = [
    'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction'
]

# Nominal features (unordered categorical)
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

# Default mode values for categorical features
modus_dict = {
    "BusinessTravel": "Travel_Rarely", "Department": "Research & Development",
    "EducationField": "Life Sciences", "Gender": "Male",
    "JobRole": "Sales Executive", "MaritalStatus": "Married",
    "WorkLifeBalance": 3.0, "JobSatisfaction": 4.0, "EnvironmentSatisfaction": 3.0
}

# Bounds for feature scaling/normalization
bounds = {
    'TotalWorkingYears': [-2.47, 2.54],
    'TrainingTimesLastYear': [-1.79, 1.41],
    'YearsAtCompany': [-2.43, 2.49],
}

# Available options for nominal categorical features
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

# Available options for ordinal categorical features
ordinal_options = {
    "WorkLifeBalance": [1, 2, 3, 4],
    "JobSatisfaction": [1, 2, 3, 4],
    "EnvironmentSatisfaction": [1, 2, 3, 4]
}

# Display main header and title
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Attrition Prediction Dashboard</h1>
    <p style="text-align: center; font-size: 16px; color: gray;">by 3Sigma Squad</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Hero section explaining the app's purpose
st.markdown('<div class="section-header"><h3><i class="fa-solid fa-brain"></i> What This App Does</h3></div>', unsafe_allow_html=True)
st.markdown("""
This dashboard helps HR identify employees at risk of leaving the company using machine learning.

**What you can do:**  
- Predict attrition for individuals or in bulk  
- View risk scores and key contributing factors  
- Download predictions and act accordingly

To get started, choose a page below or use the sidebar navigation.
""")

# Section for showing dummy template
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="fa-solid fa-file-excel"></i> Batch Prediction: Excel Template</h3></div>', unsafe_allow_html=True)
st.markdown("Use this sample format to prepare your employee data for batch prediction.")

# Generate dummy data for template
num_rows = 3
dummy_data = {}

# Create sample numerical data
for col in selected_num_cols:
    median = median_dict.get(col, 0)
    dummy_data[col] = [median + i for i in range(num_rows)]

# Create sample ordinal data
for col in selected_ordinal_cols:
    modus = modus_dict.get(col, 1)
    dummy_data[col] = [modus + i for i in range(num_rows)]

# Create sample nominal data
for col in selected_nominal_cols:
    options = nominal_options.get(col, [''])
    row_values = []
    for i in range(num_rows):
        if len(options) > i:
            row_values.append(options[i])
        else:
            row_values.append(random.choice(options))
    dummy_data[col] = row_values

# Create DataFrame from dummy data and display it
df_dummy = pd.DataFrame(dummy_data)
st.dataframe(df_dummy)

# Create Excel file in memory for download
output_dummy = io.BytesIO()
with pd.ExcelWriter(output_dummy, engine='xlsxwriter') as writer:
    df_dummy.to_excel(writer, index=False, sheet_name='Template')

dummy_excel = output_dummy.getvalue()

# Download button for the template
st.download_button(
    label="Download Example Template",
    data=dummy_excel,
    file_name='example_template.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Quick actions section with navigation buttons
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="fa-solid fa-location-arrow"></i> Quick Actions</h3></div>', unsafe_allow_html=True)

# Create three columns for navigation buttons
col1, col2, col3 = st.columns(3)

# Navigation buttons to other pages
with col1:
    if st.button("Dashboard", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")  # Navigate to dashboard page

with col2:
    if st.button("Individual Prediction", use_container_width=True):
        st.switch_page("pages/2_Individual_Prediction.py")  # Navigate to individual prediction page

with col3:
    if st.button("Batch Prediction", use_container_width=True):
        st.switch_page("pages/3_Batch_Prediction.py")  # Navigate to batch prediction page

# Footer with current timestamp
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>3Sigma Squad © 2024 | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)