# Import required libraries
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import random

# Configure Streamlit page settings
st.set_page_config(page_title="Welcome - 3Sigma Squad", layout="wide")

# Define custom CSS styles for section headers
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
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
    <h1 style="color: black; margin: 0; text-align: center;">Know Who's at Risk</h1>
    <p style="text-align: center; font-size: 16px; color: gray;">Attrition App by 3Sigma Squad</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Hero section explaining the app's purpose
st.markdown('<div class="section-header"><h3><i class="bi bi-window-stack"></i> What This App Does</h3></div>', unsafe_allow_html=True)
st.markdown("""
This dashboard helps HR identify employees at risk of leaving the company using machine learning.

**What you can do:**  
- Predict attrition for individuals or in bulk  
- View risk scores and key contributing factors  
- Download predictions and act accordingly

To get started, choose a page below or use the sidebar navigation.
""")

# Section for showing dashboard template
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="bi bi-file-earmark-spreadsheet"></i> Dashboard: Excel Template</h3></div>', unsafe_allow_html=True)
st.markdown("Use this sample format to prepare your employee data for dashboard.")

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

# Generate and display batch prediction template
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="bi bi-file-earmark-excel"></i> Batch Prediction: Excel Template</h3></div>', unsafe_allow_html=True)
st.markdown("Use this sample format to prepare your employee data for batch prediction.")

num_rows = 6  
dummy_data = {}

dummy_data['EmployeeID'] = [100, 101, 102, 103, 104, 105]
dummy_data['Age'] = [34, 28, 45, 33, 29, 52]
dummy_data['DistanceFromHome'] = [8, 15, 3, 12, 22, 5]
dummy_data['Education'] = [3, 2, 4, 3, 1, 4]
dummy_data['JobLevel'] = [2, 1, 3, 2, 1, 4]
dummy_data['MonthlyIncome'] = [85420, 45200, 92800, 58900, 38750, 118600]
dummy_data['NumCompaniesWorked'] = [2, 1, 3, 1, 0, 4]
dummy_data['PercentSalaryHike'] = [14, 18, 12, 16, 21, 13]
dummy_data['StockOptionLevel'] = [1, 0, 2, 1, 0, 3]
dummy_data['TotalWorkingYears'] = [12, 5, 18, 9, 4, 25]
dummy_data['TrainingTimesLastYear'] = [3, 2, 4, 3, 2, 5]
dummy_data['YearsAtCompany'] = [8, 3, 15, 7, 2, 22]
dummy_data['YearsSinceLastPromotion'] = [2, 1, 5, 1, 0, 8]
dummy_data['YearsWithCurrManager'] = [3, 2, 8, 4, 1, 12]
dummy_data['EnvironmentSatisfaction'] = [3, 2, 4, 3, 2, 4]
dummy_data['JobSatisfaction'] = [3, 3, 4, 3, 2, 4]
dummy_data['WorkLifeBalance'] = [4, 3, 4, 3, 2, 3]
dummy_data['JobInvolvement'] = [3, 2, 2, 3, 2, 3]
dummy_data['PerformanceRating'] = [3, 4, 3, 3, 4, 3]
dummy_data['AvgWorkHours'] = [7.523894561, 8.145672389, 7.892456123, 7.634521987, 6.847293156, 8.756342891]
dummy_data['AbsentDays'] = [22, 28, 18, 25, 31, 16]
dummy_data['OverTime'] = [45, 12, 78, 23, 8, 156]

dummy_data['Attrition'] = ["No", "Yes", "No", "No", "Yes", "No"]
dummy_data['BusinessTravel'] = ["Travel_Rarely", "Travel_Frequently", "Non-Travel",
                                "Travel_Rarely", "Travel_Frequently", "Travel_Rarely"]
dummy_data['Department'] = ["Sales", "Research & Development", "Human Resources",
                            "Research & Development", "Sales", "Research & Development"]
dummy_data['EducationField'] = ["Business", "Life Sciences", "Human Resources",
                                "Medical", "Marketing", "Life Sciences"]
dummy_data['Gender'] = ["Male", "Female", "Female", "Male", "Female", "Male"]
dummy_data['JobRole'] = ["Sales Executive", "Research Scientist", "HR Manager",
                         "Laboratory Technician", "Sales Representative", "Research Director"]
dummy_data['MaritalStatus'] = ["Married", "Single", "Married", "Divorced", "Single", "Married"]

df_dummy = pd.DataFrame(dummy_data)
st.dataframe(df_dummy)

# Create downloadable Excel template
output_dummy = io.BytesIO()
with pd.ExcelWriter(output_dummy, engine='xlsxwriter') as writer:
    df_dummy.to_excel(writer, index=False, sheet_name='Template')

dummy_excel = output_dummy.getvalue()

st.download_button(
    label="Download Batch Template",
    data=dummy_excel,
    file_name='example_template.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Quick actions section with navigation buttons
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="bi bi-cursor"></i> Quick Actions</h3></div>', unsafe_allow_html=True)

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
    <small>3Sigma Squad © 2025 | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
