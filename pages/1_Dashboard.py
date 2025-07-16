# Employee Attrition Dashboard - 3Sigma Squad
# This dashboard provides real-time insights into employee retention and attrition patterns

import streamlit as st
import plotly.express as px
import pandas as pd
import io
import random

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# Set page configuration with title, layout, and sidebar state
st.set_page_config(
    page_title="Employee Attrition Dashboard - 3Sigma Squad", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

# Custom CSS for enhanced visual appearance
st.markdown("""
<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css'>
<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css'>
<style>
    .main-header {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(102,126,234,0.2);
        position: relative;
        overflow: hidden;
    }            
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .section-header {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .insight-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA SCHEMA DEFINITIONS
# =============================================================================

# Define numerical columns for the dataset
selected_num_cols = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'AvgWorkHours', 'OverTime'
]

# Define ordinal columns (ranked/ordered categories)
selected_ordinal_cols = [
    'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction'
]

# Define nominal columns (categorical without order)
selected_nominal_cols = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus'
]

# =============================================================================
# DEFAULT VALUES AND OPTIONS
# =============================================================================

# Median values for numerical columns (used for template generation)
median_dict = {
    "Age": 36.0, "DistanceFromHome": 7.0, "MonthlyIncome": 48980.0,
    "NumCompaniesWorked": 2.0, "PercentSalaryHike": 14.0,
    "TotalWorkingYears": 10.0, "TrainingTimesLastYear": 3.0,
    "YearsAtCompany": 5.0, "YearsSinceLastPromotion": 1.0,
    "YearsWithCurrManager": 3.0, "AvgWorkHours": 7.4, "OverTime": 0.0
}

# Mode values for categorical columns (most common values)
modus_dict = {
    "BusinessTravel": "Travel_Rarely", "Department": "Research & Development",
    "EducationField": "Life Sciences", "Gender": "Male",
    "JobRole": "Sales Executive", "MaritalStatus": "Married",
    "WorkLifeBalance": 3.0, "JobSatisfaction": 4.0, "EnvironmentSatisfaction": 3.0
}

# Available options for each categorical column
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

# =============================================================================
# HEADER SECTION
# =============================================================================

# Display main dashboard header with title and description
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Employee Attrition Dashboard</h1>
    <p style="color: black; margin: 0; opacity: 0.9; text-align: center;">Real-time insights into employee retention and attrition patterns</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# FILE UPLOAD AND TEMPLATE GENERATION
# =============================================================================

# File uploader for Excel/CSV files
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])

# Display Excel template section
st.markdown("#### <i class='bi bi-file-earmark-spreadsheet'></i> Excel Template", unsafe_allow_html=True)

# Generate dummy data for template (3 rows of sample data)
num_rows = 3
dummy_data = {}

# Generate numerical columns with incremental values based on median
for col in selected_num_cols:
    median = median_dict.get(col, 0)
    dummy_data[col] = [median + i for i in range(num_rows)]

# Generate ordinal columns with incremental values based on mode
for col in selected_ordinal_cols:
    modus = modus_dict.get(col, 1)
    dummy_data[col] = [modus + i for i in range(num_rows)]

# Generate nominal columns with varied options
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

# =============================================================================
# TEMPLATE DOWNLOAD FUNCTIONALITY
# =============================================================================

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

# =============================================================================
# DATA LOADING
# =============================================================================

# Load data from uploaded file or default file
if uploaded_file is not None:
    # Determine file type based on extension and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    # Load default dataset if no file is uploaded
    df = pd.read_excel('data/raw/attrition.xlsx')

st.markdown("---")

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

# Create sidebar header for filters
st.sidebar.header("Filters & Controls")

# Department filter dropdown
departments = ['All'] + list(df['Department'].unique())
selected_dept = st.sidebar.selectbox("Department", departments)

# Job Role filter (dependent on department selection)
if selected_dept != 'All':
    # Filter job roles based on selected department
    job_roles = ['All'] + list(df[df['Department'] == selected_dept]['JobRole'].unique())
else:
    # Show all job roles if no department is selected
    job_roles = ['All'] + list(df['JobRole'].unique())
selected_role = st.sidebar.selectbox("Job Role", job_roles)

# Age range filter slider
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

# =============================================================================
# DATA FILTERING
# =============================================================================

# Apply filters to the dataset
filtered_df = df.copy()

# Apply department filter
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]

# Apply job role filter
if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]

# Apply age range filter
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Define color scheme for attrition visualization
color_map = {
    'Yes': '#e74c3c',    # Red for employees who left
    'No': 'lightgrey'    # Gray for employees who stayed
}

# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

# Calculate key metrics from filtered data
total_employees = len(filtered_df)
attrited_employees = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
attrition_rate = (attrited_employees / total_employees * 100) if total_employees > 0 else 0
average_tenure = filtered_df['YearsAtCompany'].mean()
average_job_satisfaction = filtered_df['JobSatisfaction'].mean()
high_performers_attrited = len(filtered_df[(filtered_df['Attrition'] == 'Yes') & (filtered_df['PerformanceRating'] >= 3)])

# Executive Summary
# Display the main header with icon for the executive summary section
st.markdown('<div class="section-header"><h3><i class="bi bi-card-checklist"></i> Executive Summary</h3></div>', unsafe_allow_html=True)

# Create 5 columns for key metrics display
col1, col2, col3, col4, col5 = st.columns(5)

# Column 1: Display attrition rate metric with comparison to baseline
with col1:
    st.metric(
        label="Attrition Rate", 
        value=f"{attrition_rate:.1f}%",
        delta=f"{attrition_rate - 16.1:.1f}%" if attrition_rate != 16.1 else None
    )

# Column 2: Display total employees with filtering indication
with col2:
    st.metric(
        label="Total Employees", 
        value=f"{total_employees:,}",
        delta=f"{total_employees - len(df)} filtered" if total_employees != len(df) else None
    )

# Column 3: Display number of employees who left with comparison to baseline
with col3:
    st.metric(
        label="Attrited", 
        value=f"{attrited_employees}",
        delta=f"{attrited_employees - 237}" if attrited_employees != 711 else None
    )

# Column 4: Display average tenure with comparison to baseline
with col4:
    st.metric(
        label="Avg Tenure", 
        value=f"{average_tenure:.1f} yrs",
        delta=f"{average_tenure - 7.0:.1f}" if abs(average_tenure - 7.0) > 0.1 else None
    )

# Column 5: Display average job satisfaction with comparison to baseline
with col5:
    st.metric(
        label="Job Satisfaction", 
        value=f"{average_job_satisfaction:.1f}/4",
        delta=f"{average_job_satisfaction - 2.7:.1f}" if abs(average_job_satisfaction - 2.7) > 0.1 else None
    )

# Key Insights - Risk Level Assessment
# Determine risk level and corresponding color/message based on attrition rate
if attrition_rate > 20:
    insight_color = "#ff4444"  # Red for high risk
    insight = "⚠️ High Risk: Attrition rate is above 20% - immediate attention required!"
elif attrition_rate > 15:
    insight_color = "#ff8800"  # Orange for moderate risk
    insight = "⚡ Moderate Risk: Attrition rate is elevated - monitor closely"
else:
    insight_color = "#00aa00"  # Green for healthy range
    insight = "✅ Healthy: Attrition rate is within acceptable range"

# Display the risk assessment banner with appropriate styling
st.markdown(f"""
<div style="background: {insight_color}20; border: 1px solid {insight_color}; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
    {insight}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Demographics & Segmentation
# Display the demographics section header
st.markdown('<div class="section-header"><h3><i class="bi bi-person-gear"></i> Demographics & Segmentation</h3></div>', unsafe_allow_html=True)

# Create 2 columns for the first row of demographic charts
col1, col2 = st.columns(2)

# Age Group Analysis
# Create age group bins and labels for categorization
bins = [1, 25, 35, 45, 100]
labels = ['<25', '25-35', '35-45', '45+']
filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, right=False)

# Group by age group and attrition status, calculate percentages
grouped = filtered_df.groupby(['AgeGroup', 'Attrition'], observed=False).size().reset_index(name='Count')
total_per_group = grouped.groupby('AgeGroup', observed=False)['Count'].transform('sum')
grouped['Percentage'] = grouped['Count'] / total_per_group * 100

# Display age group attrition chart
with col1:
    fig_age = px.bar(
        grouped, 
        x='AgeGroup', 
        y='Percentage', 
        color='Attrition', 
        title='Attrition Rate by Age Group',
        color_discrete_map=color_map
    )
    fig_age.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig_age, use_container_width=True)

# Gender Analysis
# Group by gender and attrition status, calculate percentages
gender_grouped = filtered_df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
gender_total = gender_grouped.groupby('Gender')['Count'].transform('sum')
gender_grouped['Percentage'] = gender_grouped['Count'] / gender_total * 100

# Display gender attrition sunburst chart
with col2:
    fig_gender = px.sunburst(
        gender_grouped,
        path=['Gender', 'Attrition'],
        values='Count',
        title='Attrition by Gender',
        color='Attrition',
        color_discrete_map=color_map
    )
    fig_gender.update_layout(height=400)
    st.plotly_chart(fig_gender, use_container_width=True)

# Create 2 columns for the second row of demographic charts
col3, col4 = st.columns(2)

# Marital Status Analysis
# Create crosstab and calculate percentages for marital status
marital_crosstab = pd.crosstab(filtered_df['MaritalStatus'], filtered_df['Attrition'])
marital_percent = marital_crosstab.div(marital_crosstab.sum(axis=1), axis=0) * 100
marital_melted = marital_percent.reset_index().melt(id_vars='MaritalStatus', var_name='Attrition', value_name='Percent')

# Display marital status attrition chart
with col3:
    fig_marital = px.bar(
        marital_melted, 
        x='MaritalStatus', 
        y='Percent', 
        color='Attrition', 
        title='Attrition by Marital Status',
        color_discrete_map=color_map
    )
    fig_marital.update_layout(height=400)
    st.plotly_chart(fig_marital, use_container_width=True)

# Education Level Analysis
# Create crosstab and calculate percentages for education level
edu_crosstab = pd.crosstab(filtered_df['Education'], filtered_df['Attrition'])
edu_percent = edu_crosstab.div(edu_crosstab.sum(axis=1), axis=0) * 100
edu_melted = edu_percent.reset_index().melt(id_vars='Education', var_name='Attrition', value_name='Percent')

# Display education level attrition chart
with col4:
    fig_education = px.bar(
        edu_melted, 
        x='Education', 
        y='Percent', 
        color='Attrition', 
        title='Attrition by Education Level',
        color_discrete_map=color_map
    )
    fig_education.update_layout(height=400)
    st.plotly_chart(fig_education, use_container_width=True)

# Automatic Demographics Insights
st.markdown("#### Key Demographics Insights")

def generate_demographics_insights(df):
    """
    Generate automatic insights from demographic data analysis
    
    Args:
        df: Filtered dataframe containing employee data
    
    Returns:
        List of insight strings with demographic analysis
    """
    insights = []
    
    # Age Group Insights
    # Calculate attrition percentages by age group
    age_attrition = df.groupby(['AgeGroup', 'Attrition'], observed=False).size().reset_index(name='Count')
    age_total = age_attrition.groupby('AgeGroup', observed=False)['Count'].transform('sum')
    age_attrition['Percentage'] = age_attrition['Count'] / age_total * 100
    
    # Find age group with highest attrition rate
    yes_attrition = age_attrition[age_attrition['Attrition'] == 'Yes']
    if not yes_attrition.empty:
        highest_age_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmax()]
        insights.append(f"**Age Group Risk**: {highest_age_attrition['AgeGroup']} age group has the highest attrition rate at {highest_age_attrition['Percentage']:.1f}%")
    
    # Gender Insights
    # Calculate attrition percentages by gender
    gender_attrition = df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
    gender_total = gender_attrition.groupby('Gender')['Count'].transform('sum')
    gender_attrition['Percentage'] = gender_attrition['Count'] / gender_total * 100
    
    # Compare gender attrition rates
    gender_yes = gender_attrition[gender_attrition['Attrition'] == 'Yes']
    if len(gender_yes) > 1:
        gender_diff = abs(gender_yes.iloc[0]['Percentage'] - gender_yes.iloc[1]['Percentage'])
        if gender_diff > 5:  # Significant difference threshold
            higher_gender = gender_yes.loc[gender_yes['Percentage'].idxmax()]
            insights.append(f"**Gender Pattern**: {higher_gender['Gender']} employees show {gender_diff:.1f}% higher attrition rate")
        else:
            insights.append(f"**Gender Balance**: Attrition rates are relatively balanced across genders (difference < 5%)")
    
    # Marital Status Insights
    # Calculate attrition percentages by marital status
    marital_attrition = pd.crosstab(df['MaritalStatus'], df['Attrition'])
    marital_percent = marital_attrition.div(marital_attrition.sum(axis=1), axis=0) * 100
    
    # Find marital status with highest attrition
    if 'Yes' in marital_percent.columns:
        highest_marital = marital_percent['Yes'].idxmax()
        highest_marital_rate = marital_percent['Yes'].max()
        insights.append(f"**Marital Status Impact**: {highest_marital} employees have the highest attrition rate at {highest_marital_rate:.1f}%")
    
    # Education Level Insights
    # Calculate attrition percentages by education level
    edu_attrition = pd.crosstab(df['Education'], df['Attrition'])
    edu_percent = edu_attrition.div(edu_attrition.sum(axis=1), axis=0) * 100
    
    # Analyze education level patterns
    if 'Yes' in edu_percent.columns:
        # Map education levels to readable names
        edu_mapping = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        edu_percent_named = edu_percent.rename(index=edu_mapping)
        
        highest_edu = edu_percent_named['Yes'].idxmax()
        highest_edu_rate = edu_percent_named['Yes'].max()
        lowest_edu = edu_percent_named['Yes'].idxmin()
        lowest_edu_rate = edu_percent_named['Yes'].min()
        
        insights.append(f"**Education Pattern**: {highest_edu} level shows highest attrition ({highest_edu_rate:.1f}%), while {lowest_edu} shows lowest ({lowest_edu_rate:.1f}%)")
    
    # Overall demographic summary
    total_employees = len(df)
    total_attrition = len(df[df['Attrition'] == 'Yes'])
    attrition_rate = (total_attrition / total_employees) * 100
    
    insights.append(f"**Overall**: {total_attrition} out of {total_employees} employees left ({attrition_rate:.1f}% attrition rate)")
    
    return insights

# Generate and display demographic insights
demographic_insights = generate_demographics_insights(filtered_df)

# Display insights in a numbered format
for i, insight in enumerate(demographic_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Work Factors & Satisfaction
# Display the work factors section header
st.markdown('<div class="section-header"><h3><i class="bi bi-briefcase"></i> Work Factors & Satisfaction</h3></div>', unsafe_allow_html=True)

# Create 2 columns for work factor charts
col1, col2 = st.columns(2)

# Work-Life Balance Analysis
with col1:
    # Create violin plot to show work-life balance distribution
    fig_balance = px.violin(
        filtered_df, 
        x='Attrition', 
        y='WorkLifeBalance', 
        color='Attrition',
        title='Work-Life Balance Distribution',
        color_discrete_map=color_map
    )
    fig_balance.update_layout(height=400)
    st.plotly_chart(fig_balance, use_container_width=True)

# Overtime Analysis
with col2:
    # Create box plot for overtime days distribution (better for numerical data)
    fig_overtime = px.box(
        filtered_df,
        x='Attrition',
        y='OverTime',
        color='Attrition',
        title='Overtime Days Distribution by Attrition',
        color_discrete_map=color_map,
        points="outliers"  # Show outlier points
    )
    fig_overtime.update_layout(height=400)
    st.plotly_chart(fig_overtime, use_container_width=True)

# Create 2 columns for satisfaction charts
col3, col4 = st.columns(2)

# Job Satisfaction Analysis
with col3:
    # Create histogram for job satisfaction levels
    fig_satisfaction = px.histogram(
        filtered_df,
        x='JobSatisfaction',
        color='Attrition',
        title='Job Satisfaction Levels',
        color_discrete_map=color_map,
        barmode='group',
        nbins=4
    )
    fig_satisfaction.update_layout(height=400)
    st.plotly_chart(fig_satisfaction, use_container_width=True)

# Environment Satisfaction Analysis
with col4:
    # Create histogram for environment satisfaction
    fig_env = px.histogram(
        filtered_df,
        x='EnvironmentSatisfaction',
        color='Attrition',
        title='Environment Satisfaction',
        color_discrete_map=color_map,
        barmode='group',
        nbins=4
    )
    fig_env.update_layout(height=400)
    st.plotly_chart(fig_env, use_container_width=True)

# Automatic Work Factors & Satisfaction Insights
st.markdown("#### Work Environment Insights")

def generate_work_insights(df):
    """
    Generate automatic insights from work environment and satisfaction data
    
    Args:
        df: Filtered dataframe containing employee data
    
    Returns:
        List of insight strings with work environment analysis
    """
    insights = []
    
    # Work-Life Balance Analysis
    # Compare average work-life balance between attrited and retained employees
    wlb_yes = df[df['Attrition'] == 'Yes']['WorkLifeBalance'].mean()
    wlb_no = df[df['Attrition'] == 'No']['WorkLifeBalance'].mean()
    wlb_diff = wlb_no - wlb_yes
    
    # Categorize the difference and provide insights
    if wlb_diff > 0.3:
        insights.append(f"**Work-Life Balance**: Employees who left had {wlb_diff:.1f} points lower work-life balance (Average: {wlb_yes:.1f} vs {wlb_no:.1f})")
    elif wlb_diff > 0.1:
        insights.append(f"**Work-Life Balance**: Slight difference in work-life balance between leavers and stayers ({wlb_diff:.1f} points)")
    else:
        insights.append(f"**Work-Life Balance**: No significant difference in work-life balance scores")
    
    # Overtime Analysis
    # Compare average overtime between attrited and retained employees
    overtime_yes = df[df['Attrition'] == 'Yes']['OverTime'].mean()
    overtime_no = df[df['Attrition'] == 'No']['OverTime'].mean()
    overtime_diff = overtime_yes - overtime_no
    
    # Categorize overtime impact
    if overtime_diff > 10:
        insights.append(f"**Overtime Alert**: Employees who left worked {overtime_diff:.1f} more overtime days on average ({overtime_yes:.1f} vs {overtime_no:.1f})")
    elif overtime_diff > 5:
        insights.append(f"**Overtime Concern**: Moderate overtime difference between leavers and stayers ({overtime_diff:.1f} days)")
    else:
        insights.append(f"**Overtime Balance**: Similar overtime patterns across both groups")
    
    # Job Satisfaction Analysis
    # Compare average job satisfaction between groups
    job_sat_yes = df[df['Attrition'] == 'Yes']['JobSatisfaction'].mean()
    job_sat_no = df[df['Attrition'] == 'No']['JobSatisfaction'].mean()
    job_sat_diff = job_sat_no - job_sat_yes
    
    # Categorize job satisfaction impact
    if job_sat_diff > 0.5:
        insights.append(f"**Job Satisfaction**: Strong correlation with retention - leavers scored {job_sat_diff:.1f} points lower ({job_sat_yes:.1f} vs {job_sat_no:.1f})")
    elif job_sat_diff > 0.2:
        insights.append(f"**Job Satisfaction**: Moderate impact on retention ({job_sat_diff:.1f} points difference)")
    else:
        insights.append(f"**Job Satisfaction**: Similar satisfaction levels across groups")
    
    # Environment Satisfaction Analysis
    # Compare average environment satisfaction between groups
    env_sat_yes = df[df['Attrition'] == 'Yes']['EnvironmentSatisfaction'].mean()
    env_sat_no = df[df['Attrition'] == 'No']['EnvironmentSatisfaction'].mean()
    env_sat_diff = env_sat_no - env_sat_yes
    
    # Categorize environment satisfaction impact
    if env_sat_diff > 0.5:
        insights.append(f"**Environment Impact**: Work environment strongly affects retention - {env_sat_diff:.1f} points difference ({env_sat_yes:.1f} vs {env_sat_no:.1f})")
    elif env_sat_diff > 0.2:
        insights.append(f"**Environment Factor**: Work environment moderately impacts retention ({env_sat_diff:.1f} points)")
    else:
        insights.append(f"**Environment Neutral**: Environment satisfaction similar across groups")
    
    # Low satisfaction risk analysis
    # Calculate risk for employees with low job satisfaction
    low_job_sat = len(df[(df['JobSatisfaction'] <= 2) & (df['Attrition'] == 'Yes')])
    total_low_job_sat = len(df[df['JobSatisfaction'] <= 2])
    
    if total_low_job_sat > 0:
        low_sat_risk = (low_job_sat / total_low_job_sat) * 100
        insights.append(f"**Low Satisfaction Risk**: {low_sat_risk:.1f}% of employees with low job satisfaction (≤2) eventually left")
    
    return insights

# Generate and display work environment insights
work_insights = generate_work_insights(filtered_df)

# Display insights in a numbered format
for i, insight in enumerate(work_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Compensation & Performance
# Display the compensation section header
st.markdown('<div class="section-header"><h3><i class="bi bi-coin"></i> Compensation & Performance</h3></div>', unsafe_allow_html=True)

# Create 2 columns for compensation charts
col1, col2 = st.columns(2)

# Performance Rating Analysis
with col1:
    # Group performance ratings by attrition and calculate percentages
    performance_data = filtered_df.groupby(['PerformanceRating', 'Attrition']).size().reset_index(name='Count')
    performance_total = performance_data.groupby('PerformanceRating')['Count'].transform('sum')
    performance_data['Percentage'] = performance_data['Count'] / performance_total * 100

    # Create bar chart for performance vs attrition
    fig_performance = px.bar(
        performance_data,
        x='PerformanceRating',
        y='Percentage',
        color='Attrition',
        title='Attrition by Performance Rating',
        color_discrete_map=color_map
    )
    fig_performance.update_layout(height=400)
    st.plotly_chart(fig_performance, use_container_width=True)

# Salary Hike vs Income Analysis
with col2:
    # Create scatter plot to show relationship between salary hike and income
    fig_hike = px.scatter(
        filtered_df,
        x='PercentSalaryHike',
        y='MonthlyIncome',
        color='Attrition',
        title='Salary Hike vs Income',
        color_discrete_map=color_map,
        opacity=0.6,
        hover_data=['YearsAtCompany', 'JobSatisfaction']  # Additional info on hover
    )
    fig_hike.update_layout(height=400)
    st.plotly_chart(fig_hike, use_container_width=True)

# Monthly Income Distribution Analysis
# Create histogram for income distribution comparison
fig_income = px.histogram(
        filtered_df, 
        x='MonthlyIncome', 
        color='Attrition', 
        title='Monthly Income Distribution',
        color_discrete_map=color_map,
        nbins=30,
        opacity=0.7
)
fig_income.update_layout(height=400)
st.plotly_chart(fig_income, use_container_width=True)

# Automatic Compensation & Performance Insights
st.markdown("#### Compensation & Performance Insights")

def generate_compensation_insights(df):
    """
    Generate automatic insights from compensation and performance data
    
    Args:
        df: Filtered dataframe containing employee data
    
    Returns:
        List of insight strings with compensation and performance analysis
    """
    insights = []
    
    # Performance Rating Analysis
    # Calculate attrition percentages by performance rating
    perf_attrition = df.groupby(['PerformanceRating', 'Attrition']).size().reset_index(name='Count')
    perf_total = perf_attrition.groupby('PerformanceRating')['Count'].transform('sum')
    perf_attrition['Percentage'] = perf_attrition['Count'] / perf_total * 100
    
    # Find performance ratings with highest and lowest attrition
    yes_attrition = perf_attrition[perf_attrition['Attrition'] == 'Yes']
    if not yes_attrition.empty:
        highest_perf_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmax()]
        lowest_perf_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmin()]
        
        insights.append(f"**Performance Paradox**: Rating {highest_perf_attrition['PerformanceRating']} has highest attrition ({highest_perf_attrition['Percentage']:.1f}%), while rating {lowest_perf_attrition['PerformanceRating']} has lowest ({lowest_perf_attrition['Percentage']:.1f}%)")
    
    # Monthly Income Analysis
    # Compare average income between attrited and retained employees
    income_yes = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
    income_no = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
    income_diff = income_no - income_yes
    income_diff_percent = (income_diff / income_yes) * 100
    
    # Categorize income difference impact
    if income_diff > 5000:
        insights.append(f"**Salary Gap**: Employees who left earned ${income_diff:,.0f} less on average (${income_yes:,.0f} vs ${income_no:,.0f} - {income_diff_percent:.1f}% difference)")
    elif income_diff > 1000:
        insights.append(f"**Moderate Income Gap**: ${income_diff:,.0f} average income difference between leavers and stayers")
    else:
        insights.append(f"**Income Parity**: Similar income levels across both groups (${income_diff:,.0f} difference)")
    
    # Salary Hike Analysis
    # Compare average salary hikes between groups
    hike_yes = df[df['Attrition'] == 'Yes']['PercentSalaryHike'].mean()
    hike_no = df[df['Attrition'] == 'No']['PercentSalaryHike'].mean()
    hike_diff = hike_no - hike_yes
    
    # Categorize salary hike impact
    if hike_diff > 2:
        insights.append(f"**Salary Hike Impact**: Employees who stayed received {hike_diff:.1f}% higher salary increases ({hike_yes:.1f}% vs {hike_no:.1f}%)")
    elif hike_diff > 0.5:
        insights.append(f"**Modest Hike Difference**: {hike_diff:.1f}% difference in salary increases")
    else:
        insights.append(f"**Equal Opportunity**: Similar salary hike patterns across both groups")
    
    # Low income risk analysis
    # Calculate risk for employees with below-median income
    median_income = df['MonthlyIncome'].median()
    low_income_threshold = median_income * 0.8  # 80% of median income
    
    low_income_attrition = len(df[(df['MonthlyIncome'] < low_income_threshold) & (df['Attrition'] == 'Yes')])
    total_low_income = len(df[df['MonthlyIncome'] < low_income_threshold])
    
    if total_low_income > 0:
        low_income_risk = (low_income_attrition / total_low_income) * 100
        insights.append(f"**Low Income Risk**: {low_income_risk:.1f}% of employees earning below ${low_income_threshold:,.0f} eventually left")
    
    # High performer retention analysis
    # Analyze attrition among high-performing employees
    high_performers = df[df['PerformanceRating'] >= 4]  # Assuming rating 4+ is high performance
    if len(high_performers) > 0:
        high_perf_attrition = len(high_performers[high_performers['Attrition'] == 'Yes'])
        high_perf_rate = (high_perf_attrition / len(high_performers)) * 100
        
        # Categorize high performer retention
        if high_perf_rate > 15:
            insights.append(f"**High Performer Alert**: {high_perf_rate:.1f}% of high performers (rating 4+) left the company - critical talent retention issue")
        else:
            insights.append(f"**High Performer Retention**: Good retention of high performers - only {high_perf_rate:.1f}% attrition rate")
    
    # Compensation vs Performance correlation analysis
    # Calculate correlation between performance rating and income
    avg_income_by_rating = df.groupby('PerformanceRating')['MonthlyIncome'].mean()
    if len(avg_income_by_rating) > 1:
        # Calculate correlation coefficient
        performance_ratings = avg_income_by_rating.index.values
        income_values = avg_income_by_rating.values
        
        # Use numpy for correlation calculation
        import numpy as np
        income_correlation = np.corrcoef(performance_ratings, income_values)[0, 1]
        
        # Categorize correlation strength
        if income_correlation > 0.5:
            insights.append(f"**Pay-Performance Alignment**: Strong correlation ({income_correlation:.2f}) between performance rating and compensation")
        elif income_correlation > 0.2:
            insights.append(f"**Moderate Pay-Performance Link**: Moderate correlation ({income_correlation:.2f}) between performance and compensation")
        else:
            insights.append(f"**Pay-Performance Misalignment**: Weak correlation ({income_correlation:.2f}) between performance and compensation - review pay equity")
    
    return insights

# Generate and display compensation insights
compensation_insights = generate_compensation_insights(filtered_df)

# Display insights in a numbered format
for i, insight in enumerate(compensation_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Heatmap Analysis
st.markdown('<div class="section-header"><h3><i class="bi bi-building"></i> Department & Role Analysis</h3></div>', unsafe_allow_html=True)

# Create pivot table for heatmap
# Transform attrition data into a matrix format where rows are job roles and columns are departments
pivot = (
    filtered_df.assign(IsAttrited=filtered_df['Attrition'].eq('Yes').astype(int))  # Convert attrition to binary (1 for Yes, 0 for No)
    .groupby(['JobRole', 'Department'])['IsAttrited']  # Group by job role and department
    .mean()  # Calculate average attrition rate for each combination
    .mul(100)  # Convert to percentage
    .reset_index()
    .pivot(index='JobRole', columns='Department', values='IsAttrited')  # Reshape to matrix format
    .fillna(0)  # Fill empty cells with 0
)

# Create interactive heatmap visualization
fig_heat = px.imshow(
    pivot,
    text_auto='.1f',  # Show percentage values with 1 decimal place
    color_continuous_scale='RdYlBu_r',  # Red-Yellow-Blue color scale (reversed)
    aspect='auto',  # Auto-adjust aspect ratio
    title='Attrition Rate Heatmap: Job Role × Department (%)'
)
fig_heat.update_xaxes(side="top")  # Move x-axis labels to top
fig_heat.update_layout(height=600)  # Set chart height
fig_heat.update_traces(textfont=dict(size=12))  # Adjust text size

st.plotly_chart(fig_heat, use_container_width=True)

# Automatic Department & Role Analysis Insights
st.markdown("#### Department & Role Insights")

def generate_department_insights(df):
    """
    Generate automated insights about department and role attrition patterns
    
    Args:
        df: DataFrame containing employee data with Department, JobRole, and Attrition columns
    
    Returns:
        List of insight strings with key findings
    """
    insights = []
    
    # Calculate attrition rates by department
    dept_attrition = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
    dept_total = dept_attrition.groupby('Department')['Count'].transform('sum')
    dept_attrition['Percentage'] = dept_attrition['Count'] / dept_total * 100
    
    # Find departments with highest and lowest attrition
    dept_yes = dept_attrition[dept_attrition['Attrition'] == 'Yes']
    if not dept_yes.empty:
        highest_dept = dept_yes.loc[dept_yes['Percentage'].idxmax()]
        lowest_dept = dept_yes.loc[dept_yes['Percentage'].idxmin()]
        
        insights.append(f"**Department Risk**: {highest_dept['Department']} has highest attrition at {highest_dept['Percentage']:.1f}%, while {lowest_dept['Department']} has lowest at {lowest_dept['Percentage']:.1f}%")
    
    # Calculate attrition rates by job role
    role_attrition = df.groupby(['JobRole', 'Attrition']).size().reset_index(name='Count')
    role_total = role_attrition.groupby('JobRole')['Count'].transform('sum')
    role_attrition['Percentage'] = role_attrition['Count'] / role_total * 100
    
    # Find job roles with highest and lowest attrition
    role_yes = role_attrition[role_attrition['Attrition'] == 'Yes']
    if not role_yes.empty:
        highest_role = role_yes.loc[role_yes['Percentage'].idxmax()]
        lowest_role = role_yes.loc[role_yes['Percentage'].idxmin()]
        
        insights.append(f"**Role Risk**: {highest_role['JobRole']} shows highest attrition at {highest_role['Percentage']:.1f}%, while {lowest_role['JobRole']} shows lowest at {lowest_role['Percentage']:.1f}%")
    
    # Analyze department-role combinations from heatmap
    # Recreate pivot table to analyze specific combinations
    pivot = (
        df.assign(IsAttrited=df['Attrition'].eq('Yes').astype(int))
        .groupby(['JobRole', 'Department'])['IsAttrited']
        .mean()
        .mul(100)
        .reset_index()
        .pivot(index='JobRole', columns='Department', values='IsAttrited')
        .fillna(0)
    )
    
    # Find highest risk combinations
    max_combinations = []
    for col in pivot.columns:
        for idx in pivot.index:
            if pivot.loc[idx, col] > 0:  # Only consider combinations that exist
                max_combinations.append({
                    'JobRole': idx,
                    'Department': col,
                    'AttritionRate': pivot.loc[idx, col]
                })
    
    if max_combinations:
        # Sort by attrition rate (highest first)
        max_combinations.sort(key=lambda x: x['AttritionRate'], reverse=True)
        
        # Get top 3 highest risk combinations
        top_risk = max_combinations[:3]
        if top_risk:
            insights.append(f"**Highest Risk Combination**: {top_risk[0]['JobRole']} in {top_risk[0]['Department']} ({top_risk[0]['AttritionRate']:.1f}% attrition)")
        
        # Get combinations with 0% attrition (if any)
        zero_attrition = [combo for combo in max_combinations if combo['AttritionRate'] == 0]
        if zero_attrition:
            insights.append(f"**Perfect Retention**: {len(zero_attrition)} role-department combinations have 0% attrition")
    
    # Department size analysis
    dept_sizes = df['Department'].value_counts()
    largest_dept = dept_sizes.index[0]
    smallest_dept = dept_sizes.index[-1]
    
    # Check if largest department has high attrition
    largest_dept_attrition = dept_yes[dept_yes['Department'] == largest_dept]['Percentage'].iloc[0] if len(dept_yes[dept_yes['Department'] == largest_dept]) > 0 else 0
    
    insights.append(f"**Department Scale**: {largest_dept} is largest department ({dept_sizes[largest_dept]} employees) with {largest_dept_attrition:.1f}% attrition")
    
    # Role diversity analysis
    role_counts = df['JobRole'].value_counts()
    most_common_role = role_counts.index[0]
    most_common_role_attrition = role_yes[role_yes['JobRole'] == most_common_role]['Percentage'].iloc[0] if len(role_yes[role_yes['JobRole'] == most_common_role]) > 0 else 0
    
    insights.append(f"**Role Distribution**: {most_common_role} is most common role ({role_counts[most_common_role]} employees) with {most_common_role_attrition:.1f}% attrition")
    
    # Critical combinations analysis - flag high-risk scenarios
    critical_combinations = [combo for combo in max_combinations if combo['AttritionRate'] > 30]
    if critical_combinations:
        insights.append(f"**Critical Alert**: {len(critical_combinations)} role-department combinations have >30% attrition rate")
    
    return insights

# Generate and display insights
department_insights = generate_department_insights(filtered_df)

# Display insights in a nice format
for i, insight in enumerate(department_insights, 1):
    st.markdown(f"**{i}.** {insight}")

# Dataset Preview
st.markdown("---")
st.markdown('<div class="section-header"><h3><i class="bi bi-database-check"></i> Dataset Preview</h3></div>', unsafe_allow_html=True)

# Expandable filter box
with st.expander("Filter & Sort Options"):
    col1, col2 = st.columns(2)

    with col1:
        # Filter options - select column and value to filter by
        filter_col = st.selectbox("Filter by column", df.select_dtypes(include='object').columns, index=0)
        filter_val = st.selectbox("Value", df[filter_col].dropna().unique())

    with col2:
        # Sort options - select column and order
        sort_col = st.selectbox("Sort by column", df.columns, index=0)
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], horizontal=True)

    # Apply filter & sort
    filtered_df = df[df[filter_col] == filter_val]  # Filter data based on selected criteria
    sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))  # Sort filtered data

# Display the filtered and sorted dataframe
st.dataframe(sorted_df.iloc[:, 1:].head(100), use_container_width=True, height=200)
