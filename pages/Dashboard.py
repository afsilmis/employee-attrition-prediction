import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Employee Attrition Dashboard - 3Sigma Squad", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
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

# Header
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Employee Attrition Dashboard</h1>
    <p style="color: black; margin: 0; opacity: 0.9; text-align: center;">Real-time insights into employee retention and attrition patterns</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Load data
uploaded_file = st.file_uploader("Upload a CSV file (optional)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df = pd.read_csv('data/raw/attrition.csv')
st.markdown("---")


# Sidebar filters
st.sidebar.header("Filters & Controls")

# Department filter
departments = ['All'] + list(df['Department'].unique())
selected_dept = st.sidebar.selectbox("Department", departments)

# Job Role filter  
if selected_dept != 'All':
    job_roles = ['All'] + list(df[df['Department'] == selected_dept]['JobRole'].unique())
else:
    job_roles = ['All'] + list(df['JobRole'].unique())
selected_role = st.sidebar.selectbox("Job Role", job_roles)

# Age range filter
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

# Apply filters
filtered_df = df.copy()
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

# Color scheme
color_map = {
    'Yes': '#e74c3c',
    'No': 'lightgrey'
}

# Calculate metrics
total_employees = len(filtered_df)
attrited_employees = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
attrition_rate = (attrited_employees / total_employees * 100) if total_employees > 0 else 0
average_tenure = filtered_df['YearsAtCompany'].mean()
average_job_satisfaction = filtered_df['JobSatisfaction'].mean()
high_performers_attrited = len(filtered_df[(filtered_df['Attrition'] == 'Yes') & (filtered_df['PerformanceRating'] >= 3)])

# Executive Summary
st.markdown('<div class="section-header"><h3>Executive Summary</h3></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        label="Attrition Rate", 
        value=f"{attrition_rate:.1f}%",
        delta=f"{attrition_rate - 16.1:.1f}%" if attrition_rate != 16.1 else None
    )

with col2:
    st.metric(
        label="Total Employees", 
        value=f"{total_employees:,}",
        delta=f"{total_employees - len(df)} filtered" if total_employees != len(df) else None
    )

with col3:
    st.metric(
        label="Attrited", 
        value=f"{attrited_employees}",
        delta=f"{attrited_employees - 237}" if attrited_employees != 237 else None
    )

with col4:
    st.metric(
        label="Avg Tenure", 
        value=f"{average_tenure:.1f} yrs",
        delta=f"{average_tenure - 7.0:.1f}" if abs(average_tenure - 7.0) > 0.1 else None
    )

with col5:
    st.metric(
        label="Job Satisfaction", 
        value=f"{average_job_satisfaction:.1f}/4",
        delta=f"{average_job_satisfaction - 2.7:.1f}" if abs(average_job_satisfaction - 2.7) > 0.1 else None
    )

# Key Insights
if attrition_rate > 20:
    insight_color = "#ff4444"
    insight = "⚠️ High Risk: Attrition rate is above 20% - immediate attention required!"
elif attrition_rate > 15:
    insight_color = "#ff8800"
    insight = "⚡ Moderate Risk: Attrition rate is elevated - monitor closely"
else:
    insight_color = "#00aa00"
    insight = "✅ Healthy: Attrition rate is within acceptable range"

st.markdown(f"""
<div style="background: {insight_color}20; border: 1px solid {insight_color}; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
    {insight}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Demographics & Segmentation
st.markdown('<div class="section-header"><h3>Demographics & Segmentation</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Age Group Analysis
bins = [1, 25, 35, 45, 100]
labels = ['<25', '25-35', '35-45', '45+']
filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, right=False)

grouped = filtered_df.groupby(['AgeGroup', 'Attrition'], observed=False).size().reset_index(name='Count')
total_per_group = grouped.groupby('AgeGroup', observed=False)['Count'].transform('sum')
grouped['Percentage'] = grouped['Count'] / total_per_group * 100

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
gender_grouped = filtered_df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
gender_total = gender_grouped.groupby('Gender')['Count'].transform('sum')
gender_grouped['Percentage'] = gender_grouped['Count'] / gender_total * 100

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

col3, col4 = st.columns(2)

# Marital Status
marital_crosstab = pd.crosstab(filtered_df['MaritalStatus'], filtered_df['Attrition'])
marital_percent = marital_crosstab.div(marital_crosstab.sum(axis=1), axis=0) * 100
marital_melted = marital_percent.reset_index().melt(id_vars='MaritalStatus', var_name='Attrition', value_name='Percent')

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

# Education Level
edu_crosstab = pd.crosstab(filtered_df['Education'], filtered_df['Attrition'])
edu_percent = edu_crosstab.div(edu_crosstab.sum(axis=1), axis=0) * 100
edu_melted = edu_percent.reset_index().melt(id_vars='Education', var_name='Attrition', value_name='Percent')

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
    insights = []
    
    # Age Group Insights
    age_attrition = df.groupby(['AgeGroup', 'Attrition'], observed=False).size().reset_index(name='Count')
    age_total = age_attrition.groupby('AgeGroup', observed=False)['Count'].transform('sum')
    age_attrition['Percentage'] = age_attrition['Count'] / age_total * 100
    
    # Find highest attrition age group
    yes_attrition = age_attrition[age_attrition['Attrition'] == 'Yes']
    if not yes_attrition.empty:
        highest_age_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmax()]
        insights.append(f"**Age Group Risk**: {highest_age_attrition['AgeGroup']} age group has the highest attrition rate at {highest_age_attrition['Percentage']:.1f}%")
    
    # Gender Insights
    gender_attrition = df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
    gender_total = gender_attrition.groupby('Gender')['Count'].transform('sum')
    gender_attrition['Percentage'] = gender_attrition['Count'] / gender_total * 100
    
    gender_yes = gender_attrition[gender_attrition['Attrition'] == 'Yes']
    if len(gender_yes) > 1:
        gender_diff = abs(gender_yes.iloc[0]['Percentage'] - gender_yes.iloc[1]['Percentage'])
        if gender_diff > 5:  # Significant difference
            higher_gender = gender_yes.loc[gender_yes['Percentage'].idxmax()]
            insights.append(f"**Gender Pattern**: {higher_gender['Gender']} employees show {gender_diff:.1f}% higher attrition rate")
        else:
            insights.append(f"**Gender Balance**: Attrition rates are relatively balanced across genders (difference < 5%)")
    
    # Marital Status Insights
    marital_attrition = pd.crosstab(df['MaritalStatus'], df['Attrition'])
    marital_percent = marital_attrition.div(marital_attrition.sum(axis=1), axis=0) * 100
    
    if 'Yes' in marital_percent.columns:
        highest_marital = marital_percent['Yes'].idxmax()
        highest_marital_rate = marital_percent['Yes'].max()
        insights.append(f"**Marital Status Impact**: {highest_marital} employees have the highest attrition rate at {highest_marital_rate:.1f}%")
    
    # Education Level Insights
    edu_attrition = pd.crosstab(df['Education'], df['Attrition'])
    edu_percent = edu_attrition.div(edu_attrition.sum(axis=1), axis=0) * 100
    
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

# Generate and display insights
demographic_insights = generate_demographics_insights(filtered_df)

# Display insights in a nice format
for i, insight in enumerate(demographic_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Work Factors & Satisfaction
st.markdown('<div class="section-header"><h3>Work Factors & Satisfaction</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Work-Life Balance
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

with col2:
    # Overtime Days Analysis - Box plot lebih cocok untuk data numerik
    fig_overtime = px.box(
        filtered_df,
        x='Attrition',
        y='OvertimeDays',
        color='Attrition',
        title='Overtime Days Distribution by Attrition',
        color_discrete_map=color_map,
        points="outliers"  # Show outlier points
    )
    fig_overtime.update_layout(height=400)
    st.plotly_chart(fig_overtime, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Job Satisfaction
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

with col4:
    # Environment Satisfaction
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
    insights = []
    
    # Work-Life Balance Analysis
    wlb_yes = df[df['Attrition'] == 'Yes']['WorkLifeBalance'].mean()
    wlb_no = df[df['Attrition'] == 'No']['WorkLifeBalance'].mean()
    wlb_diff = wlb_no - wlb_yes
    
    if wlb_diff > 0.3:
        insights.append(f"**Work-Life Balance**: Employees who left had {wlb_diff:.1f} points lower work-life balance (Average: {wlb_yes:.1f} vs {wlb_no:.1f})")
    elif wlb_diff > 0.1:
        insights.append(f"**Work-Life Balance**: Slight difference in work-life balance between leavers and stayers ({wlb_diff:.1f} points)")
    else:
        insights.append(f"**Work-Life Balance**: No significant difference in work-life balance scores")
    
    # Overtime Analysis
    overtime_yes = df[df['Attrition'] == 'Yes']['OvertimeDays'].mean()
    overtime_no = df[df['Attrition'] == 'No']['OvertimeDays'].mean()
    overtime_diff = overtime_yes - overtime_no
    
    if overtime_diff > 10:
        insights.append(f"**Overtime Alert**: Employees who left worked {overtime_diff:.1f} more overtime days on average ({overtime_yes:.1f} vs {overtime_no:.1f})")
    elif overtime_diff > 5:
        insights.append(f"**Overtime Concern**: Moderate overtime difference between leavers and stayers ({overtime_diff:.1f} days)")
    else:
        insights.append(f"**Overtime Balance**: Similar overtime patterns across both groups")
    
    # Job Satisfaction Analysis
    job_sat_yes = df[df['Attrition'] == 'Yes']['JobSatisfaction'].mean()
    job_sat_no = df[df['Attrition'] == 'No']['JobSatisfaction'].mean()
    job_sat_diff = job_sat_no - job_sat_yes
    
    if job_sat_diff > 0.5:
        insights.append(f"**Job Satisfaction**: Strong correlation with retention - leavers scored {job_sat_diff:.1f} points lower ({job_sat_yes:.1f} vs {job_sat_no:.1f})")
    elif job_sat_diff > 0.2:
        insights.append(f"**Job Satisfaction**: Moderate impact on retention ({job_sat_diff:.1f} points difference)")
    else:
        insights.append(f"**Job Satisfaction**: Similar satisfaction levels across groups")
    
    # Environment Satisfaction Analysis
    env_sat_yes = df[df['Attrition'] == 'Yes']['EnvironmentSatisfaction'].mean()
    env_sat_no = df[df['Attrition'] == 'No']['EnvironmentSatisfaction'].mean()
    env_sat_diff = env_sat_no - env_sat_yes
    
    if env_sat_diff > 0.5:
        insights.append(f"**Environment Impact**: Work environment strongly affects retention - {env_sat_diff:.1f} points difference ({env_sat_yes:.1f} vs {env_sat_no:.1f})")
    elif env_sat_diff > 0.2:
        insights.append(f"**Environment Factor**: Work environment moderately impacts retention ({env_sat_diff:.1f} points)")
    else:
        insights.append(f"**Environment Neutral**: Environment satisfaction similar across groups")
    
    # Low satisfaction risk analysis
    low_job_sat = len(df[(df['JobSatisfaction'] <= 2) & (df['Attrition'] == 'Yes')])
    total_low_job_sat = len(df[df['JobSatisfaction'] <= 2])
    
    if total_low_job_sat > 0:
        low_sat_risk = (low_job_sat / total_low_job_sat) * 100
        insights.append(f"**Low Satisfaction Risk**: {low_sat_risk:.1f}% of employees with low job satisfaction (≤2) eventually left")
    
    return insights

# Generate and display insights
work_insights = generate_work_insights(filtered_df)

# Display insights in a nice format
for i, insight in enumerate(work_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Compensation & Performance
st.markdown('<div class="section-header"><h3>Compensation & Performance</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Performance Rating Analysis
    performance_data = filtered_df.groupby(['PerformanceRating', 'Attrition']).size().reset_index(name='Count')
    performance_total = performance_data.groupby('PerformanceRating')['Count'].transform('sum')
    performance_data['Percentage'] = performance_data['Count'] / performance_total * 100

    fig_performance = px.bar(
        performance_data,
        x='PerformanceRating',
        y='Percentage',
        color='Attrition',
        title='Performance Rating vs Attrition',
        color_discrete_map=color_map
    )
    fig_performance.update_layout(height=400)
    st.plotly_chart(fig_performance, use_container_width=True)

with col2:
    # Salary Hike vs Attrition
    fig_hike = px.scatter(
        filtered_df,
        x='PercentSalaryHike',
        y='MonthlyIncome',
        color='Attrition',
        title='Salary Hike vs Income',
        color_discrete_map=color_map,
        opacity=0.6,
        hover_data=['YearsAtCompany', 'JobSatisfaction']
    )
    fig_hike.update_layout(height=400)
    st.plotly_chart(fig_hike, use_container_width=True)

# Monthly Income Distribution
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
    insights = []
    
    # Performance Rating Analysis
    perf_attrition = df.groupby(['PerformanceRating', 'Attrition']).size().reset_index(name='Count')
    perf_total = perf_attrition.groupby('PerformanceRating')['Count'].transform('sum')
    perf_attrition['Percentage'] = perf_attrition['Count'] / perf_total * 100
    
    # Find performance rating with highest attrition
    yes_attrition = perf_attrition[perf_attrition['Attrition'] == 'Yes']
    if not yes_attrition.empty:
        highest_perf_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmax()]
        lowest_perf_attrition = yes_attrition.loc[yes_attrition['Percentage'].idxmin()]
        
        insights.append(f"**Performance Paradox**: Rating {highest_perf_attrition['PerformanceRating']} has highest attrition ({highest_perf_attrition['Percentage']:.1f}%), while rating {lowest_perf_attrition['PerformanceRating']} has lowest ({lowest_perf_attrition['Percentage']:.1f}%)")
    
    # Monthly Income Analysis
    income_yes = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
    income_no = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
    income_diff = income_no - income_yes
    income_diff_percent = (income_diff / income_yes) * 100
    
    if income_diff > 5000:
        insights.append(f"**Salary Gap**: Employees who left earned ${income_diff:,.0f} less on average (${income_yes:,.0f} vs ${income_no:,.0f} - {income_diff_percent:.1f}% difference)")
    elif income_diff > 1000:
        insights.append(f"**Moderate Income Gap**: ${income_diff:,.0f} average income difference between leavers and stayers")
    else:
        insights.append(f"**Income Parity**: Similar income levels across both groups (${income_diff:,.0f} difference)")
    
    # Salary Hike Analysis
    hike_yes = df[df['Attrition'] == 'Yes']['PercentSalaryHike'].mean()
    hike_no = df[df['Attrition'] == 'No']['PercentSalaryHike'].mean()
    hike_diff = hike_no - hike_yes
    
    if hike_diff > 2:
        insights.append(f"**Salary Hike Impact**: Employees who stayed received {hike_diff:.1f}% higher salary increases ({hike_yes:.1f}% vs {hike_no:.1f}%)")
    elif hike_diff > 0.5:
        insights.append(f"**Modest Hike Difference**: {hike_diff:.1f}% difference in salary increases")
    else:
        insights.append(f"**Equal Opportunity**: Similar salary hike patterns across both groups")
    
    # Low income risk analysis
    median_income = df['MonthlyIncome'].median()
    low_income_threshold = median_income * 0.8  # 80% of median
    
    low_income_attrition = len(df[(df['MonthlyIncome'] < low_income_threshold) & (df['Attrition'] == 'Yes')])
    total_low_income = len(df[df['MonthlyIncome'] < low_income_threshold])
    
    if total_low_income > 0:
        low_income_risk = (low_income_attrition / total_low_income) * 100
        insights.append(f"**Low Income Risk**: {low_income_risk:.1f}% of employees earning below ${low_income_threshold:,.0f} eventually left")
    
    # High performer retention analysis
    high_performers = df[df['PerformanceRating'] >= 4]  # Assuming 4+ is high performance
    if len(high_performers) > 0:
        high_perf_attrition = len(high_performers[high_performers['Attrition'] == 'Yes'])
        high_perf_rate = (high_perf_attrition / len(high_performers)) * 100
        
        if high_perf_rate > 15:
            insights.append(f"**High Performer Alert**: {high_perf_rate:.1f}% of high performers (rating 4+) left the company - critical talent retention issue")
        else:
            insights.append(f"**High Performer Retention**: Good retention of high performers - only {high_perf_rate:.1f}% attrition rate")
    
    # Compensation vs Performance correlation
    avg_income_by_rating = df.groupby('PerformanceRating')['MonthlyIncome'].mean()
    if len(avg_income_by_rating) > 1:
        # Calculate correlation between performance rating and average income
        performance_ratings = avg_income_by_rating.index.values
        income_values = avg_income_by_rating.values
        
        # Use numpy correlation
        import numpy as np
        income_correlation = np.corrcoef(performance_ratings, income_values)[0, 1]
        
        if income_correlation > 0.5:
            insights.append(f"**Pay-Performance Alignment**: Strong correlation ({income_correlation:.2f}) between performance rating and compensation")
        elif income_correlation > 0.2:
            insights.append(f"**Moderate Pay-Performance Link**: Moderate correlation ({income_correlation:.2f}) between performance and compensation")
        else:
            insights.append(f"**Pay-Performance Misalignment**: Weak correlation ({income_correlation:.2f}) between performance and compensation - review pay equity")
    
    return insights

# Generate and display insights
compensation_insights = generate_compensation_insights(filtered_df)

# Display insights in a nice format
for i, insight in enumerate(compensation_insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.markdown("---")

# Heatmap Analysis
st.markdown('<div class="section-header"><h3>Department & Role Analysis</h3></div>', unsafe_allow_html=True)

# Create pivot table for heatmap
pivot = (
    filtered_df.assign(IsAttrited=filtered_df['Attrition'].eq('Yes').astype(int))
    .groupby(['JobRole', 'Department'])['IsAttrited']
    .mean()
    .mul(100)
    .reset_index()
    .pivot(index='JobRole', columns='Department', values='IsAttrited')
    .fillna(0)
)

fig_heat = px.imshow(
    pivot,
    text_auto='.1f',
    color_continuous_scale='RdYlBu_r',
    aspect='auto',
    title='Attrition Rate Heatmap: Job Role × Department (%)'
)
fig_heat.update_xaxes(side="top")
fig_heat.update_layout(height=600)
fig_heat.update_traces(textfont=dict(size=12))

st.plotly_chart(fig_heat, use_container_width=True)

# Automatic Department & Role Analysis Insights
st.markdown("#### Department & Role Insights")

def generate_department_insights(df):
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
        # Sort by attrition rate
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
    
    # Critical combinations analysis
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
st.markdown('<div class="section-header"><h3>Dataset Preview</h3></div>', unsafe_allow_html=True)

# Expandable filter box
with st.expander("Filter & Sort Options"):
    col1, col2 = st.columns(2)

    with col1:
        filter_col = st.selectbox("Filter by column", df.select_dtypes(include='object').columns, index=0)
        filter_val = st.selectbox("Value", df[filter_col].dropna().unique())

    with col2:
        sort_col = st.selectbox("Sort by column", df.columns, index=0)
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], horizontal=True)

    # Apply filter & sort
    filtered_df = df[df[filter_col] == filter_val]
    sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))

st.dataframe(sorted_df.iloc[:, 1:].head(100), use_container_width=True, height=200)

# Footer with additional insights
if uploaded_file is not None:
    try:
        # Read uploaded CSV
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)

        if df.empty or df.columns.size == 0:
            st.error("The uploaded file is empty or contains no columns.")
        else:
            if 'Attrition' in df.columns:
                # Retention Opportunity
                if 'YearsAtCompany' in df.columns and 'PerformanceRating' in df.columns:
                    mask_ret = df['YearsAtCompany'].between(1, 3) & (df['PerformanceRating'] >= 3)
                    count_ret = df[mask_ret].shape[0]
                    retention_text = f"Focus on {count_ret} employees with 1–3 years of experience and strong performance."
                else:
                    retention_text = "Required columns 'YearsAtCompany' and/or 'PerformanceRating' are missing."

                # Department Focus
                if 'Department' in df.columns:
                    dept_risk = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean())
                    top_depts = dept_risk.sort_values(ascending=False).head(2).index.tolist()
                    dept_focus_text = f"Higher attrition rates observed in: {' and '.join(top_depts)} departments."
                else:
                    dept_focus_text = "Column 'Department' is missing."

                # Demographic Trends
                age_risk = df[(df['Age'] < 25) & (df['Attrition'] == 'Yes')].shape[0] if 'Age' in df.columns else 0
                single_risk = df[(df['MaritalStatus'] == 'Single') & (df['Attrition'] == 'Yes')].shape[0] if 'MaritalStatus' in df.columns else 0
                demographic_text = f"{age_risk} younger employees (<25) and {single_risk} single employees are at higher risk."

                # Display footer insight
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 2rem;">
                    <h4><i class="fa-solid fa-square-poll-vertical"></i> Dashboard Insights</h4>
                    <ul>
                        <li><strong>Retention Opportunity:</strong> {retention_text}</li>
                        <li><strong>Department Focus:</strong> {dept_focus_text}</li>
                        <li><strong>Demographic Trends:</strong> {demographic_text}</li>
                    </ul>
                    <p><small><i class="fa-solid fa-calendar-day"></i>  Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</small></p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to read file: {e}")
