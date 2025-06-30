import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Employee Attrition Dashboard - 3Sigma Squad", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r'data\raw\attrition.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File 'attrition.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">📊 Employee Attrition Dashboard</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Real-time insights into employee retention and attrition patterns</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔧 Filters & Controls")
st.sidebar.markdown("---")

# Department filter
departments = ['All'] + list(df['Department'].unique())
selected_dept = st.sidebar.selectbox("📋 Department", departments)

# Job Role filter  
if selected_dept != 'All':
    job_roles = ['All'] + list(df[df['Department'] == selected_dept]['JobRole'].unique())
else:
    job_roles = ['All'] + list(df['JobRole'].unique())
selected_role = st.sidebar.selectbox("💼 Job Role", job_roles)

# Age range filter
age_range = st.sidebar.slider("👥 Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

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
st.markdown('<div class="section-header"><h2>📈 Executive Summary</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        label="🎯 Attrition Rate", 
        value=f"{attrition_rate:.1f}%",
        delta=f"{attrition_rate - 16.1:.1f}%" if attrition_rate != 16.1 else None
    )

with col2:
    st.metric(
        label="👥 Total Employees", 
        value=f"{total_employees:,}",
        delta=f"{total_employees - len(df)} filtered" if total_employees != len(df) else None
    )

with col3:
    st.metric(
        label="❌ Attrited", 
        value=f"{attrited_employees}",
        delta=f"{attrited_employees - 237}" if attrited_employees != 237 else None
    )

with col4:
    st.metric(
        label="⏰ Avg Tenure", 
        value=f"{average_tenure:.1f} yrs",
        delta=f"{average_tenure - 7.0:.1f}" if abs(average_tenure - 7.0) > 0.1 else None
    )

with col5:
    st.metric(
        label="😊 Job Satisfaction", 
        value=f"{average_job_satisfaction:.1f}/4",
        delta=f"{average_job_satisfaction - 2.7:.1f}" if abs(average_job_satisfaction - 2.7) > 0.1 else None
    )

with col6:
    st.metric(
        label="⭐ High Performers Lost", 
        value=f"{high_performers_attrited}",
        delta=f"-{high_performers_attrited}" if high_performers_attrited > 0 else "0"
    )

# Key Insights
if attrition_rate > 20:
    insight_color = "#ff4444"
    insight = "⚠️ **High Risk**: Attrition rate is above 20% - immediate attention required!"
elif attrition_rate > 15:
    insight_color = "#ff8800"
    insight = "⚡ **Moderate Risk**: Attrition rate is elevated - monitor closely"
else:
    insight_color = "#00aa00"
    insight = "✅ **Healthy**: Attrition rate is within acceptable range"

st.markdown(f"""
<div style="background: {insight_color}20; border: 1px solid {insight_color}; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
    {insight}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Demographics & Segmentation
st.markdown('<div class="section-header"><h2>👥 Demographics & Segmentation</h2></div>', unsafe_allow_html=True)

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
        title='🎂 Attrition Rate by Age Group',
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
        title='👫 Attrition by Gender',
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
        title='💑 Attrition by Marital Status',
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
        title='🎓 Attrition by Education Level',
        color_discrete_map=color_map
    )
    fig_education.update_layout(height=400)
    st.plotly_chart(fig_education, use_container_width=True)

st.markdown("---")

# Work Factors & Satisfaction
st.markdown('<div class="section-header"><h2>💼 Work Factors & Satisfaction</h2></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Work-Life Balance
    fig_balance = px.violin(
        filtered_df, 
        x='Attrition', 
        y='WorkLifeBalance', 
        color='Attrition',
        title='⚖️ Work-Life Balance Distribution',
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
        title='⏰ Overtime Days Distribution by Attrition',
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
        title='😊 Job Satisfaction Levels',
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
        title='🏢 Environment Satisfaction',
        color_discrete_map=color_map,
        barmode='group',
        nbins=4
    )
    fig_env.update_layout(height=400)
    st.plotly_chart(fig_env, use_container_width=True)

st.markdown("---")

# Compensation & Performance
st.markdown('<div class="section-header"><h2>💰 Compensation & Performance</h2></div>', unsafe_allow_html=True)

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
        title='⭐ Performance Rating vs Attrition',
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
        title='📈 Salary Hike vs Income',
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
        title='💵 Monthly Income Distribution',
        color_discrete_map=color_map,
        nbins=30,
        opacity=0.7
)
fig_income.update_layout(height=400)
st.plotly_chart(fig_income, use_container_width=True)

st.markdown("---")

# Heatmap Analysis
st.markdown('<div class="section-header"><h2>🔥 Department & Role Analysis</h2></div>', unsafe_allow_html=True)

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
    title='🎯 Attrition Rate Heatmap: Job Role × Department (%)'
)
fig_heat.update_xaxes(side="top")
fig_heat.update_layout(height=600)
fig_heat.update_traces(textfont=dict(size=12))

st.plotly_chart(fig_heat, use_container_width=True)

# Footer with additional insights
st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 2rem;">
    <h4>📊 Dashboard Insights</h4>
    <ul>
        <li><strong>High-Risk Factors:</strong> Overtime work, low work-life balance, and job satisfaction below 2.5</li>
        <li><strong>Retention Opportunity:</strong> Focus on employees with 1-3 years experience and high performers</li>
        <li><strong>Department Focus:</strong> Sales and HR departments show higher attrition rates</li>
        <li><strong>Demographic Trends:</strong> Younger employees (<25) and single employees are at higher risk</li>
    </ul>
    <p><small>📅 Last updated: {} | 🏢 3Sigma Squad Analytics</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)