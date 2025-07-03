import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Welcome - 3Sigma Squad", layout="wide")

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

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


# Header Section
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Welcome to 3Sigma Squad</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Hero Section
st.markdown('<div class="section-header"><h3>Advanced Analytics & Prediction Platform</h3></div>', unsafe_allow_html=True)
st.markdown("""
Welcome to the advanced analytics platform from 3Sigma Squad. Gain deep insights and accurate predictions to support your business decision-making.

**Key Features:**<br>
- **Interactive Dashboard** — Real-time data visualizations<br>
- **Prediction Engine** — Machine learning predictive models <br>
- **High Accuracy** — Optimized models for best performance
""", unsafe_allow_html=True)

# Statistics Section
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Predictions",
        value="1,234",
        delta="23 today"
    )

with col2:
    st.metric(
        label="Model Accuracy",
        value="94.2%",
        delta="2.1%"
    )

with col3:
    st.metric(
        label="Active Users",
        value="156",
        delta="12 this week"
    )

with col4:
    st.metric(
        label="Data Points",
        value="50.2K",
        delta="1.2K new"
    )

# Recent Activity Section
st.markdown("---")
st.markdown('<div class="section-header"><h3>Recent Activity</h3></div>', unsafe_allow_html=True)

# Sample data - replace with your actual data
recent_data = pd.DataFrame({
    'Time': ['10:30', '09:15', '08:45', '07:30'],
    'Activity': [
        'New prediction completed',
        'Dashboard updated',
        'Model retrained',
        'Data sync completed'
    ],
    'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success']
})

st.dataframe(recent_data, use_container_width=True)

# Quick Actions
st.markdown("---")
st.markdown('<div class="section-header"><h3>Quick Actions</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Open Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard.py")  # Adjust path as needed

with col2:
    if st.button("Make Prediction", use_container_width=True):
        st.switch_page("pages/prediction.py")  # Adjust path as needed

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>3Sigma Squad © 2024 | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)