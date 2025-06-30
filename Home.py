import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Welcome - 3Sigma Squad", layout="wide")

# Header Section
st.title("Welcome to 3Sigma Squad 👋")
st.markdown("---")

# Hero Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Advanced Analytics & Prediction Platform
    
    Selamat datang di platform analitik canggih dari **3Sigma Squad**. 
    Dapatkan insights mendalam dan prediksi akurat untuk mendukung pengambilan keputusan bisnis Anda.
    
    **Fitur Utama:**
    - 📊 **Dashboard Interaktif** - Visualisasi data real-time
    - 🔮 **Prediction Engine** - Model prediksi berbasis AI
    - 📈 **Advanced Analytics** - Analisis mendalam dengan berbagai metrik
    - 🎯 **Akurasi Tinggi** - Model yang telah dioptimasi untuk performa terbaik
    """)

with col2:
    st.info("""
    **Quick Navigation**
    
    Gunakan sidebar di kiri untuk:
    - 📊 **Dashboard** - Lihat overview data
    - 🔮 **Prediction** - Buat prediksi baru
    """)

# Statistics Section
st.markdown("---")
st.subheader("📈 Platform Statistics")

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
st.subheader("🕒 Recent Activity")

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
st.subheader("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Open Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard.py")  # Adjust path as needed

with col2:
    if st.button("🔮 Make Prediction", use_container_width=True):
        st.switch_page("pages/prediction.py")  # Adjust path as needed

with col3:
    if st.button("📈 View Analytics", use_container_width=True):
        st.info("Feature coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>3Sigma Squad © 2024 | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)