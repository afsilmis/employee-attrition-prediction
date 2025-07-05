import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from catboost import CatBoostClassifier
import io
import plotly.express as px
import random

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# Load models dan artifacts
model = CatBoostClassifier()
model.load_model("models/catboost_simplified_model.cbm")
transformer = joblib.load('models/transformer.pkl')
ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')

# Parameter preprocessing
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

threshold = 0.73

# Header
st.markdown("""
<div>
    <h1 style="color: black; margin: 0; text-align: center;">Batch Resignation Prediction</h1>
    <p style="color: black; margin: 0; opacity: 0.9; text-align: center;">Bulk Attrition Forecast — Analyze Resignation Risk for Multiple Employees</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

import pandas as pd
import streamlit as st

uploaded_file = st.file_uploader("Upload file", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write(df)

# === Show Dummy Template ===
st.markdown("### <i class='fa-solid fa-file-excel'></i> Excel Template", unsafe_allow_html=True)

num_rows = 3
dummy_data = {}

for col in selected_num_cols:
    median = median_dict.get(col, 0)
    dummy_data[col] = [median + i for i in range(num_rows)]

for col in selected_ordinal_cols:
    modus = modus_dict.get(col, 1)
    dummy_data[col] = [modus + i for i in range(num_rows)]

for col in selected_nominal_cols:
    options = nominal_options.get(col, [''])
    row_values = []
    for i in range(num_rows):
        if len(options) > i:
            row_values.append(options[i])
        else:
            row_values.append(random.choice(options))
    dummy_data[col] = row_values

df_dummy = pd.DataFrame(dummy_data)
st.dataframe(df_dummy)

# Tombol download template
output_dummy = io.BytesIO()
with pd.ExcelWriter(output_dummy, engine='xlsxwriter') as writer:
    df_dummy.to_excel(writer, index=False, sheet_name='Template')

dummy_excel = output_dummy.getvalue()

st.download_button(
    label="Download Example Template",
    data=dummy_excel,
    file_name='example_template.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

if uploaded_file is not None:
    # Tentukan jenis file berdasarkan ekstensi
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

    st.markdown("### <i class='fa-solid fa-table'></i> Data Preview", unsafe_allow_html=True)
    st.dataframe(df_input.head())


    # === Preprocessing ===
    df = df_input.copy()

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

    # Predict probabilities
    probabilities = model.predict_proba(X_all)
    proba_resign = probabilities[:, 1]
    predictions = (proba_resign >= threshold).astype(int)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Buat DataFrame hasil
    result_df = df_input.copy()
    result_df['probability_resign'] = proba_resign
    result_df['prediction'] = predictions

    # Tambah top 3 features per baris
    top_features = []
    for i in range(len(df)):
        shap_df = pd.DataFrame({
            'feature': selected_num_cols + selected_ordinal_cols + expected_ohe_columns,
            'shap_value': shap_values_pos[i].flatten()
        })
        top3 = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(3)
        top_features.append(', '.join(top3['feature'].tolist()))
    result_df['top_3_features'] = top_features
  
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Create result DataFrame
    result_df = df_input.copy()
    result_df['probability_resign'] = proba_resign
    result_df['prediction'] = predictions
    
    # Add top 3 features per row
    feature_names = selected_num_cols + selected_ordinal_cols + expected_ohe_columns
    top_features = []
    for i in range(len(df)):
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_pos[i].flatten()
        })
        top3 = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(3)
        top_features.append(', '.join(top3['feature'].tolist()))
    
    result_df['top_3_features'] = top_features
    
    # === VISUALIZATION SECTION ===
    st.write("---")
    st.markdown("### <i class='fa-solid fa-chart-simple'></i> Prediction Analytics", unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Overview", "Feature Insights", "Results"])
    
    with tab1:
        st.write("### Prediction Summary")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        total_employees = len(result_df)
        resign_count = sum(result_df['prediction'])
        stay_count = total_employees - resign_count
        avg_probability = result_df['probability_resign'].mean()
        
        with col1:
            st.metric("Total Employees", total_employees)
        with col2:
            st.metric("Predicted to Resign", resign_count, 
                     delta=f"{resign_count/total_employees*100:.1f}%")
        with col3:
            st.metric("Predicted to Stay", stay_count,
                     delta=f"{stay_count/total_employees*100:.1f}%")
        with col4:
            st.metric("Avg. Resign Probability", f"{avg_probability:.2f}",
                     delta=f"{(avg_probability-0.5)*100:+.1f}% vs baseline")
        
        # Visualization row
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for prediction distribution
            pred_counts = result_df['prediction'].value_counts()
            fig_pie = px.pie(
                values=pred_counts.values,
                names=['Stay', 'Resign'],
                title="Prediction Distribution",
                color_discrete_map={'Stay': '#2E8B57', 'Resign': '#DC143C'}
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Histogram of probabilities
            fig_hist = px.histogram(
                result_df, 
                x='probability_resign',
                nbins=20,
                title="Distribution of Resignation Probabilities",
                labels={'probability_resign': 'Probability of Resignation', 'count': 'Number of Employees'},
                color_discrete_sequence=['#4CAF50']
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                              annotation_text=f"Threshold: {threshold}")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:        
        # Global feature importance
        feature_importance = np.abs(shap_values_pos).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top 10 features bar chart
            top_10_features = importance_df.head(10)
            fig_feat = px.bar(
                top_10_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features (Global)",
                labels={'importance': 'Average SHAP Value', 'feature': 'Features'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig_feat.update_layout(height=500)
            st.plotly_chart(fig_feat, use_container_width=True)
        
        with col2:
            text = "##### Feature Ranking\n"
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                text += f"{i+1}. {row['feature']}\n"
            st.markdown(text)
        
        # Most common contributing factors
        all_features = []
        for features_str in result_df['top_3_features']:
            if features_str:  # Check if not empty
                all_features.extend([f.strip() for f in features_str.split(',')])
        
        if all_features:
            feature_freq = pd.Series(all_features).value_counts().head(10)
            fig_freq = px.bar(
                x=feature_freq.values,
                y=feature_freq.index,
                orientation='h',
                title="Most Frequently Contributing Features",
                labels={'x': 'Frequency', 'y': 'Features'},
                color=feature_freq.values,
                color_continuous_scale='Blues'
            )
            fig_freq.update_layout(height=400)
            st.plotly_chart(fig_freq, use_container_width=True)
    
    with tab3:
        st.write("### Complete Prediction Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_filter = st.selectbox(
                "Filter by Prediction",
                ["All", "Resign", "Stay"],
                index=0
            )
        
        with col3:
            prob_threshold = st.slider(
                "Min Probability Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        # Apply filters
        filtered_df = result_df.copy()
                
        if prediction_filter != "All":
            pred_value = 1 if prediction_filter == "Resign" else 0
            filtered_df = filtered_df[filtered_df['prediction'] == pred_value]
        
        filtered_df = filtered_df[filtered_df['probability_resign'] >= prob_threshold]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_df)} of {len(result_df)} employees")
        
        # Reorder columns for better display
        display_cols = ['probability_resign', 'prediction', 'top_3_features']
        other_cols = [col for col in filtered_df.columns if col not in display_cols]
        final_cols = other_cols + display_cols
        
        st.dataframe(
            filtered_df[final_cols],
            use_container_width=True,
            column_config={
                'probability_resign': st.column_config.NumberColumn(
                    'Resign Probability',
                    format="%.3f"
                ),
                'prediction': st.column_config.TextColumn(
                    'Prediction',
                    help="0 = Stay, 1 = Resign"
                ),
                'top_3_features': st.column_config.TextColumn(
                    'Top Contributing Factors'
                )
            }
        )
        
        # Download section
        st.write("#### Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Main results
                result_df.to_excel(writer, index=False, sheet_name='Predictions')
                
                # Summary statistics
                summary_stats = pd.DataFrame({
                    'Metric': ['Total Employees', 'Predicted to Resign', 'Predicted to Stay', 
                              'Average Resign Probability', 'High Risk (≥70%)', 'Very High Risk (≥80%)'],
                    'Value': [total_employees, resign_count, stay_count, 
                             f"{avg_probability:.3f}", 
                             len(result_df[result_df['probability_resign'] >= 0.7]),
                             len(result_df[result_df['probability_resign'] >= 0.8])]
                })
                summary_stats.to_excel(writer, index=False, sheet_name='Summary')
                
            processed_data = output.getvalue()
            
            st.download_button(
                label="Download Complete Results (Excel)",
                data=processed_data,
                file_name=f'employee_resignation_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        with col2:
            filtered_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Results (Excel)",
                data=filtered_data,
                file_name=f'filtered_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )