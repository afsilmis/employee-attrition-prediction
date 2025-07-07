# **Employee Attrition Prediction**

This project aims to build a machine learning model that can predict the likelihood of an employee leaving (attrition). The case study is a fictional company named XYZ, which is experiencing an attrition rate of approximately 15% per year. This high attrition rate is considered detrimental to the company as it causes project delays, high recruitment and training costs, and a decrease in operational efficiency.

To address this issue, the company has contracted a data science team to identify the factors causing attrition and to build a predictive model to serve as a basis for strategic decision-making by the HR team.

### **Project Stages (Stage 0–4)**

**Stage 0 – Project Initiation & Problem Framing**

This initial stage focuses on defining the business objectives and the scope of the project. The main goals are to predict employee attrition and identify the most influential factors. The management and HR teams of XYZ are the primary stakeholders in this project.

**Stage 1 – Data Acquisition & Preparation**

In this stage, we focus on collecting and preparing the data that is crucial for modeling.

* **Data Acquisition and Merging:** The first step involves loading various relevant datasets, including employee information and time-tracking data. These datasets are then merged to form a single, comprehensive data source.
* **Feature Engineering:** From the available data, we create new features that can provide additional insights. This includes information derived from time-tracking data, such as employee attendance patterns and working hours.
* **Preliminary Data Review and Preprocessing:** After merging and feature engineering, we conduct a general review of the data. This includes examining data characteristics, such as data types and basic statistics. We also remove irrelevant or redundant columns that will not be used in the modeling process.
* **In-depth Exploratory Data Analysis (EDA):** EDA is a critical part of understanding the data's characteristics. This process includes:
    * Identifying features that have a strong relationship with attrition.
    * Assessing the predictive potential of each feature.
    * Visualizing variable distributions using histograms, grouped bar charts, and stacked bar charts to understand patterns in the data.
    * Exploring correlations and relationships between variables to uncover additional insights.
* **Data Preprocessing:** This stage ensures the data is ready for modeling:
    * **Data Splitting:** The data is divided into training and testing sets with stratification to maintain the proportion of the target class.
    * **Data Quality Handling:** This involves handling duplicate data and missing values using appropriate imputation strategies.
    * **Feature Transformation:** Categorical features are converted into numerical representations through encoding. Additionally, certain numerical features may undergo transformation to reduce skewness and are standardized.
    * **Outlier Handling:** We detect and handle outliers in the data to ensure model stability.
    * **Handling Class Imbalance:** To ensure the model can learn effectively from the minority class, we apply techniques to address the class imbalance in the target variable within the training data.
* **Storing the Training and Testing Datasets:** After all preparation processes, the processed training and testing data are saved for use in the subsequent modeling stage.

**Stage 2 – Model Development & Experimentation**

This stage focuses on systematic experimentation to build, evaluate, and select the best-performing model. The process is broken down into several key steps:

* **Initial Model Selection and Training:**
    * First, a baseline model, such as Logistic Regression, is trained to serve as a performance benchmark.
    * Next, we experiment with a variety of model types to find the best candidates, grouped as follows: Linear Models, Kernel-based Models, Instance-based Models, Tree-based Models, Boosting Models, and Neural Networks.
    * The initial performance of each model on the validation data is measured using metrics like F2 Score, Precision, and Recall. 
* **Hyperparameter Tuning:** The six best-performing models from the previous stage are selected for further optimization. The tuning process is conducted systematically using Grid Search with stratified k-fold cross-validation to find the optimal combination of parameters that yields the highest F2 Score on the validation data.
* **Final Model Evaluation and Selection:** A single final model is chosen based on the best F2-Score performance after tuning, stability during cross-validation, and other relevant metrics. This model is then retrained using the entire training dataset with the optimal hyperparameters. A comprehensive final evaluation is performed on the test set. This includes an analysis of the Confusion Matrix, Specificity (TNR), and the Precision-Recall curve to ensure the model's performance is solid and reliable. Learning curves are also analyzed to ensure the model is not overfitting.

**Stage 3 – Model Evaluation & Interpretability**

This stage evaluates the model's performance, fairness, and interpretability to ensure it is both effective and trustworthy for decision-making.

* **Performance Evaluation:**
  The final CatBoost model is assessed using Accuracy, Precision, Recall, Specificity, F1-Score, F2-Score, and Average Precision (AP). Confusion matrix and precision-recall curves are analyzed. Threshold tuning (from 0.5 to 0.73) improves F2-score and eliminates false positives, enhancing the model’s recall focus.

* **Fairness Analysis:**
  The model is tested for bias across protected groups (e.g., AgeGroup, MaritalStatus) and categorical attributes (e.g., Department, JobRole). Metrics like recall and precision are compared across groups. A fairness heatmap summarizes disparities, and increasing representation is recommended.

* **Model Explainability (SHAP):**
  SHAP is used to interpret the model:
  * Global Importance
  * Feature Removal
  * Impact Direction
  * Local Explanation
  * Dependence Plots

* **Statistical Validation:**
  T-tests confirm that key features (e.g., Age, Manager Tenure) significantly differ between stayers and leavers, supporting their predictive relevance.

This stage confirms the model is accurate, interpretable, and fair—ready to support high-stakes HR decisions.

**Stage 4 – Deployment & Business Integration**

This final stage focuses on operationalizing the attrition prediction model and ensuring its integration into HR workflows. It covers deployment, monitoring, maintenance, and strategic interventions.

* **Deployment and Dashboard Development:**
  The final CatBoost model is deployed using Streamlit and hosted publicly. The dashboard includes:

  * A manual prediction form for individual employee risk scoring
  * A batch prediction interface for monthly data uploads
  * A visual summary of overall attrition risk and trends

  Each prediction returns:

  * Attrition probability
  * Risk classification
  * Top 3 contributing features (SHAP-based)
  * Suggested HR actions

* **Monitoring and Maintenance Strategy:**
  Monitoring includes system health, prediction quality, errors, and data drift. Alerts are configured (real-time, delayed, scheduled), and model retraining is scheduled every 6 months or triggered based on defined conditions (e.g., performance drop, drift >10%, HR feedback). Rework decisions follow a structured flow from internal review to business alignment.

* **Strategic Recommendation:**
  Two levels of intervention are implemented based on different types of analysis:
    
    * **Company-Level Interventions:**
    Derived from macro trends identified through exploratory data analysis (EDA) and statistical tests. These interventions are applied organization-wide to address systemic attrition drivers.

    * **Individual-Level Interventions:**
    Based on SHAP outputs from the model, these are personalized actions targeted only at high-risk individuals. HR uses the top contributing features for each employee to design focused interventions.

* **Workflow Integration:**
    The model is embedded in monthly HR routines. After generating predictions and explanations, HR reviews at-risk employees and applies tailored interventions. Actual outcomes and feedback feed back into the model retraining loop.
