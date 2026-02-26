
# 🏦 European Bank Customer Churn Prediction

This project develops a machine learning model to predict customer churn in a European bank, followed by a Streamlit web application to provide an interactive interface for churn risk assessment.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Explainability](#model-explainability)
- [Streamlit Application](#streamlit-application)
- [Setup and Running the Application](#setup-and-running-the-application)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Project Overview
Customer churn is a critical issue for banks, impacting revenue and growth. This project aims to build a robust predictive model that can identify customers at high risk of churning, allowing the bank to implement proactive retention strategies. The solution includes a comprehensive data analysis pipeline and an interactive web application for real-time churn risk assessment.

## Features
- **Data Preprocessing & Feature Engineering**: Handles missing values, encodes categorical variables, and creates insightful new features like 'Balance-to-Salary Ratio' and 'Product Density'.
- **Multiple Model Evaluation**: Compares several classification algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) to find the best performing model.
- **XGBoost Classifier**: Utilizes XGBoost, a powerful gradient boosting framework, for its superior predictive performance.
- **Model Explainability**: Employs SHAP (SHapley Additive exPlanations) to understand the factors driving churn predictions, providing transparency and actionable insights.
- **Interactive Streamlit Web Application**: A user-friendly interface for inputting customer data and instantly getting churn probability along with business recommendations.

## Dataset
The project uses a synthetic dataset named `European_Bank.csv`, which contains various customer attributes such as:
- `CustomerId`, `Surname`
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (target variable).

## Methodology
1.  **Data Loading**: The dataset is loaded using pandas.
2.  **Preprocessing**: 
    -   Non-informative features like `CustomerId` and `Surname` are removed.
    -   Categorical features (`Geography`, `Gender`) are one-hot encoded.
3.  **Feature Engineering**: New features are created to enhance model performance:
    -   `Balance_Salary_Ratio`
    -   `Product_Density`
    -   `Engagement_Product_Interact`
    -   `Age_Tenure_Interact`
4.  **Data Splitting**: The data is split into training and testing sets with stratification to maintain class distribution.
5.  **Feature Scaling**: Numerical features are scaled using `StandardScaler`.
6.  **Model Training**: Logistic Regression, Decision Tree, Random Forest, and XGBoost models are trained on the preprocessed data.
7.  **Model Evaluation**: Models are evaluated based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Model Explainability
SHAP values are used to explain the predictions of the best performing model (XGBoost). This provides insights into which features contribute most to a customer's churn probability, both globally (feature importance plot) and for individual predictions (SHAP summary plot).

## Streamlit Application
The Streamlit application allows users to interactively input customer characteristics and receive an immediate churn risk assessment. It utilizes the trained XGBoost model and scaler to predict churn probability and provides business recommendations based on the risk level.

### Application Features:
-   **Customer Profile Input**: Sliders and select boxes for various customer attributes.
-   **Real-time Prediction**: Displays churn probability (Risk Score).
-   **Risk Categorization**: Classifies risk as LOW, MEDIUM, or HIGH.
-   **Business Recommendations**: Offers actionable advice based on the predicted risk.

## Setup and Running the Application

### Prerequisites
-   Python 3.8+
-   `pip` package manager
-   Google Colab environment (recommended)

### Installation
1.  **Clone the repository (if applicable, otherwise download the notebook)**.
2.  **Install required libraries**: Execute the following commands in your environment:
    ```bash
    !pip install shap xgboost streamlit pandas numpy scikit-learn
    ```

### Running the Notebook
Execute the cells in the provided Jupyter/Colab notebook sequentially. This will:
1.  Load the dataset.
2.  Perform data preprocessing and feature engineering.
3.  Train and evaluate the machine learning models.
4.  Generate model explainability plots.
5.  Save the trained XGBoost model (`churn_model.pkl`), the `StandardScaler` (`scaler.pkl`), and feature names (`features.pkl`).
6.  Create the `app.py` Streamlit application file.

### Running the Streamlit Web Application
After running all cells in the notebook, including the one that generates `app.py` and saves the model artifacts, run the Streamlit application using `localtunnel`:

```python
import urllib
print("IP Address:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
!streamlit run app.py & npx localtunnel --port 8501
```

This will provide a public URL (e.g., `https://xxxx-xxxx-xx.loca.lt`) which you can open in your browser to access the interactive churn risk simulator.

## Results
The model evaluation section provides a comparative analysis of different models. XGBoost generally performs best, achieving high ROC-AUC scores, which is crucial for churn prediction where minimizing false negatives (missing actual churners) is important. The Streamlit app demonstrates real-time predictions based on user inputs.

## Future Enhancements
-   Integration with a MLOps pipeline for continuous model retraining and deployment.
-   Exploration of deep learning models for churn prediction.
-   Incorporation of A/B testing for retention strategies.
-   More sophisticated feature engineering based on domain expertise.
-   User authentication and database integration for enterprise-level deployment.
