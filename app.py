import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LassoCV
import streamlit as st

def detect_outliers(df, contamination=0.1):
    """Detect outliers in numerical columns using Isolation Forest."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        model = IsolationForest(contamination=contamination, random_state=42)
        df[col] = df[col].fillna(df[col].mean())  # Replace missing values before detecting outliers
        outlier_labels = model.fit_predict(df[[col]])
        outliers[col] = (outlier_labels == -1).sum()
    return outliers

def impute_missing_values(df):
    """Handle missing value imputation for numerical and categorical data."""
    df = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns

    # Impute numerical columns if any exist
    if len(numerical_cols) > 0:
        numerical_imputer = KNNImputer(n_neighbors=5)
        try:
            df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
        except ValueError:
            st.warning("Numerical columns could not be imputed. Ensure valid numerical data.")
    
    # Impute categorical columns if any exist
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    return df

def handle_datetime_columns(df):
    """Convert date-like strings to datetime objects."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).any():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def feature_engineering(df):
    """Perform feature engineering by adding useful derived features."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(df[col].abs())
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
    return df

def clean_dataset(df):
    """Clean the dataset by handling missing values, outliers, and datetime columns."""
    if df.empty:
        st.warning("The uploaded dataset is empty. Please provide a valid dataset.")
        return df, {}

    df = df.copy()
    df = impute_missing_values(df)
    outliers = detect_outliers(df)
    df = handle_datetime_columns(df)
    df = feature_engineering(df)
    return df, outliers

def evaluate_data_quality(df):
    """Evaluate data quality using completeness, consistency, and accuracy scores."""
    if df.empty:
        return {"completeness": 0, "consistency": 0, "accuracy": 0}
    completeness_score = 1 - df.isnull().mean().mean()
    consistency_score = (df.nunique() / len(df)).mean()
    accuracy_score = 1 - df.duplicated().mean()
    quality_score = {
        "completeness": completeness_score,
        "consistency": consistency_score,
        "accuracy": accuracy_score,
    }
    return quality_score

# Streamlit UI
st.title('DATA CLEANING TOOL')
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Dataset Overview:")
    st.write(df.head())
    
    st.write("Data Quality Scores (Before Cleaning):")
    before_quality = evaluate_data_quality(df)
    st.write(before_quality)
    
    cleaned_df, outliers = clean_dataset(df)
    
    st.write("Cleaned Dataset:")
    st.write(cleaned_df.head())
    
    st.write("Data Quality Scores (After Cleaning):")
    after_quality = evaluate_data_quality(cleaned_df)
    st.write(after_quality)
    
    st.write("Outliers in the dataset:")
    st.write(outliers)
    
    cleaned_csv = cleaned_df.to_csv(index=False)
    st.download_button("Download Cleaned Dataset", cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")
