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
        df[col] = df[col].fillna(df[col].mean())  # Fill NaN with mean to allow model fitting
        outlier_labels = model.fit_predict(df[[col]])
        outliers[col] = (outlier_labels == -1).sum()
    return outliers

def impute_missing_values(df):
    """Handle missing value imputation for numerical, categorical, and datetime data."""
    df = df.copy()

    # Handle numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce invalid values to NaN
    numerical_imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    # Handle datetime columns
    datetime_cols = df.select_dtypes(include=['object']).columns
    for col in datetime_cols:
        # Convert to datetime if possible
        if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).any():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else pd.Timestamp.now())

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
    df = df.copy()
    df = impute_missing_values(df)
    outliers = detect_outliers(df)
    df = handle_datetime_columns(df)
    df = feature_engineering(df)
    return df, outliers

def feature_selection(df):
    """Perform feature selection using LassoCV and Chi-Square test."""
    X = df.select_dtypes(include=[np.number])
    y = df['target'] if 'target' in df.columns else X.iloc[:, 0]
    X = X.applymap(lambda x: max(0, x))  # Ensure all values are positive for Chi-Square
    X = X.fillna(0)

    # LassoCV for numerical feature selection
    lasso = LassoCV(cv=5).fit(X, y)
    selected_features = X.columns[lasso.coef_ != 0]

    # Chi-Square for categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        chi2_selector = SelectKBest(chi2, k='all')
        X_cat = df[categorical_cols].apply(lambda col: pd.factorize(col)[0])
        chi2_selector.fit(X_cat, y)
        chi2_features = X_cat.columns[chi2_selector.get_support()]
    else:
        chi2_features = []

    return list(selected_features), list(chi2_features)

def evaluate_data_quality(df):
    """Evaluate data quality using completeness, consistency, and accuracy scores."""
    completeness_score = 1 - df.isnull().mean().mean()
    consistency_score = (df.nunique() / len(df)).mean()
    accuracy_score = 1 - df.duplicated().mean()
    quality_score = {
        "completeness": completeness_score,
        "consistency": consistency_score,
        "accuracy": accuracy_score
    }
    return quality_score

# Streamlit UI
st.title('Data Cleaning Tool')

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Dataset Overview:")
    st.write(df.head())

    # Data Quality Before Cleaning
    st.write("Data Quality Scores (Before Cleaning):")
    before_quality = evaluate_data_quality(df)
    st.write(before_quality)

    # Clean the Dataset
    cleaned_df, outliers = clean_dataset(df)

    st.write("Cleaned Dataset:")
    st.write(cleaned_df.head())

    # Data Quality After Cleaning
    st.write("Data Quality Scores (After Cleaning):")
    after_quality = evaluate_data_quality(cleaned_df)
    st.write(after_quality)

    # Feature Selection
    important_features, chi2_features = feature_selection(cleaned_df)

    st.write("Outliers in the Dataset:")
    st.write(outliers)

    st.write("Selected Features using LassoCV:")
    st.write(important_features)

    st.write("Selected Features using Chi-Square Test:")
    st.write(chi2_features)

    # Download the Cleaned Dataset
    cleaned_csv = cleaned_df.to_csv(index=False)
    st.download_button("Download Cleaned Dataset", cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")
