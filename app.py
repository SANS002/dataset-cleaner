
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.impute import KNNImputer, SimpleImputer
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import LassoCV
# import streamlit as st
# import wikipedia

# def get_missing_value_from_internet(category, row_data):
#     """
#     Fetch missing data from the internet based on category and row data.
#     :param category: The category of missing data (e.g., language, movie, politics, finance).
#     :param row_data: The corresponding row data for context.
#     :return: Fetched data or a default message if data cannot be fetched.
#     """
#     try:
#         search_term = " ".join([str(value) for value in row_data.values if pd.notnull(value)])
#         if category == "movie":
#             return wikipedia.summary(search_term, sentences=1)
#         elif category == "language":
#             return f"Language-related information about {search_term}"
#         elif category == "politics":
#             return f"Political information about {search_term}"
#         elif category == "finance":
#             return f"Financial information about {search_term}"
#         else:
#             return wikipedia.summary(search_term, sentences=1)
#     except wikipedia.exceptions.DisambiguationError as e:
#         return f"Multiple entries found: {e.options}"
#     except wikipedia.exceptions.PageError:
#         return "No page found for the search term."
#     except Exception as e:
#         return f"Error while fetching data: {e}"

# def detect_outliers(df, contamination=0.1):
#     """Detect outliers in numerical columns using Isolation Forest."""
#     outliers = {}
#     for col in df.select_dtypes(include=[np.number]).columns:
#         model = IsolationForest(contamination=contamination, random_state=42)
#         df[col] = df[col].fillna(df[col].mean())
#         outlier_labels = model.fit_predict(df[[col]])
#         outliers[col] = (outlier_labels == -1).sum()
#     return outliers

# def impute_missing_values(df):
#     """Handle missing value imputation for numerical and categorical data."""
#     numerical_cols = df.select_dtypes(include=[np.number]).columns
#     numerical_imputer = KNNImputer(n_neighbors=5)
#     df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

#     categorical_cols = df.select_dtypes(include=[object]).columns
#     imputer = SimpleImputer(strategy='most_frequent')
#     df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

#     return df

# def handle_datetime_columns(df):
#     """Convert date-like strings to datetime objects."""
#     for col in df.select_dtypes(include=['object']).columns:
#         if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).all():
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#     return df

# def clean_dataset(df):
#     """Clean the dataset by handling missing values, outliers, and datetime columns."""
#     df = df.copy()
#     df = impute_missing_values(df)
#     outliers = detect_outliers(df)
#     df = handle_datetime_columns(df)

#     for col in df.columns:
#         for index, row in df[df[col].isnull()].iterrows():
#             if pd.isna(row[col]):
#                 category = "general"
#                 if 'category' in row and pd.notnull(row['category']):
#                     category = row['category'].lower()
#                 missing_value = get_missing_value_from_internet(category, row)
#                 df.at[index, col] = missing_value

#     return df, outliers

# def feature_selection(df):
#     """Perform feature selection using LassoCV and Chi-Square test."""
#     X = df.select_dtypes(include=[np.number])
#     y = df['target'] if 'target' in df.columns else X.iloc[:, 0]
#     X = X.applymap(lambda x: max(0, x))
#     X = X.fillna(0)

#     lasso = LassoCV(cv=5).fit(X, y)
#     selected_features = X.columns[lasso.coef_ != 0]

#     categorical_cols = X.select_dtypes(include=['object']).columns
#     if len(categorical_cols) > 0:
#         chi2_selector = SelectKBest(chi2, k='all')
#         X_cat = X[categorical_cols].apply(lambda col: pd.factorize(col)[0])
#         chi2_selector.fit(X_cat, y)
#         chi2_features = X_cat.columns[chi2_selector.get_support()]
#     else:
#         chi2_features = []

#     return list(selected_features), list(chi2_features)

# def evaluate_data_quality(df):
#     """Evaluate data quality using completeness, consistency, and accuracy scores."""
#     completeness_score = 1 - df.isnull().mean().mean()
#     consistency_score = (df.nunique() / len(df)).mean()
#     accuracy_score = 1 - df.duplicated().mean()

#     quality_score = {
#         "completeness": completeness_score,
#         "consistency": consistency_score,
#         "accuracy": accuracy_score
#     }
#     return quality_score

# # Streamlit UI
# st.title('Enhanced Data Cleaning Tool')

# uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# if uploaded_file:
#     if uploaded_file.name.endswith('.csv'):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     st.write("Dataset Overview:")
#     st.write(df.head())

#     st.write("Data Quality Scores (Before Cleaning):")
#     before_quality = evaluate_data_quality(df)
#     st.write(before_quality)

#     cleaned_df, outliers = clean_dataset(df)

#     st.write("Cleaned Dataset:")
#     st.write(cleaned_df.head())

#     st.write("Data Quality Scores (After Cleaning):")
#     after_quality = evaluate_data_quality(cleaned_df)
#     st.write(after_quality)

#     important_features, chi2_features = feature_selection(cleaned_df)

#     st.write("Outliers in the dataset:")
#     st.write(outliers)

#     st.write("Selected Features using LassoCV:")
#     st.write(important_features)

#     st.write("Selected Features using Chi-Square Test:")
#     st.write(chi2_features)

#     cleaned_csv = cleaned_df.to_csv(index=False)
#     st.download_button("Download Cleaned Dataset", cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.impute import KNNImputer, SimpleImputer
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import LassoCV
# import streamlit as st

# def detect_outliers(df, contamination=0.1):
#     """Detect outliers in numerical columns using Isolation Forest."""
#     outliers = {}
#     for col in df.select_dtypes(include=[np.number]).columns:
#         model = IsolationForest(contamination=contamination, random_state=42)
#         df[col] = df[col].fillna(df[col].mean())
#         outlier_labels = model.fit_predict(df[[col]])
#         outliers[col] = (outlier_labels == -1).sum()
#     return outliers

# def impute_missing_values(df):
#     """Handle missing value imputation for numerical and categorical data."""
#     numerical_cols = df.select_dtypes(include=[np.number]).columns
#     numerical_imputer = KNNImputer(n_neighbors=5)
#     df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

#     categorical_cols = df.select_dtypes(include=[object]).columns
#     imputer = SimpleImputer(strategy='most_frequent')
#     df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

#     return df

# def handle_datetime_columns(df):
#     """Convert date-like strings to datetime objects."""
#     for col in df.select_dtypes(include=['object']).columns:
#         if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).all():
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#     return df

# def clean_dataset(df):
#     """Clean the dataset by handling missing values, outliers, and datetime columns."""
#     df = df.copy()
#     df = impute_missing_values(df)
#     outliers = detect_outliers(df)
#     df = handle_datetime_columns(df)
#     return df, outliers

# def feature_selection(df):
#     """Perform feature selection using LassoCV and Chi-Square test."""
#     X = df.select_dtypes(include=[np.number])
#     y = df['target'] if 'target' in df.columns else X.iloc[:, 0]
#     X = X.applymap(lambda x: max(0, x))
#     X = X.fillna(0)

#     lasso = LassoCV(cv=5).fit(X, y)
#     selected_features = X.columns[lasso.coef_ != 0]

#     categorical_cols = X.select_dtypes(include=['object']).columns
#     if len(categorical_cols) > 0:
#         chi2_selector = SelectKBest(chi2, k='all')
#         X_cat = X[categorical_cols].apply(lambda col: pd.factorize(col)[0])
#         chi2_selector.fit(X_cat, y)
#         chi2_features = X_cat.columns[chi2_selector.get_support()]
#     else:
#         chi2_features = []

#     return list(selected_features), list(chi2_features)

# def evaluate_data_quality(df):
#     """Evaluate data quality using completeness, consistency, and accuracy scores."""
#     completeness_score = 1 - df.isnull().mean().mean()
#     consistency_score = (df.nunique() / len(df)).mean()
#     accuracy_score = 1 - df.duplicated().mean()

#     quality_score = {
#         "completeness": completeness_score,
#         "consistency": consistency_score,
#         "accuracy": accuracy_score
#     }
#     return quality_score

# # Streamlit UI
# st.title('Enhanced Data Cleaning Tool')

# uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# if uploaded_file:
#     if uploaded_file.name.endswith('.csv'):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     st.write("Dataset Overview:")
#     st.write(df.head())

#     st.write("Data Quality Scores (Before Cleaning):")
#     before_quality = evaluate_data_quality(df)
#     st.write(before_quality)

#     cleaned_df, outliers = clean_dataset(df)

#     st.write("Cleaned Dataset:")
#     st.write(cleaned_df.head())

#     st.write("Data Quality Scores (After Cleaning):")
#     after_quality = evaluate_data_quality(cleaned_df)
#     st.write(after_quality)

#     important_features, chi2_features = feature_selection(cleaned_df)

#     st.write("Outliers in the dataset:")
#     st.write(outliers)

#     st.write("Selected Features using LassoCV:")
#     st.write(important_features)

#     st.write("Selected Features using Chi-Square Test:")
#     st.write(chi2_features)

#     cleaned_csv = cleaned_df.to_csv(index=False)
#     st.download_button("Download Cleaned Dataset", cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")

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
        df[col] = df[col].fillna(df[col].mean())
        outlier_labels = model.fit_predict(df[[col]])
        outliers[col] = (outlier_labels == -1).sum()
    return outliers

def impute_missing_values(df):
    """Handle missing value imputation for numerical and categorical data."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    categorical_cols = df.select_dtypes(include=[object]).columns
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    return df

def handle_datetime_columns(df):
    """Convert date-like strings to datetime objects."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).all():
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
    X = X.applymap(lambda x: max(0, x))
    X = X.fillna(0)

    lasso = LassoCV(cv=5).fit(X, y)
    selected_features = X.columns[lasso.coef_ != 0]

    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        chi2_selector = SelectKBest(chi2, k='all')
        X_cat = X[categorical_cols].apply(lambda col: pd.factorize(col)[0])
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
st.title('Enhanced Data Cleaning Tool')

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

    important_features, chi2_features = feature_selection(cleaned_df)

    st.write("Outliers in the dataset:")
    st.write(outliers)

    st.write("Selected Features using LassoCV:")
    st.write(important_features)

    st.write("Selected Features using Chi-Square Test:")
    st.write(chi2_features)

    cleaned_csv = cleaned_df.to_csv(index=False)
    st.download_button("Download Cleaned Dataset", cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")

