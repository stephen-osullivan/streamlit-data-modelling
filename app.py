import numpy as np
import pandas as pd
from sklearn import datasets
import streamlit as st

# Function to load custom dataset
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to load dataset from scikit-learn
def load_sklearn_dataset(dataset_name):
    data = getattr(datasets, f"load_{dataset_name}")()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df

# Function to generate statistics for numeric features
def numeric_stats(data):
    numeric_data = data.select_dtypes(include=[np.number])
    stats = numeric_data.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    stats_cols = stats.columns
    numeric_dtypes = numeric_data.dtypes
    missing_values = numeric_data.isnull().sum(axis=0)
    stats['Data Type'] = numeric_dtypes
    stats['Missing Values'] = missing_values
    return stats[['Data Type', 'Missing Values'] + list(stats_cols)]

# Function to generate statistics for categorical features
def categorical_stats(data):
    categorical_data = data.select_dtypes(include=['object', 'bool', 'string'])
    stats = []
    for column in categorical_data.columns:
        value_counts = categorical_data[column].value_counts()
        num_buckets = len(value_counts)
        top_value_count = value_counts.iloc[0]
        top_value_percentage = top_value_count / len(categorical_data) * 100
        top_value_name = value_counts.index[0]
        missing_values = categorical_data[column].isnull().sum()
        stats.append(
            {
                'Feature' : column,
                'Data Type': categorical_data[column].dtype,
                'Missing Values': missing_values,
                'Number of Buckets': num_buckets,
                'Top Bucket Percentage': top_value_percentage,
                'Top Bucket Name': top_value_name
                })
    return pd.DataFrame(stats).set_index('Feature')

def main():
    st.title("Dataset Analysis App")
    # File uploader for custom dataset
    st.sidebar.title("Upload Custom Dataset")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Dropdown for scikit-learn datasets
    st.sidebar.title("Load Dataset from scikit-learn")
    sklearn_datasets = [name[5:] for name in dir(datasets) if name.startswith("load_")]    
    sklearn_dataset_name = st.sidebar.selectbox("Choose Dataset", sklearn_datasets, index=sklearn_datasets.index('iris'))

    if file is not None:
        df = load_data(file)
        # Show pandas-profiling report for custom dataset
        st.write("### Dataset Overview")
        

    elif sklearn_dataset_name:
        df = load_sklearn_dataset(sklearn_dataset_name)
        # Show pandas-profiling report for scikit-learn dataset
        st.write(f"### {sklearn_dataset_name} Overview")
    
    st.header('Numeric Features')
    numeric_stats_table = numeric_stats(df)
    st.table(numeric_stats_table)

    st.header('Categorical Features')
    categorical_stats_table = categorical_stats(df)
    st.table(categorical_stats_table)

if __name__ == "__main__":
    main()