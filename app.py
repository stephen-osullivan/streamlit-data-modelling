import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import streamlit as st

import math

# Function to load custom dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to load dataset from scikit-learn
@st.cache_data
def load_sklearn_dataset(dataset_name):
    data = getattr(datasets, f"load_{dataset_name}")()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df

# Function to get the number of rows and features of each type
def data_overview(data):
    num_rows, num_features = data.shape[0], data.shape[1]
    feature_types = data.dtypes.value_counts()
    overview = pd.DataFrame(
        {'Number of Rows': [num_rows],
        'Number of Features': [num_features]} | feature_types.to_dict(),
        index=['Dataset Overview'])
    return overview

# Function to generate statistics for numeric features
def numeric_stats(data):
    numeric_data = data.select_dtypes(include=[np.number])
    stats = numeric_data.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).drop('count').T
    stats['skew'] = numeric_data.skew(axis=0).T
    stats['kurtosis'] =  numeric_data.kurt(axis=0).T
    perc_cols = [c for c in stats.columns if '%' in c]
    numeric_dtypes = numeric_data.dtypes
    missing_values = numeric_data.isnull().sum(axis=0)
    stats['dtype'] = numeric_dtypes
    stats['nulls'] = missing_values
    return stats[['dtype', 'nulls', 'mean', 'std', 'skew', 'kurtosis' ,'min'] + perc_cols +['max']]

# Function to generate statistics for categorical features
def categorical_stats(data):
    categorical_data = data.select_dtypes(include=['object', 'bool', 'string'])
    if len(categorical_data.columns) == 0:
        return pd.DataFrame()
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
                'feature' : column,
                'dtype': categorical_data[column].dtype,
                'nulls': missing_values,
                'Buckets': num_buckets,
                'Top Bucket %': top_value_percentage,
                'Top Bucket Value': top_value_name,
            })
    return pd.DataFrame(stats).set_index('feature')

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide

    st.title("Dataset Analysis App")

    ### side bar
    
    dataset_type = st.sidebar.selectbox('Choose Dataset Type', ['public', 'sklearn', 'upload'])
    if dataset_type == 'public':
        # datasets saved within the data folder of this app
        st.sidebar.title("Load Public Dataset")
        public_datasets = ['titanic']    
        public_dataset = st.sidebar.selectbox(
            "Choose Dataset", public_datasets)
        if public_dataset is not None:
            df = load_data(f'data/{public_dataset}.csv')

    elif dataset_type == 'sklearn':
        # Dropdown for scikit-learn datasets
        st.sidebar.title("Load Dataset from scikit-learn")
        sklearn_datasets = ['iris', 'wine', 'breast_cancer','diabetes']    
        sklearn_dataset_name = st.sidebar.selectbox(
            "Choose Dataset", sklearn_datasets, index=sklearn_datasets.index('iris'))
        if sklearn_dataset_name:
            df = load_sklearn_dataset(sklearn_dataset_name)
         
    elif dataset_type == 'upload':
        # File uploader for custom dataset
        st.sidebar.title("Upload Custom Dataset")
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            df = load_data(file)    

    ### main element
    tab1, tab2, tab3 = st.tabs(["Upload and Stats", "Plotting", "Modelling"])
    with tab1:
        st.header('Data Overview')
        data_overview_table = data_overview(df)
        st.dataframe(data_overview_table)

        st.header('Data Snippet')
        st.dataframe(df.head(10), use_container_width=True)

        st.header('Numeric Feature Stats')
        numeric_stats_table = numeric_stats(df)
        bar_columns = numeric_stats_table.columns[2:]
        st.dataframe(
            numeric_stats_table.style.format(precision=2),
            width=1000,
            )

        st.header('Categorical Feature Stats')
        categorical_stats_table = categorical_stats(df)
        st.dataframe(
            categorical_stats_table,
            width = 800)
    with tab2:
        st.header('Visualization Options')
        visualization_option = st.radio('Select Visualization', ['Correlation Matrix and KDE', 'Pairplot with Correlation'])

        if visualization_option == 'Pairplot with Correlation':
            st.header('Pairplot with Correlation')
            # Create a pairplot
            sns.set(style="ticks")
            g = sns.pairplot(df.select_dtypes(include=np.number), diag_kind='kde')

            # Calculate correlations
            correlations = df.select_dtypes(include=np.number).corr()

            # Add correlation values to the lower left corner
            for i, j in zip(*np.tril_indices_from(g.axes, -1)):
                g.axes[i, j].annotate("%.2f" % correlations.iloc[i, j], (0.1, 0.1), xycoords='axes fraction', ha='center', va='center', fontsize=8, color="black")

            st.pyplot(g)

        elif visualization_option == 'Correlation Matrix and KDE':
            st.header('Correlation Matrix and KDE Plots')
            # Calculate correlations
            numeric_features = df.select_dtypes(include=np.number).columns
            correlations = df[numeric_features].corr()
            num_numeric_features = len(numeric_features)
            # Plot KDE plots for numeric features
            st.subheader('KDE Plots')

            cols = num_numeric_features if num_numeric_features < 5 else 5
            rows = 1 if num_numeric_features < 5 else math.ceil(num_numeric_features/5)
            fig, axs = plt.subplots(rows, cols, figsize=(24, 5 * rows))
            for feature, ax in zip(numeric_features, axs.ravel()):
                sns.kdeplot(df[feature], fill=True, ax = ax)
            st.pyplot(fig, use_container_width=False)
            
            # Plot correlation matrix as a heatmap
    
            fig, ax = plt.subplots(1, 1, figsize=(2+num_numeric_features//2, 2+num_numeric_features//2))
            cmap = sns.diverging_palette(250, 10, as_cmap=True)

            sns.heatmap(
                correlations, annot=True, fmt=".2f", cmap=cmap, cbar=False, 
                vmin = -1, vmax = 1, center = 0, ax=ax)
            st.pyplot(fig, use_container_width=False)

        

if __name__ == "__main__":
    main()