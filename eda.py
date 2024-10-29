# Exploratory data analysis on the BAF dataset.
# https://arxiv.org/abs/2211.13358


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = "data"


def preprocess():
    # Read the data from the CSV file.
    df = pd.read_csv(f"{DATA_DIR}/baf.csv")

    # Display the first few rows of the data.
    print(df.head())

    # Display the data types of each column.
    print(df.dtypes)

    # Display the summary statistics of the data.
    pd.set_option("display.max_columns", None)
    print(df.describe())

    # From the summary statistics, we see that the 'device_fraud_count' column is all zeros. We can thus remove it from our dataset.
    df.drop(columns=["device_fraud_count"], inplace=True)

    # We also see that some columns have missing values, such as 'session_length_in_minutes', as it contains -1 values.
    # Display the number of -1 values in each column.
    # The main issues here are 'prev_address_months_count' and 'bank_months_count', which have a significant number of missing values.
    print(df.isin([-1]).sum())

    # Display the correlation matrix of the data.
    # We see a high negative correlation between 'velocity_4w' and 'month'. This is because 'velocity_4w' is calculated based on the month.
    # We can thus remove 'velocity_4w' from our dataset.
    numer_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr_matrix = df[numer_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()
    df.drop(columns=["velocity_4w"], inplace=True)

    # Display the distribution of the target variable.
    # We see that the target variable is imbalanced, with more non-fraudulent transactions than fraudulent transactions.
    # To handle this, we will use ensemble methods such as XGBoost, which can handle imbalanced datasets.
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="fraud_bool")
    plt.title("Distribution of Target Variable")
    plt.show()

    # Perform one-hot encoding on categorical columns.
    categ_cols = df.select_dtypes(include="object").columns
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categ_cols])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(categ_cols),
    )
    df.drop(columns=categ_cols, axis=1, inplace=True)
    df = pd.concat([df, one_hot_df], axis=1, copy=False)

    # Return the preprocessed data.
    return df


if __name__ == "__main__":
    preprocess()
