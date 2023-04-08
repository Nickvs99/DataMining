import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    
    df = pd.read_csv("data.csv", sep=";")

    df = set_df_types(df)

    basic_df = get_basic_df(df)
    numerical_df = get_numerical_df(df)
    categorical_df = get_categorical_df(df)

    show_basic_stats(basic_df)
    show_numerical_stats(df, numerical_df)
    show_categorical_stats(categorical_df)

def set_df_types(df):

    numeric_columns = [
        "How many students do you estimate there are in the room?",
        "What is your stress level (0-100)?",
        "How many hours per week do you do sports (in whole hours)? ",
        "Give a random number",
    ]

    categorical_columns = [
        "Have you taken a course on machine learning?",
        "Have you taken a course on information retrieval?",
        "Have you taken a course on statistics?",
        "Have you taken a course on databases?",
        "What is your gender?",
        "I have used ChatGPT to help me with some of my study assignments ",
        "Did you stand up to come to your previous answer    ?"
    ]

    # Convert columns to a numeric value, non numeric values are set to NaN
    df[numeric_columns] =  df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    
    df[categorical_columns] = df[categorical_columns].astype("category")
    
    return df

def get_basic_df(df):

    def n_invalid(series):
        return series.size - series.count()

    def percentage_valid(series):
        return series.count() / series.size

    def most_frequent_value(series):
        return series.value_counts().idxmax()

    def most_frequent_count(series):
        return series.value_counts().max()

    return df.agg(["dtype", "size", n_invalid,  percentage_valid, "nunique", most_frequent_value, most_frequent_count])

def get_numerical_df(df):

    def quantile_1(series):
        return series.quantile(0.01)

    def quantile_25(series):
        return series.quantile(0.25)
    
    def quantile_75(series):
        return series.quantile(0.75)
    
    def quantile_99(series):
        return series.quantile(0.99)

    numerical_columns = df.select_dtypes(include=np.number)
    
    return numerical_columns.agg(["mean", "std", "min", quantile_1, quantile_25, "median", quantile_75, quantile_99, "max"])

def get_categorical_df(df):

    def category_frequencies(series):
        
        value_counts = series.value_counts()

        categories = value_counts.index.tolist()
        frequencies = value_counts.values.tolist()

        # Combine categories with their respective frequency
        return list(zip(categories, frequencies))

    categorical_columns = df.select_dtypes(include='category')
    
    return categorical_columns.agg([category_frequencies])

def show_basic_stats(df):

    print_header("Basic stats")
    print(df.transpose())

    n_invalids = [df[column]["n_invalid"] for column in df.columns]
    labels = list(df.columns)

    plt.barh(labels, n_invalids)
    plt.title("Number of invalid/missing entries")

    plt.tight_layout()
    plt.show()

def show_numerical_stats(df, numerical_df):

    print_header("Numerical stats")
    print(numerical_df.transpose())

    numerical_columns = df.select_dtypes(include=np.number)

    for column in numerical_columns:

        values = df[column].values
        plt.hist(values, bins=50)
        plt.axvline(numerical_df[column]["mean"], label="mean", color="orange", linestyle="--")

        plt.title(column)
        plt.ylabel("Frequency")
        
        plt.legend()
        
        plt.show()

def show_categorical_stats(df):

    
    # Temporarily remove the max column width limit to print all output
    pd.set_option('display.max_colwidth', None)
    
    print_header("Categorical stat")
    print(df.transpose())
    
    pd.reset_option('display.max_colwidth')

    for column in df:

        category_frequencies = df[column].category_frequencies

        categories = [category_frequency[0] for category_frequency in category_frequencies]
        frequencies = [category_frequency[1] for category_frequency in category_frequencies]

        plt.pie(frequencies, labels=categories)
        plt.title(column)
        plt.show()

def print_header(header, char="="):

    print("\n")
    print(f"{char*10} {header} {char*(120 - len(header))}")
    print()

if __name__ == "__main__":
    main()