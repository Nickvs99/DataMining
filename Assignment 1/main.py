import numpy as np
import pandas as pd

def main():
    
    df = pd.read_csv("data.csv", sep=";")

    df = set_df_types(df)

    print_basic_stats(df)
    print_numerical_stats(df)
    print_categorial_stats(df)
    

def set_df_types(df):

    numeric_columns = [
        "How many students do you estimate there are in the room?",
        "What is your stress level (0-100)?",
        "How many hours per week do you do sports (in whole hours)? ",
        "Give a random number",
    ]

    categorial_columns = [
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
    
    df[categorial_columns] = df[categorial_columns].astype("category")
    
    return df

def print_basic_stats(df):

    def n_invalid(series):
        return series.size - series.count()

    def percentage_valid(series):
        return series.count() / series.size

    def most_frequent_value(series):
        return series.value_counts().idxmax()

    def most_frequent_count(series):
        return series.value_counts().max()

    print_header("Basic data")
    print(df.agg(["dtype", "size", n_invalid,  percentage_valid, "nunique", most_frequent_value, most_frequent_count]).transpose())

def print_numerical_stats(df):

    def quantile_1(series):
        return series.quantile(0.01)

    def quantile_25(series):
        return series.quantile(0.25)
    
    def quantile_75(series):
        return series.quantile(0.75)
    
    def quantile_99(series):
        return series.quantile(0.99)

    numerical_columns = df.select_dtypes(include=np.number)
    
    print_header("Numerical data")
    print(numerical_columns.agg(["mean", "std", "min", quantile_1, quantile_25, "median", quantile_75, quantile_99, "max"]).transpose())


def print_categorial_stats(df):

    def category_frequencies(series):
        
        value_counts = series.value_counts()

        categories = value_counts.index.tolist()
        frequencies = value_counts.values.tolist()

        # Combine categories with their respective frequency
        return list(zip(categories, frequencies))

    categorial_columns = df.select_dtypes(include='category')
    
    # Temporarily remove the max column width limit to print all output
    pd.set_option('display.max_colwidth', None)
    
    print_header("Categorial data")
    print(categorial_columns.agg([category_frequencies]).transpose())
    
    pd.reset_option('display.max_colwidth')

def print_header(header, char="="):

    print("\n")
    print(f"{char*10} {header} {char*(120 - len(header))}")
    print()

if __name__ == "__main__":
    main()