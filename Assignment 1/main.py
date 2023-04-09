import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    
    df = pd.read_csv("data.csv", sep=";")

    set_df_types(df)
    column_name_map = update_column_names(df)

    basic_df = get_basic_df(df)
    numerical_df = get_numerical_df(df)
    categorical_df = get_categorical_df(df)

    show_basic_stats(basic_df)
    show_numerical_stats(df, numerical_df, column_name_map)
    show_categorical_stats(categorical_df, column_name_map)

    show_correlation_matrix(df)

    plot_scatterplot(df, "Sport", "Stress level")
    plot_boxplot(df, "Stress level", ["Machine learning", "Information retrieval", "Statistics", "Databases"])


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


def update_column_names(df):

    column_names = [
        "Timestamp",
        "Program",
        "Machine learning",
        "Information retrieval",
        "Statistics",
        "Databases",
        "Gender",
        "ChatGPT",
        "Birthday",
        "Students estimate",
        "Stand up",
        "Stress level",
        "Sport",
        "Random number",
        "Bedtime",
        "Good day (#1)",
        "Good day (#2)",
    ]

    # Creates a dictionary which maps the shortened column name to it's longer version
    # e.g. Program --> What programme are you in?
    column_name_map = dict(zip(column_names, df.columns))
    
    df.columns = column_names

    return column_name_map


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
    print_df(df.transpose())

    n_invalids = [df[column]["n_invalid"] for column in df.columns]

    labels = df.columns.values
    plt.barh(labels, n_invalids)
    plt.title("Number of invalid/missing entries")

    plt.tight_layout()
    plt.show()


def show_numerical_stats(df, numerical_df, column_name_map):

    print_header("Numerical stats")
    print_df(numerical_df.transpose())

    numerical_columns = df.select_dtypes(include=np.number)

    for column in numerical_columns:

        values = df[column].values
        
        plt.hist(values, bins=50)
        plt.axvline(numerical_df[column]["mean"], label="mean", color="orange", linestyle="--")

        plt.title(column_name_map[column])
        plt.ylabel("Frequency")
        
        plt.legend()
        
        plt.show()


def show_categorical_stats(df, column_name_map):

    print_header("Categorical stats")
    print_df(df.transpose())
    
    for column in df:

        category_frequencies = df[column].category_frequencies

        categories = [category_frequency[0] for category_frequency in category_frequencies]
        frequencies = [category_frequency[1] for category_frequency in category_frequencies]

        plt.pie(frequencies, labels=categories)
        plt.title(column_name_map[column])
        plt.show()


def print_df(df):
    with pd.option_context('display.max_colwidth', None):
        print(df)


def print_header(header, char="="):

    print("\n")
    print(f"{char*10} {header} {char*(120 - len(header))}")
    print()


def show_correlation_matrix(df):
    
    df_copy = df.copy()

    # Convert all categorical data to numerical data
    # https://stackoverflow.com/a/32011969/12132063
    categorical_columns = df_copy.select_dtypes(['category']).columns
    df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.cat.codes)

    # Calculate the correlation matrix, for some reason method="pearson" does not give a matrix with one's on the diagonal
    correlation_matrix = df_copy.corr(method="kendall", numeric_only=True)

    print_header("Correlation matrix")
    print_df(correlation_matrix)

    plt.matshow(correlation_matrix)
    
    plt.title("Correlation matrix")

    labels = correlation_matrix.columns.values
    tick_locations = range(len(labels))
    plt.xticks(tick_locations, labels, rotation=30, horizontalalignment="left")
    plt.yticks(tick_locations, labels)

    plt.colorbar()

    plt.show()

def plot_scatterplot(df, index1, index2):

    values1 = df[index1].values
    values2 = df[index2].values
    
    plt.scatter(values1, values2)
    
    plt.xlabel(index1)
    plt.ylabel(index2)
    
    plt.show()

def plot_boxplot(df, index1, indices):
    """
    Creates a boxplot for multiple columns.
    Index1 has to be a numerical column
    Indices has to be an array of categorical columns
    """

    data = []
    labels = []

    for index in indices:

        for category in df[index].cat.categories:

            labels.append(f"{index} - {category}")

            # Get all rows where the column value is equal to the current category
            rows = df.loc[df[index] == category]
            values = rows[index1].values
            
            # Remove nan values from the data
            values = values[~np.isnan(values)]

            data.append(values)

    plt.boxplot(data, labels=labels)

    plt.ylabel(index1)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()