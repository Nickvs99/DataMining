import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_cleaning import clean_df
from feature_engineering import run_feature_engineering

from evaluators.category_evaluator import CategoryEvaluator

from predictors.knn_predictor import KnnPredictor
from predictors.naive_bayes_predictor   import NaiveBayesPredictor

from validators.basic_validator import BasicValidator
from validators.k_fold_validator import KFoldValidator


def main():
    
    df = pd.read_csv("data.csv", sep=";")
    
    column_name_map = update_column_names(df)
    set_df_types(df)

    # Randomize order of rows, https://stackoverflow.com/a/34879805/12132063
    # Use a random_state to have the same order between runs
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    df_clean_remove = clean_df(df, method="remove")
    df_clean_replace = clean_df(df, method="replace")
    
    # Run feature engineering on the cleaned versions
    for dataframe in [df_clean_remove, df_clean_replace]:
        run_feature_engineering(dataframe)

    # # Get basic data plots and tables
    # save_suffixs = ["dirty", "remove", "replace"]
    # dfs = [df, df_clean_remove, df_clean_replace]
    # for dataframe, save_suffix in zip(dfs, save_suffixs):
    #     run_df(dataframe, column_name_map, save_suffix=save_suffix)

    # Only keep these columns
    df = df_clean_replace[["Gender", "Stress level", "Sport"]].copy()

    normalize_df(df)

    n_rows = len(df.index)

    test_fraction = 1/3
    n_test_rows = int(n_rows * test_fraction)

    test_df = df[:n_test_rows]
    other_df = df[n_test_rows:]
    
    target = "Gender"
    for k_folds in range(2, 20):

        predictor = NaiveBayesPredictor(target, n_category_bins=5)
        # predictor = KnnPredictor(target, k=k, n=2)
    
        evaluator = CategoryEvaluator(target, predictor)

        # validator = BasicValidator(other_df, evaluator, predictor, validate_fraction=0.2)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=k_folds)
        score, std_error = validator.validate()
        
        print(f"k={k_folds}, predication accuracy: {score} +- {std_error}")


def run_df(df, column_name_map, save_suffix=""):
           
    basic_df = get_basic_df(df)
    numerical_df = get_numerical_df(df)
    categorical_df = get_categorical_df(df)

    show_basic_stats(basic_df, save_suffix=save_suffix)
    show_numerical_stats(df, numerical_df, column_name_map, save_suffix=save_suffix)
    show_categorical_stats(categorical_df, column_name_map, save_suffix=save_suffix)

    show_correlation_matrix(df, save_filename=f"correlation_{save_suffix}")

    plot_scatterplot(df, "Sport", "Stress level", save_filename=f"scatter_sport_stress_{save_suffix}")
    plot_boxplot(df, "Stress level", ["Gender"], save_filename=f"boxplot_stress_gender_{save_suffix}")

    plot_stress_sport_gender(df, save_suffix=save_suffix)


def set_df_types(df):

    numeric_columns = [
        "Students estimate",
        "Stress level",
        "Sport",
        "Random number",
    ]

    categorical_columns = [
        "Machine learning",
        "Information retrieval",
        "Statistics",
        "Databases",
        "Gender",
        "ChatGPT",
        "Stand up"
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

def normalize_df(df):
    """
    Normalizes the numerical columns. 
    Currently, the values are divided by the max value.
    Another possibility is to also shift the values such that the minvalue is 0
    """

    numerical_columns = df.select_dtypes(include=np.number)
    for column in numerical_columns:

        max_value = df[column].max()

        df[column] /= max_value


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


def show_basic_stats(df, save_suffix=""):

    print_header("Basic stats")
    print_df(df.transpose())
    save_df(df.transpose(), f"tables/basic_stats_{save_suffix}.tex")

    n_invalids = [df[column]["n_invalid"] for column in df.columns]

    labels = df.columns.values
    plt.barh(labels, n_invalids)
    plt.title("Number of invalid/missing entries")

    plt.tight_layout()
    plt.savefig(f"figures/invalids_{save_suffix}.png", dpi=400)
    plt.show()


def show_numerical_stats(df, numerical_df, column_name_map, save_suffix=""):

    print_header("Numerical stats")
    print_df(numerical_df.transpose())
    save_df(numerical_df.transpose(), f"tables/numerical_stats_{save_suffix}.tex")

    numerical_columns = df.select_dtypes(include=np.number)

    for column in numerical_columns:

        values = df[column].values
        
        plt.hist(values, bins=20)
        plt.axvline(numerical_df[column]["mean"], label="mean", color="orange", linestyle="--")

        plt.title(column_name_map[column] if column in column_name_map else column)
        plt.ylabel("Frequency")
        
        plt.legend()   

        plt.savefig(f"figures/numeric_{column}_{save_suffix}.png", dpi=400)
        plt.show()


def show_categorical_stats(df, column_name_map, save_suffix=""):

    print_header("Categorical stats")
    print_df(df.transpose())
    save_df(df.transpose(), f"tables/categorical_stats_{save_suffix}.tex")
    
    for column in df:

        category_frequencies = df[column].category_frequencies

        categories = [category_frequency[0] for category_frequency in category_frequencies]
        frequencies = [category_frequency[1] for category_frequency in category_frequencies]

        plt.pie(frequencies, labels=categories)
        plt.title(column_name_map[column] if column in column_name_map else column)

        plt.savefig(f"figures/category_{column}_{save_suffix}.png", dpi=400)
        plt.show()

def save_df(df, path):

    def float_formatter(value, exp_treshold=3):
        """
        Print values with a low exponent using decimal notation.
        Print values with a large exponent using scientific notation.
        """

        if 10**(-exp_treshold) < abs(value) < 10**exp_treshold:
            return f"{value:.2f}"
        else:
            return f"{value:.2e}"
    
    with pd.option_context('display.max_colwidth', None):
        df.to_latex(path, float_format=lambda value:float_formatter(value))

def print_df(df):
    with pd.option_context('display.max_colwidth', None):
        print(df)


def print_header(header, char="="):

    print("\n")
    print(f"{char*10} {header} {char*(120 - len(header))}")
    print()


def show_correlation_matrix(df, save_filename=""):
    
    df_copy = df.copy()

    # Convert all categorical data to numerical data
    # https://stackoverflow.com/a/32011969/12132063
    categorical_columns = df_copy.select_dtypes(['category']).columns
    df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.cat.codes)

    # Calculate the correlation matrix, for some reason method="pearson" does not give a matrix with one's on the diagonal
    correlation_matrix = df_copy.corr(method="kendall", numeric_only=True)

    print_header("Correlation matrix")
    print_df(correlation_matrix)

    plt.imshow(correlation_matrix)
    
    plt.title("Correlation matrix")

    labels = correlation_matrix.columns.values
    tick_locations = range(len(labels))
    plt.xticks(tick_locations, labels, rotation=30, horizontalalignment="right")
    plt.yticks(tick_locations, labels)

    plt.colorbar()
    plt.tight_layout()

    if save_filename:
        plt.savefig(f"figures/{save_filename}.png", dpi=400)

    plt.show()

def plot_scatterplot(df, index1, index2, save_filename=""):

    values1 = df[index1].values
    values2 = df[index2].values
    
    plt.scatter(values1, values2)
    
    plt.xlabel(index1)
    plt.ylabel(index2)
    
    plt.tight_layout()
   
    if save_filename:
        plt.savefig(f"figures/{save_filename}.png", dpi=400)

    plt.show()

def plot_boxplot(df, index1, indices, save_filename=""):
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

    if save_filename:
        plt.savefig(f"figures/{save_filename}.png", dpi=400)

    plt.show()

def plot_stress_sport_gender(df, save_suffix=""):

    stress_values = df["Stress level"].values
    sport_values = df["Sport"].values
    gender_values = df["Gender"].values

    cdict = {"female": 'C6', "gender fluid": "C10"}


    # fig, ax = plt.subplots()
    for gender in np.unique(gender_values):
        ix = np.where(gender_values == gender)
        color = cdict[gender] if gender in cdict else None
        plt.scatter(sport_values[ix], stress_values[ix], label=gender, c=color, s=50)
    
    plt.xlabel("Sport")
    plt.ylabel("Stress level")
    
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"figures/stress_sport_gender_{save_suffix}.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    main()