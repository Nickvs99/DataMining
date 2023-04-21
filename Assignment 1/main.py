import numpy as np
import pandas as pd

from column_selection import select_columns
from data_cleaning import clean_df
from data_exploration import explore_df
from df_util import print_header
import errors
from hyper_parameters import run_hyper_parameter_research
from feature_engineering import run_feature_engineering

from evaluators.category_evaluator import CategoryEvaluator
from evaluators.numerical_evaluator import NumericalEvaluator

from predictors.knn_predictor import KnnPredictor
from predictors.linear_regression_predictor import LinearRegressionPredictor
from predictors.naive_bayes_predictor   import NaiveBayesPredictor

from validators.basic_validator import BasicValidator
from validators.k_fold_validator import KFoldValidator


def main():
    
    df = pd.read_csv("data.csv", sep=",")
    
    column_name_map = update_column_names(df)
    set_df_types(df)

    df = df.drop(columns=["Timestamp", "Birthday"])

    # Randomize order of rows, https://stackoverflow.com/a/34879805/12132063
    # Use a random_state to have the same order between runs
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    df_clean_remove = clean_df(df, method="remove")
    df_clean_replace = clean_df(df, method="replace")
    
    df_clean_remove = run_feature_engineering(df_clean_remove)
    df_clean_replace = run_feature_engineering(df_clean_replace)

    # # Explore data, which gets basic data plots and tables
    # save_suffixs = ["dirty", "remove", "replace"]
    # dfs = [df, df_clean_remove, df_clean_replace]
    # for dataframe, save_suffix in zip(dfs, save_suffixs):
    #     explore_df(dataframe, column_name_map, save_suffix=save_suffix)

    # Split df into a test and train set
    df = df_clean_replace.copy()
    n_rows = len(df.index)

    test_fraction = 1/3
    n_test_rows = int(n_rows * test_fraction)

    other_df = df[n_test_rows:]

    # # Select columns which perform the best with a given validator
    # target = "Gender"
    # predictor = NaiveBayesPredictor(target, n_category_bins=5)
    # evaluator = CategoryEvaluator(target, predictor)
    # validator = KFoldValidator(other_df, evaluator, predictor, n_folds=10) 
    # columns = select_columns(df_clean_replace, validator, mandatory=["Gender", "Stress level"], prefered=["Sport"])
    columns = ["Gender", "Stress level", "Sport"]
    
    print(f"Selected columns: {columns}")
    df = df[columns].copy()
    
    test_df = df[:n_test_rows]
    other_df = df[n_test_rows:]

    run_hyper_parameter_research(other_df, "Gender", "Stress level")



def set_df_types(df):

    numeric_columns = [
        "Students estimate",
        "Stress level",
        "Sport",
        "Random number",
    ]

    categorical_columns = [
        "Program",
        "Machine learning",
        "Information retrieval",
        "Statistics",
        "Databases",
        "Gender",
        "ChatGPT",
        "Stand up",
        "Good day (#1)",
        "Good day (#2)",
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


if __name__ == "__main__":
    main()