import numpy as np
import pandas as pd

from column_selection import select_columns
from data_cleaning import clean_df
from data_exploration import explore_df
from df_util import print_header, save_df, print_df, normalize_df
import errors
from hyper_parameters import run_hyper_parameter_research
from feature_engineering import run_feature_engineering

from evaluators.category_evaluator import CategoryEvaluator
from evaluators.numerical_evaluator import NumericalEvaluator

from predictors.knn_predictor import KnnPredictor
from predictors.regression_predictor import RegressionPredictor
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

    # Select columns which perform the best with a given validator
    # target = "Gender"
    # predictor = NaiveBayesPredictor(target, n_category_bins=5)
    # evaluator = CategoryEvaluator(target, predictor)
    # validator = KFoldValidator(other_df, evaluator, predictor, n_folds=10) 
    # columns = select_columns(df_clean_replace, validator, mandatory=["Gender", "Stress level"], prefered=["Sport", "Sleep level"])
    columns = ["Gender", "Stress level", "Sport"]
    
    print(f"Selected columns: {columns}")
    df = df[columns].copy()
    
    other_df = df[n_test_rows:]
    run_hyper_parameter_research(other_df, "Gender", "Stress level")
    
    # Compute final numerical results
    error_functions = [
        ("MSE", errors.MSE),
        ("MAE", errors.MAE),
    ]
    target = "Stress level"

    for error_label, error_func in error_functions:
        predictor = RegressionPredictor(target)
        predictor.error_func = error_func
        predictor.regression_func = lambda x, w: w * x**0.5

        evaluator = NumericalEvaluator(target, predictor)
        evaluator.error_func = error_func

        validator = BasicValidator(df, evaluator, predictor, validate_fraction=1/3)
        score, std_error = validator.validate()
        
        weight_df = predictor.get_weight_df()
        print_header(f"Numerical results {error_label}")
        print(f"{error_label} - Score = {score}")
        print_df(weight_df)
        save_df(weight_df, f"tables/regression_weights_{error_label}.tex")


    # Compute final categorical results
    target = "Gender"

    df = normalize_df(df)

    predictor = NaiveBayesPredictor(target, n_category_bins=14)
    evaluator = CategoryEvaluator(target, predictor)
    validator = BasicValidator(df, evaluator, predictor, validate_fraction=1/3)
    score, std_error = validator.validate()
    confusion_df = validator.compute_confusion_df()
    print_header("Confusion matrix - Naive bayes")
    print(f"Score: {score} +- {std_error}")
    print_df(confusion_df)
    save_df(confusion_df, "tables/confusion_naive_bayes.tex")

    predictor = KnnPredictor(target, k=9, n=4)
    evaluator = CategoryEvaluator(target, predictor)
    validator = BasicValidator(df, evaluator, predictor, validate_fraction=1/3)
    score, std_error = validator.validate()
    confusion_df = validator.compute_confusion_df()
    print_header("Confusion matrix - Knn")
    print(f"Score: {score} +- {std_error}")
    print_df(confusion_df)
    save_df(confusion_df, "tables/confusion_knn.tex")



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