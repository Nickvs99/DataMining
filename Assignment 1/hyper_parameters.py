import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from df_util import normalize_df, print_header, save_df,  print_df
import errors

from evaluators.category_evaluator import CategoryEvaluator
from evaluators.numerical_evaluator import NumericalEvaluator

from predictors.knn_predictor import KnnPredictor
from predictors.regression_predictor import RegressionPredictor
from predictors.naive_bayes_predictor   import NaiveBayesPredictor

from validators.basic_validator import BasicValidator
from validators.k_fold_validator import KFoldValidator

def run_hyper_parameter_research(df, cat_target, num_target):
    
    run_categorical_research(df, cat_target)
    run_numerical_research(df, num_target)

def run_categorical_research(df, target):

    df = normalize_df(df)

    k_fold_research(df, target)
    naive_bayes_research(df, target)
    knn_research(df, target)

def run_numerical_research(df, target):

    regression_research(df, target)

def k_fold_research(df, target):

    k_fold_values = range(2, 100)

    scores, errors = [], []
    for k_fold in k_fold_values:

        predictor = NaiveBayesPredictor(target)
        evaluator = CategoryEvaluator(target, predictor)

        validator = KFoldValidator(df, evaluator, predictor, n_folds=k_fold)
        
        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Kfold - Best number of folds: {k_fold_values[scores.index(max(scores))]}")
    plot_between(k_fold_values, scores, errors)
    
    plt.xlabel("K folds")
    plt.ylabel("Score")
    plt.title("K fold research")

    plt.savefig("figures/hyper_kfold.png")
    plt.show()

def naive_bayes_research(df, target):

    n_category_values = range(1, 50)

    scores, errors = [], []
    for n_categories in n_category_values:

        predictor = NaiveBayesPredictor(target, n_category_bins=n_categories)
        evaluator = CategoryEvaluator(target, predictor)

        validator = KFoldValidator(df, evaluator, predictor, n_folds=10)

        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Naive bayes - Best number of categories: {n_category_values[scores.index(max(scores))]}")
    plot_between(n_category_values, scores, errors)
    
    plt.xlabel("N categories")
    plt.ylabel("Score")
    plt.title("Naive bayes research")

    plt.savefig("figures/hyper_bayes_ncategories.png")
    plt.show()

def knn_research(df, target):

    knn_research_k(df, target)
    knn_research_n(df, target)

def knn_research_k(df, target):
    
    k_values = range(1, 50)

    scores, errors = [], []
    for k in k_values:

        predictor = KnnPredictor(target, k=k)
        evaluator = CategoryEvaluator(target, predictor)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=10)

        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Knn - Best number of neighbours: {k_values[scores.index(max(scores))]}")
    plot_between(k_values, scores, errors)
    
    plt.xlabel("K neighbours")
    plt.ylabel("Score")
    plt.title("Knn research")

    plt.savefig("figures/hyper_knn_k.png")
    plt.show()

def knn_research_n(df, target):
    
    n_values = np.arange(1, 10, 0.5)

    scores, errors = [], []
    for n in n_values:

        predictor = KnnPredictor(target, n=n)
        evaluator = CategoryEvaluator(target, predictor)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=10)

        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Knn - Best distance metric: {n_values[scores.index(max(scores))]}")
    plot_between(n_values, scores, errors)
    
    plt.xlabel("Distance metric")
    plt.ylabel("Score")
    plt.title("Knn research")

    plt.savefig("figures/hyper_knn_n.png")
    plt.show()


def regression_research(df, target):
    
    regression_functions = [
        ("wx", lambda x, w:  w * x),
        ("x/w", lambda x, w: x/w),
        ("wx^2", lambda x, w: w * x**2),
        ("wx^0.5", lambda x, w: w * x**0.5),
        ("w^x", lambda x, w: w**x),
        ("e^wx", lambda x, w: np.e ** (w* x))
    ]

    error_functions = [
        ("MSE", errors.MSE),
        ("MAE", errors.MAE),
    ]

    results_df = pd.DataFrame(
        index = [regression_function[0] for regression_function in regression_functions],
        columns=[error_function[0] for error_function in error_functions]
    )

    for error_label, error_func in error_functions:
        for regression_label, regression_func in regression_functions:


            predictor = RegressionPredictor(target)
            predictor.error_func = error_func
            predictor.regression_func = regression_func

            evaluator = NumericalEvaluator(target, predictor)
            evaluator.error_func = error_func

            validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
            score, std_error = validator.validate()

            results_df.loc[regression_label, error_label] = f"{score:.2f} +- {std_error:.2f}"

    print_header("Regression relsults")
    print_df(results_df)
    save_df(results_df, "tables/regression_results.tex")

def plot_between(x, y, yerror):

    x, y, yerror = np.array(x), np.array(y), np.array(yerror)


    plt.plot(x, y)
    plt.fill_between(x, y -  yerror, y + yerror, alpha=0.2)


