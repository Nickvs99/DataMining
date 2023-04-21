import matplotlib.pyplot as plt
import numpy as np

from df_util import normalize_df
import errors

from evaluators.category_evaluator import CategoryEvaluator
from evaluators.numerical_evaluator import NumericalEvaluator

from predictors.knn_predictor import KnnPredictor
from predictors.linear_regression_predictor import LinearRegressionPredictor
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

    linear_regression_research(df, target)

def k_fold_research(df, target):

    k_fold_values = range(2, 100)

    scores, errors = [], []
    for k_fold in k_fold_values:
        print(k_fold)
        predictor = NaiveBayesPredictor(target)
        evaluator = CategoryEvaluator(target, predictor)

        validator = KFoldValidator(df, evaluator, predictor, n_folds=k_fold)
        
        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"K fold results: {list(zip(k_fold_values, scores))}")
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

    print(f"N categories results: {list(zip(n_category_values, scores))}")
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
        print(k)

        predictor = KnnPredictor(target, k=k)
        evaluator = CategoryEvaluator(target, predictor)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=10)

        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Knn k results: {list(zip(k_values, scores))}")
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
        print(n)
        predictor = KnnPredictor(target, n=n)
        evaluator = CategoryEvaluator(target, predictor)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=10)

        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    print(f"Knn n results: {list(zip(n_values, scores))}")
    plot_between(n_values, scores, errors)
    
    plt.xlabel("Distance metric")
    plt.ylabel("Score")
    plt.title("Knn research")

    plt.savefig("figures/hyper_knn_n.png")
    plt.show()


def linear_regression_research(df, target):
    
    error_functions = [errors.MAE, errors.MSE]
    labels = ["MAE", "MSE"]

    # for error_func, label in zip(error_functions, labels):

    # Numerical prediction
    predictor = LinearRegressionPredictor("Stress level")
    predictor.error_func = errors.MSE

    evaluator = NumericalEvaluator("Stress level", predictor)
    evaluator.error_func = errors.MAE

    validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
    score, std_error = validator.validate()
    print(f"Numerical error: {score}")

def plot_between(x, y, yerror):

    x, y, yerror = np.array(x), np.array(y), np.array(yerror)


    plt.plot(x, y)
    plt.fill_between(x, y -  yerror, y + yerror, alpha=0.2)


