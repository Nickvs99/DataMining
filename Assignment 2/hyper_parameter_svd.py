import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from evaluators.recommender_evaluator import RecommenderEvaluator
from predictors.svd_predictor import SVDPredictor, prepare_df_svd
from validators.k_fold_validator import KFoldValidator

import logging
from logger import logger

def main():

    df = prepare_df_svd("data/training_set.csv")

    predictor_target = "weighted_relevance"
    evaluator_target = "relevance"
    row_attribute = "srch_id"
    column_attribute = "prop_id"
    similarity_attributes = ["srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count"]

    optimal_weights = optimize_weights_NM(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes)
    # optimal_weights = []
    optimal_n_clusters = optimize_n_clusters(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes, optimal_weights)
    optimal_n_svd_dimensions = optimize_n_svd_dimensions(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes, optimal_weights, optimal_n_clusters)

    logger.info(f"Optimal {optimal_weights}, {optimal_n_clusters}, {optimal_n_svd_dimensions}")

def set_df_types(df):
    
    categorical_columns = [
        "srch_id",
        "prop_id",
        "click_bool",
        "booking_bool",
    ]      

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df

def optimize_weights(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes):

    def get_initial_weights():
        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute, row_similarity_attributes=similarity_attributes)
        predictor.training_df = df
        return predictor.compute_row_similarity_weights()

    def validate_svd_recommender(similarity_weights):
        
        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute,
                            row_similarity_attributes=similarity_attributes,
                            row_similarity_weights=similarity_weights)
        
        evaluator = RecommenderEvaluator(evaluator_target, predictor, predictor.row_attribute)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
        score, std_error = validator.validate()

        logger.setLevel(logging.PROGRESS)
        logger.progress(f"Score: {score} +- {std_error}. weights: {similarity_weights}")
        logger.setLevel(logging.INFO)

        return -score

    initial_weights = get_initial_weights()
    # validate_svd_recommender(initial_weights)

    response = scipy.optimize.minimize(validate_svd_recommender, initial_weights)

    optimal_weights = response.x

    validate_svd_recommender(optimal_weights)

    return optimal_weights

def optimize_weights_NM(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes):
    
    def get_initial_weights():
        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute, row_similarity_attributes=similarity_attributes)
        predictor.training_df = df
        return predictor.compute_row_similarity_weights()
    
    def get_initial_complex(initial_weights):
            
            initial_weights = initial_weights / 2
            N = len(initial_weights)

            simplex = np.empty((N + 1, N), dtype=initial_weights.dtype)
            simplex[0] = initial_weights
            for k in range(N):
                y = np.array(initial_weights, copy=True)
                if y[k] != 0:
                    y[k] = 4*y[k]
                else:
                    y[k] = 0.001
                
                simplex[k + 1] = y

            return simplex

    def validate_svd_recommender(similarity_weights):
        
        similarity_weights[similarity_weights < 0] = 0

        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute,
                            row_similarity_attributes=similarity_attributes,
                            row_similarity_weights=similarity_weights)
        
        evaluator = RecommenderEvaluator(column_attribute, predictor, row_attribute, evaluator_target)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
        score, std_error = validator.validate()

        logger.setLevel(logging.PROGRESS)
        logger.progress(f"Score: {score} +- {std_error}. weights: {similarity_weights}")
        logger.setLevel(logging.INFO)

        return -score

    logger.status("Starting optimization procedure")
    logger.setLevel(logging.INFO)

    initial_weights = get_initial_weights()

    validate_svd_recommender(initial_weights)

    initial_simplex = get_initial_complex(initial_weights)

    response = scipy.optimize.fmin(validate_svd_recommender, initial_weights, initial_simplex=initial_simplex, full_output=True, maxfun=50)

    initial_score = validate_svd_recommender(initial_weights)
    optimal_weights, minimum = response[0], response[1]
    
    logger.info(f"Initial score: {-initial_score}. weights: {initial_weights}")
    logger.info(f"Optimized score: {-minimum}. weights: {optimal_weights}")

    return optimal_weights


def plot_between(x, y, yerror):

    x, y, yerror = np.array(x), np.array(y), np.array(yerror)

    plt.plot(x, y)
    plt.fill_between(x, y -  yerror, y + yerror, alpha=0.2)

    
def optimize_n_clusters(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes, similarity_weights, show=False):

    cluster_sizes = range(2, 20)

    scores, errors = [], []
    for n_clusters in cluster_sizes:
        logger.setLevel(logging.PROGRESS)
        logger.progress(f"Current cluster size {n_clusters}")
        logger.setLevel(logging.INFO)

        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute, row_similarity_attributes=similarity_attributes, row_similarity_weights=similarity_weights,
                n_clusters=n_clusters)

        evaluator = RecommenderEvaluator(column_attribute, predictor, row_attribute, evaluator_target)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    plot_between(cluster_sizes, scores, errors)
    
    plt.xlabel("N clusters")
    plt.ylabel("Score")
    plt.title("N cluster research")

    plt.savefig("figures/hyper/n_clusters.png")

    if show:
        plt.show()
    else:
        plt.close()

    best_score_index = scores.index(max(scores))
    best_n_clusters = cluster_sizes[best_score_index]
    best_score = scores[best_score_index]
    best_error = errors[best_score_index]

    logger.info(f"Best N clusters: {best_n_clusters}. Score {best_score} +- {best_error}")
    return best_n_clusters

    
def optimize_n_svd_dimensions(df, predictor_target, evaluator_target, row_attribute, column_attribute, similarity_attributes, similarity_weights, n_clusters, show=False):

    svd_dimension_values = range(1, n_clusters)
    scores, errors = [], []
    for n_svd_dimensions in svd_dimension_values:
        logger.setLevel(logging.PROGRESS)
        logger.progress(f"Current SVD dimensions {n_svd_dimensions}")
        logger.setLevel(logging.INFO)

        predictor = SVDPredictor(predictor_target, row_attribute, column_attribute, 
                                row_similarity_attributes=similarity_attributes,
                                row_similarity_weights=similarity_weights,
                                n_clusters=n_clusters,
                                n_svd_dimensions=n_svd_dimensions)

        evaluator = RecommenderEvaluator(column_attribute, predictor, row_attribute, evaluator_target)
        validator = KFoldValidator(df, evaluator, predictor, n_folds=2)
        score, std_error = validator.validate()

        scores.append(score)
        errors.append(std_error)

    plot_between(svd_dimension_values, scores, errors)
    
    plt.xlabel("N SVD dimensions")
    plt.ylabel("Score")
    plt.title("SVD research")

    plt.savefig("figures/hyper/n_svd_dimensions.png")
    
    if show:
        plt.show()
    else:
        plt.close()

    best_score_index = scores.index(max(scores))
    best_n_svd_dimensions = svd_dimension_values[best_score_index]
    best_score = scores[best_score_index]
    best_error = errors[best_score_index]
    
    logger.info(f"Best N SVD dimensions {best_n_svd_dimensions}. Score {best_score} +- {best_error}")
    return best_n_svd_dimensions

if __name__ == "__main__":
    main()
