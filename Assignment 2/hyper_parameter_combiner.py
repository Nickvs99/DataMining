import matplotlib.pyplot as plt
import numpy as np

import IO
from evaluators.recommender_evaluator import RecommenderEvaluator
from predictors.correlation_predictor import CorrelationPredictor, prepare_df_correlation
from predictors.svd_predictor import SVDPredictor, prepare_df_svd

from validators.k_fold_validator import KFoldValidator

from recommender_combiner import RecommenderCombiner, prepare_df_combiner
from recommender_scorer import RecommenderScorer

from logger import logger

def main():

    input_path = "data/training_set_10000.csv"
    target = "weighted_relevance"
    n_folds = 5

    # TODO add hyperparameters for each predictor for final results
    # (predictor, df_preparation, output_path)
    predictors = [
        (SVDPredictor(target, "srch_id", "prop_id", row_similarity_attributes=["srch_adults_count", "srch_children_count", "srch_room_count"]), prepare_df_svd, "output/svd.csv"),
        (CorrelationPredictor(target, drop_attributes=["relevance"]), prepare_df_correlation, "output/correlation.csv"),
    ]

    logger.status("Computing recommendations for each predictor ")
    for predictor, prepare_df_func, output_path in predictors:

        df = prepare_df_func(input_path)
        evaluator = RecommenderEvaluator("prop_id", predictor, "srch_id", "relevance")
        validator = KFoldValidator(df, evaluator, predictor, n_folds=n_folds)
        validator.validate(output_path=output_path, overwrite_output=False)

    df_combiner = prepare_df_combiner(input_path)
    output_paths = [predictor_tpl[2] for predictor_tpl in predictors]
    
    temp_output_path = "output/temp.csv"
    weight_values = np.arange(0, 1.01, 0.2)
    score_values = np.empty(len(weight_values))
    error_values = np.empty(len(weight_values))

    logger.status("Computing score for each weight value")
    for i, weight_value in enumerate(weight_values):

        weights = [weight_value, 1 - weight_value]
        scores = np.empty(n_folds)

        for j in range(1, n_folds + 1):
            
            logger.progress(f"Weight: {weight_value}, fold: {j - 1}")

            combiner_inputs = [IO.get_path_pattern(path) % j for path in output_paths]

            combiner = RecommenderCombiner(combiner_inputs, weights=weights)
            combiner.combine(output_path=temp_output_path)

            scorer = RecommenderScorer(temp_output_path, df_combiner, "relevance")
            scores[j - 1] = scorer.score()

        score_values[i] = np.mean(scores)
        error_values[i] = np.std(scores)

    plot_between(weight_values, score_values, error_values)
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.savefig("figures/hyper/combiner_weights.png")
    plt.show()

    max_score_index = list(score_values).index(np.max(score_values))
    best_weight = weight_values[max_score_index]
    best_score = score_values[max_score_index]
    best_error = error_values[max_score_index]

    logger.info(f"Best weight {best_weight} with a score of {best_score} +- {best_error}")

def plot_between(x, y, yerror):

    plt.plot(x, y)
    plt.fill_between(x, y -  yerror, y + yerror, alpha=0.2)

if __name__ == "__main__":
    main()