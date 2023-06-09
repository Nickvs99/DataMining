import numpy as np
import pandas as pd
import warnings

from data_exploration import explore_df, plot_attribute_correlation, show_correlation_matrix
import feature_engineering as fe
from logger import logger

from evaluators.recommender_evaluator import RecommenderEvaluator
from predictors.svd_predictor import SVDPredictor
from predictors.single_attribute_predictor import SingleAttributePredictor
from predictors.correlation_predictor import CorrelationPredictor, prepare_df_correlation
from validators.basic_validator import BasicValidator

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():

    df = prepare_df_correlation("data/training_set_100000.csv")
    
    # explore_df(df, save_suffix="test", show=False)
    
    # The targets for the predictor and evaluator are different, this is intentional.
    # The predictor is trained on the weighted relevance since it removes any position bias
    # But relevance is needed to compute the actual prediction power

    # predictor = SingleAttributePredictor("weighted_relevance", "prop_review_score")
    # predictor = SVDPredictor("weighted_relevance", "srch_id", "prop_id", row_similarity_attributes=["srch_adults_count", "srch_children_count", "srch_room_count"])
    predictor = CorrelationPredictor("weighted_relevance", drop_attributes=["relevance"])
    predictor.fill_func = lambda series: series.median()
    # predictor.fill_func = lambda series: series.mean()

    evaluator = RecommenderEvaluator("prop_id", predictor, "srch_id", "relevance")
    validator = BasicValidator(df, evaluator, predictor)
    score, std_error = validator.validate()
    logger.info(f"Score: {score} +- {std_error}")
    # validator = BasicValidator(df, evaluator, predictor)
    # score, std_error = validator.validate()
    # logger.info(f"Score: {score} +- {std_error}")

    # # Compute prediction for test set, test set has 4 less columns than training set
    # logger.status("Computing recommendations for test set ")
    # test_df = pd.read_csv("data/test_set.csv", sep=",", usecols=usecols[:-4])
    # predictor.train(df)
    # evaluator = RecommenderEvaluator(target, predictor, "srch_id")
    # evaluator.write_kaggle_prediction(test_df, output_path="data/output.csv")

def prepare_df_all(input_path):
    
    logger.status("Reading data into dataframe")

    # Use all except the datetime
    usecols = [i for i in range(54) if i != 1]
    df = pd.read_csv(input_path, sep=",", usecols=usecols)
    df = set_df_types(df)

    df = fe.add_relevance_column(df)
    df = fe.add_weighted_relevance_column(df)

    return df

def set_df_types(df):

    # Most types are automatically set by pandas, however it does not
    # detect boolean/categorical columns

    categorical_columns = [
        "srch_id",
        "site_id",
        "visitor_location_country_id",
        "prop_country_id",
        "prop_id",
        "prop_brand_bool",
        "srch_destination_id",
        "srch_saturday_night_bool",
        "random_bool",
        "click_bool",
        "booking_bool",
        "promotion_flag"
    ]   

    for i in range(1, 9):
        categorical_columns.append(f"comp{i}_rate")
        categorical_columns.append(f"comp{i}_inv")

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df

if __name__ == "__main__":
    main()