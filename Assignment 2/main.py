import numpy as np
import pandas as pd
import warnings

from data_exploration import explore_df, plot_attribute_correlation, show_correlation_matrix
from feature_engineering import run_feature_engineering
from logger import logger

from evaluators.recommender_evaluator import RecommenderEvaluator
from predictors.svd_predictor import SVDPredictor
from predictors.single_attribute_predictor import SingleAttributePredictor
from validators.basic_validator import BasicValidator

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    logger.status("Reading data into dataframe")

    # TEMP solution. Remove date_time from the df since the full dataset consumes more memory than my laptop has :(
    usecols = ["srch_id", "prop_id", "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count", "click_bool", "booking_bool", "position", "prop_review_score"]
    # columns_to_skip = ["date_time", "prop_location_score2", "position", "srch_destination_id", "srch_adults_count", "srch_children_count", "srch_room_count", "random_bool", "click_bool", "booking_bool"]
    # for i in range(1, 9):
    #     columns_to_skip.append(f"comp{i}_rate")
    #     columns_to_skip.append(f"comp{i}_inv")
    #     columns_to_skip.append(f"comp{i}_rate_percent_diff")

    df = pd.read_csv("data/training_set_100000.csv", sep=",", usecols=usecols)
    df = set_df_types(df)

    # explore_df(df, save_suffix="test", show=False)

    df = run_feature_engineering(df)
    # df.drop(columns=["position", "gross_bookings_usd"], inplace=True)

    # plot_attribute_correlation(df, "relevance", save_suffix=None, show=True)
    # plot_attribute_correlation(df, "weighted_relevance", save_suffix="", show=True)    

    # The targets for the predictor and evaluator are different, this is intentional.
    # The predictor is trained on the weighted relevance since it removes any position bias
    # But relevance is needed to compute the actual prediction power
    n_validation_rows = int(len(df.index) * 0.2)

    validation_df = df[:n_validation_rows]
    training_df = df[n_validation_rows:]

    # predictor = SingleAttributePredictor("weighted_relevance", "prop_review_score")
    predictor = SVDPredictor("weighted_relevance", "srch_id", "prop_id", row_similarity_attributes=["srch_adults_count", "srch_children_count", "srch_room_count"])
    
    evaluator = RecommenderEvaluator("prop_id", predictor, "srch_id", "relevance")
    predictor.train(training_df)
    evaluator.write_recommendations(validation_df, output_path="output/svd.csv")
    
    # validator = BasicValidator(df, evaluator, predictor)
    # score, std_error = validator.validate()
    # logger.info(f"Score: {score} +- {std_error}")

    # # Compute prediction for test set, test set has 4 less columns than training set
    # logger.status("Computing recommendations for test set ")
    # test_df = pd.read_csv("data/test_set.csv", sep=",", usecols=usecols[:-4])
    # predictor.train(df)
    # evaluator = RecommenderEvaluator(target, predictor, "srch_id")
    # evaluator.write_kaggle_prediction(test_df, output_path="data/output.csv")


def set_df_types(df):

    # Most types are automatically set by pandas, however it does not
    # detect boolean/categorical columns

    categorical_columns = [
        "srch_id",
        # "site_id",
        # "visitor_location_country_id",
        # "prop_country_id",
        # "prop_id",
        # "prop_brand_bool",
        # "srch_destination_id",
        # "srch_saturday_night_bool",
        # "random_bool",
        # "click_bool",
        # "booking_bool",
        # "promotion_flag"
    ]   

    # for i in range(1, 9):
    #     categorical_columns.append(f"comp{i}_rate")
    #     categorical_columns.append(f"comp{i}_inv")

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df

if __name__ == "__main__":
    main()