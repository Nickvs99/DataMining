import numpy as np
import pandas as pd
import warnings

from data_exploration import explore_df, plot_relevance_correlation
from feature_engineering import run_feature_engineering

from evaluators.recommender_evaluator import RecommenderEvaluator
from predictors.single_attribute_predictor import SingleAttributePredictor
from validators.basic_validator import BasicValidator

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # TEMP solution. Remove date_time from the df since the full dataset consumes more memory than my laptop has :(
    usecols = [i for i in range(54) if i != 1] 
    df = pd.read_csv("data/training_set_100000.csv", sep=",", usecols=usecols)

    df = set_df_types(df)

    # explore_df(df, save_suffix="", show=False)

    df = run_feature_engineering(df)

    # plot_relevance_correlation(df, save_suffix="", show=False)    

    target = "relevance"
    predictor = SingleAttributePredictor(target, "prop_review_score")
    evaluator = RecommenderEvaluator(target, predictor, "srch_id")
    validator = BasicValidator(df, evaluator, predictor)
    score, std_error = validator.validate()
    print(f"{score} +- {std_error}")


    # Compute prediction for test set, test set has 4 less columns than training set
    test_df = pd.read_csv("data/test_set.csv", sep=",", usecols=usecols[:-4])
    predictor.train(df)
    evaluator = RecommenderEvaluator(target, predictor, "srch_id")
    evaluator.write_kaggle_prediction(test_df, output_path="data/output.csv")


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