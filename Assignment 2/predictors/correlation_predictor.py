import numpy as np
import pandas as pd

import feature_engineering as fe

from predictors.numerical_predictor import NumericalPredictor
from logger import logger

class CorrelationPredictor(NumericalPredictor):
    """
    Attempts to predict the target value based on the correlation against other attributes.
    This by itself will not give a good prediction, however it can be used to rank
    the outcomes.
    """

    def __init__(self, target, drop_attributes=[]):
        super().__init__(target)
        self.drop_attributes = drop_attributes

    def train(self, training_df):
        logger.status("Training correlation predictor")
        
        super().train(training_df)

        df = self.training_df.copy()

        logger.progress("Converting categorical attributes to numerical attributes")
        # https://stackoverflow.com/a/32011969/12132063
        categorical_columns = df.select_dtypes(['category']).columns
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

        logger.progress("Computing normalizing weights")
        # Stores the weight for each attribute. The weights normalize all attributes
        self.weight_map = {}
        for column in df.columns:
            self.weight_map[column] = 1 / df[column].max()
        
        logger.progress("Computing correlation")
        self.corr = df.corrwith(df[self.target])
        self.corr.drop(labels=[self.target] + self.drop_attributes, inplace=True)

    def predict(self, entity):

        total = 0
        for key, value in entity.items():

            if pd.isna(value) or key not in self.corr:
                continue

            total += value * self.weight_map[key] * self.corr[key]

        return total


def prepare_df_correlation(input_path):
    
    logger.status("Reading data into dataframe")

    # Use all except the datetime
    usecols = [i for i in range(54) if i != 1]
    df = pd.read_csv(input_path, sep=",", usecols=usecols)
    df = set_df_types(df)

    df = fe.add_relevance_column(df)
    df = fe.add_weighted_relevance_column(df)

    # Drop training and feature engineered columns
    df.drop(columns=["position", "gross_bookings_usd", "click_bool", "booking_bool"], inplace=True)

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
