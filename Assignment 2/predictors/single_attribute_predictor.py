
import feature_engineering as fe
from predictors.numerical_predictor import NumericalPredictor

from logger import logger

class SingleAttributePredictor(NumericalPredictor):
    """
    Dummy predictor. This predictor simply returns the value of the assigned attribute.
    """

    def __init__(self, target, attribute):
        super().__init__(target)

        self.attribute = attribute

    def train(self, training_df):
        super().train(training_df)

    def predict(self, entity):
        return entity[self.attribute]


def prepare_df_single_attribute(input_path, attribute):

    logger.status("Reading data into dataframe")

    usecols = ["position", "click_bool", "booking_bool", "srch_id", "prop_id", attribute]
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
        "prop_id",
        "random_bool",
        "click_bool",
        "booking_bool",
    ]   

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df   