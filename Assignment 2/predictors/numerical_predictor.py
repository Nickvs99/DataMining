from pandas.api.types import is_numeric_dtype

from predictors.predictor import Predictor

class NumericalPredictor(Predictor):

    def __init__(self, target):

        super().__init__(target)

    def train(self, training_df):
        
        super().train(training_df)
        
        if not is_numeric_dtype(training_df[self.target]):
            raise Exception(f"{self.target} is not a numerical column.")
