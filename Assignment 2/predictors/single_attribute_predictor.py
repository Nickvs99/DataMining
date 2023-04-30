

from predictors.numerical_predictor import NumericalPredictor

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
        