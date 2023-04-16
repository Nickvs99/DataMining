from predictors.predictor import Predictor

class CategoryPredictor(Predictor):

    def __init__(self, target):

        super().__init__(target)

    def train(self, training_df):
        
        super().train(training_df)
        
        if not training_df[self.target].dtype.name == 'category':
            raise Exception(f"{self.target} is not a categorical column.")
