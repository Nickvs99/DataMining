from predictors.predictor import Predictor

class CategoryPredictor(Predictor):

    def __init__(self, target, training_df):

        super().__init__(target, training_df)

        if not training_df[target].dtype.name == 'category':
            raise Exception(f"{target} is not a categorical column.")
