

class Predictor:

    def __init__(self, target, training_df):

        self.target = target
        self.training_df = training_df

        if target not in self.training_df.columns:
            raise Exception(f"Invalid target. Possible values are {self.training_df.columns}.")

    def predict(self, entity):
        raise NotImplementedError()
