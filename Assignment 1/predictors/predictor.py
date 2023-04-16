

class Predictor:

    def __init__(self, target):

        self.target = target

    def train(self, training_df):
        
        self.training_df = training_df

        if self.target not in self.training_df.columns:
            raise Exception(f"Invalid target. Possible values are {self.training_df.columns}.")


    def predict(self, entity):
        raise NotImplementedError()
