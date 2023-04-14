

class Evaluator:

    def __init__(self, target, predictor, test_df):

        self.target = target
        self.predictor = predictor
        self.test_df = test_df

        if self.target not in self.test_df.columns:
            raise Exception(f"Invalid target. Possible values are {self.test_df.columns}.")

    def evalutate(self):
        raise NotImplementedError()
