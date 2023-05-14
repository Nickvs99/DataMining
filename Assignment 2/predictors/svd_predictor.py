import bisect
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


from predictors.numerical_predictor import NumericalPredictor

from logger import logger

class SVDPredictor(NumericalPredictor):
    """
    A predictor based on Singular Value Decomposition. 
    
    The most similair row is used for entities who are not present in the training df 
    (which are all entities for this assignment).

    The similarity is based on the row_similarity_attributes and weighted
    through the row_similarity_weights. If no weights are present, then the 
    weights are chosed such that the values are normalized, where the max value
    of a column is 1.
    """


    def __init__(self, target, row_attribute, column_attribute,
                 row_similarity_attributes=[],
                 row_similarity_weights=[],
                 distance_metric=1,
                 n_svd_dimensions=10):
        
        super().__init__(target)

        self.row_attribute = row_attribute
        self.column_attribute = column_attribute

        self.row_similarity_attributes = row_similarity_attributes
        self.row_similarity_weights = row_similarity_weights
        self.distance_metric = distance_metric
        self.n_svd_dimensions = n_svd_dimensions

        if len(self.row_similarity_weights) != 0 and len(self.row_similarity_weights) != len(self.row_similarity_attributes):
            logger.critical(f"The dimensions of row_similarity_weights and row_similarity_attributes do not match. {len(self.row_similarity_weights)} vs {len(self.row_similarity_attributes)}")
            exit()

        # Stores the computed similarity scores for each row_attribute during prediction
        self.similarity_scores_cache = {}

    def train(self, training_df):
        logger.status("Training SVD predictor")

        super().train(training_df)

        if not self.check_valid_attributes():
            logger.critical("Training aborted due to one or more invalid attributes")
            exit()

        logger.status("Creating similarity df")
        self.similarity_df = self.get_similarity_df()

        # Set weights if they are not set
        if len(self.row_similarity_weights) == 0:
            self.row_similarity_weights = self.compute_row_similarity_weights()

        # Sort the labels, this allows for binary search later on, instead of sequential search
        self.row_labels = np.sort(self.training_df[self.row_attribute].unique()).tolist()        
        self.column_labels = np.sort(self.training_df[self.column_attribute].unique()).tolist()

        logger.status("Composing u, s, vT matrices")
        self.u_matrix, s, vT_matrix = self.compute_svd()
        self.s_matrix = np.diag(s)
        self.v_matrix = vT_matrix.transpose()

    def predict(self, entity):
        
        row_attribute_value = entity[self.row_attribute].values[0]
        column_attribute_value = entity[self.column_attribute].values[0]
        
        # Select the interesting attributes from the entity
        attributes = self.row_similarity_attributes
        entity = entity[attributes] 
        
        if row_attribute_value in self.similarity_scores_cache:
            similarity_scores = self.similarity_scores_cache[row_attribute_value]
        else:
            similarity_scores = self.compute_similarity_scores(entity)
            self.similarity_scores_cache[row_attribute_value] = similarity_scores

        max_similarity_index = similarity_scores.index(max(similarity_scores))
        column_attribute_index = self.index_sorted_list(self.column_labels, column_attribute_value)

        # Return zero if the column attribute value was not found, this occurs when a prediction is made
        # for a column value not in the training data.
        if column_attribute_index == -1:
            return 0

        prediction = self.u_matrix[max_similarity_index] @ self.s_matrix @ self.v_matrix[column_attribute_index]

        return prediction

    def check_valid_attributes(self):
        """
        Checks if row_attribute, column_attribute and all row_similarity_attributes are present
        in the training_df.
        """
        
        valid = True
        attributes = [self.row_attribute, self.column_attribute] + self.row_similarity_attributes

        for attribute in attributes:
            
            if attribute not in self.training_df:
                logger.error(f"{attribute} is not a column of the training df")
                valid = False

        return valid

    def compute_row_similarity_weights(self):

        weights = []
        for similarity_attribute in self.row_similarity_attributes:

            if is_numeric_dtype(self.similarity_df[similarity_attribute].dtype):
                max_attribute_value = self.similarity_df[similarity_attribute].max()
                weight = 1/max_attribute_value
            else:
                weight = 1
            
            weights.append(weight)

        return weights

    def compute_svd(self):
        
        target_matrix = self.get_target_matrix(self.row_labels, self.column_labels)

        logger.progress("Decomposing target matrix into u, s, vT")
        return svds(target_matrix, k=self.n_svd_dimensions)

    def get_target_matrix(self, row_labels, column_labels):
        """
        Creates a matrix with the target values as cell values. Each row is index through
        the row_labels and each column is indexed through the column labels.
        """

        shape = (len(row_labels), len(column_labels))

        rows = []
        columns = []
        data = []

        log_per_nsteps = int(min(len(self.training_df.index) / 10, 100000))

        for i, (index, row) in enumerate(self.training_df.iterrows()):
            
            if i % log_per_nsteps == 0:
                logger.progress(f"Filling target matrix {i/len(self.training_df.index) * 100:.2f}%")

            row_attribute_value = row[self.row_attribute]
            column_attribute_value = row[self.column_attribute]
            target_value = row[self.target]

            row_label_index = self.index_sorted_list(row_labels, row_attribute_value)
            column_label_index = self.index_sorted_list(column_labels, column_attribute_value)

            rows.append(row_label_index)
            columns.append(column_label_index)
            data.append(target_value)

        logger.progress(f"Finalizing target matrix")

        return csr_matrix((data, (rows, columns)), shape=shape)

    def index_sorted_list(self, sorted_list, element):
        i = bisect.bisect_left(sorted_list, element)
        if i != len(sorted_list) and sorted_list[i] == element:
            return i
        return -1

    def get_similarity_df(self):
        """
        Creates an df with all the similarity attributes, duplicates are not added.
        It's index is the row_attribute
        """

        row_attribute_values_used = set()
        
        attributes = [self.row_attribute] + self.row_similarity_attributes
       
        log_per_nsteps = int(min(len(self.training_df.index) / 10, 10000))

        # Remove any duplicates and select only the interesting columns
        similarity_df = self.training_df.drop_duplicates(subset=[self.row_attribute])
        similarity_df = similarity_df[attributes]

        return similarity_df.set_index(self.row_attribute)

    def compute_similarity_scores(self, entity):

        similarity_score_series = self.similarity_df.apply(
            lambda row: self.calc_similarity_score(entity, row),
            axis=1
        )
        
        return similarity_score_series.values.tolist()
        
    def calc_similarity_score(self, entity1, entity2):
        return -self.calc_distance(entity1, entity2)

    def calc_distance(self, entity1, entity2):
        """
        Calculates the distance between two entitys.

        entity1 is a df, since a df stores information about each column
        entity2 is a series, a df could also have been used, but would require an additional computational step

        n is a hyperparameter controlling the distance measure. 
        n = 1 --> manhattan distnace
        n = 2 --> euclidian distance
        """

        distance = 0
        for i, column in enumerate(entity1.columns):

            weight = self.row_similarity_weights[i]

            # The target column should not be taken into account in the distance measure
            if column == self.target:
                continue

            value1 = entity1[column].values[0]
            value2 = entity2[column]

            # Skip if the training test instace has a nan value
            if pd.isna(value1):
                continue
            
            # Maximum penalty if the training instance has a nan value
            elif pd.isna(value2):
                distance += 1

            elif is_numeric_dtype(entity1[column].dtype):
                distance += weight * abs(value1 - value2) ** self.distance_metric
            elif entity1[column].dtype.name == 'category':
                distance += weight * (value1 != value2) ** self.distance_metric
            else:
                logger.error("Unsupported column type")

        return distance ** (1/self.distance_metric)
        