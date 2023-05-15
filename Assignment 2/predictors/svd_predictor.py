import bisect
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.cluster.vq import kmeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial import distance

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
                 n_svd_dimensions=10,
                 n_clusters=10):
        
        super().__init__(target)

        self.row_attribute = row_attribute
        self.column_attribute = column_attribute

        self.row_similarity_attributes = row_similarity_attributes
        self.row_similarity_weights = row_similarity_weights
        self.n_svd_dimensions = n_svd_dimensions
        self.n_clusters = n_clusters

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

        # Set weights if they are not set
        if len(self.row_similarity_weights) == 0:
            self.row_similarity_weights = self.compute_row_similarity_weights()
        
        # Create a df with all similarity attributes, duplicates row_attributes are removed
        similarity_df = self.training_df.drop_duplicates(subset=[self.row_attribute])
        similarity_df = similarity_df[self.row_similarity_attributes]
        weighted_similarity_array = similarity_df.to_numpy() * self.row_similarity_weights
        
        # Calculate center of clusters, weighted through the similarity weights
        logger.status("Calculating cluster centroids")
        self.cluster_centroids, _ = kmeans(weighted_similarity_array, self.n_clusters)

        # Assign each row to a cluster
        cluster_assignings = self.get_cluster_assignings(weighted_similarity_array)
        
        # Sort the labels, this allows for binary search later on, instead of sequential search
        self.column_labels = np.sort(self.training_df[self.column_attribute].unique()).tolist()

        logger.status("Composing u, s, vT matrices")
        self.u_matrix, s, vT_matrix = self.compute_svd(cluster_assignings)
        self.s_matrix = np.diag(s)
        self.v_matrix = vT_matrix.transpose()

    def predict(self, entity):

        row_attribute_value = entity[self.row_attribute]
        column_attribute_value = entity[self.column_attribute]
        
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

        for attribute in self.row_similarity_attributes:
            if not is_numeric_dtype(self.training_df[attribute].dtype):
                logger.error(f"{attribute} is not a numeric type. Current implementation only works with numeric types.")
                valid = False

        return valid

    def compute_row_similarity_weights(self):

        weights = []
        for similarity_attribute in self.row_similarity_attributes:

            if is_numeric_dtype(self.training_df[similarity_attribute].dtype):
                max_attribute_value = self.training_df[similarity_attribute].max()
                weight = 1/max_attribute_value
            else:
                weight = 1
            
            weights.append(weight)

        return np.array(weights)

    def compute_svd(self, cluster_assignings):
        
        target_matrix = self.get_target_matrix(cluster_assignings)

        if self.n_svd_dimensions > len(self.cluster_centroids):
            logger.warning(f"The number of svd dimensions is larger than the number of clusters. The number of svd dimensions has been reduced from {self.n_svd_dimensions} to {len(self.cluster_centroids) - 1}")
            self.n_svd_dimensions = len(self.cluster_centroids) - 1

        logger.progress("Decomposing target matrix into u, s, vT")
        return svds(target_matrix, k=self.n_svd_dimensions)

    def get_target_matrix(self, cluster_assignings):
        """
        Creates a matrix with the target values as cell values. Each row is index through
        the row_labels and each column is indexed through the column labels.
        """

        shape = (len(self.cluster_centroids), len(self.column_labels))

        rows = []
        columns = []
        data = []

        for cluster_id, indices in enumerate(cluster_assignings):

            logger.progress(f"Filling target matrix {cluster_id/len(self.cluster_centroids) * 100:.2f}%")

            n_instances_in_cluster = len(indices)

            for index in indices:

                row = self.training_df.iloc[index]
                
                row_attribute_value = row[self.row_attribute]
                column_attribute_value = row[self.column_attribute]
                target_value = row[self.target]

                column_label_index = self.index_sorted_list(self.column_labels, column_attribute_value)

                rows.append(cluster_id)
                columns.append(column_label_index)
                data.append(target_value / n_instances_in_cluster)

        logger.progress(f"Finalizing target matrix")

        return csr_matrix((data, (rows, columns)), shape=shape)

    def index_sorted_list(self, sorted_list, element):
        i = bisect.bisect_left(sorted_list, element)
        if i != len(sorted_list) and sorted_list[i] == element:
            return i
        return -1

    def get_cluster_assignings(self, similarity_array):
        """
        Find the closest cluster for each instance. Return a nested list,
        where each row 0 holds all indices belonging to cluster 0, etc.
        """

        distances = distance.cdist(similarity_array, self.cluster_centroids)

        cluster_ids = np.argmin(distances, axis=1)

        indices_per_cluster = [[] for i in range(len(self.cluster_centroids))]

        for i, cluster_id in enumerate(cluster_ids):
            indices_per_cluster[cluster_id].append(i)
        
        return indices_per_cluster

    def compute_similarity_scores(self, entity):
        
        entity_array = np.expand_dims(entity.to_numpy() * self.row_similarity_weights, axis=0)

        distances = distance.cdist(entity_array, self.cluster_centroids)
        similarity_scores = -distances
        
        return similarity_scores.tolist()
