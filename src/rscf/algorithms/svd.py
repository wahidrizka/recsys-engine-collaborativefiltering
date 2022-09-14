import numpy
import pandas
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List
from . import AbstractAlgorithm

class SVD(AbstractAlgorithm):
    predictions: dict = None
    
    def __init__(self):
        self.recommender_name = SVD
    
    def get_recommendation(self, user_id: int) -> List[List[any]]:
        return self._get_articles_recommendation(user_id).values.tolist()
    
    def is_trained(self) -> bool:
        return self.predictions is not None
    
    def train(self) -> AbstractAlgorithm:
        self._compute_svd()
        
        return self
    
    def _compute_svd(self):
        interactions_pivot = self.interactions \
            .pivot_table(
                index='user_id',
                columns='article_id',
                values='weight').fillna(0)
            
        # Creating users articles matrix
        interactions_matrix = interactions_pivot.to_numpy()
        interactions_sparse_matrix =  csr_matrix(interactions_matrix)
        users_ids = list(interactions_pivot.index)
        
        # Perform SVD computation
        u, s, vt = svds(interactions_sparse_matrix, k=6)
        s_diagonal_matrix = numpy.diag(s)
        predictions_matrix = numpy.dot(numpy.dot(u, s_diagonal_matrix), vt)
        predictions_matrix_norms = (predictions_matrix - predictions_matrix.min()) / (predictions_matrix.max() - predictions_matrix.min())
        
        # Reconstructing matrix
        self.predictions =  pandas.DataFrame(
            predictions_matrix_norms, 
            columns=interactions_pivot.columns, 
            index=users_ids).transpose()
    
    def _get_articles_recommendation(self, user_id) -> DataFrame:
        articles_recommendations = self.predictions[user_id] \
                .sort_values(ascending=False) \
                    .reset_index() \
                        .rename(columns={user_id: 'strength'})
        
        
        articles_recommendations = pandas.DataFrame(
            articles_recommendations, 
            columns=['article_id', 'strength']) \
            .sort_values(by=['strength'], ascending=False, ignore_index=True)

        return articles_recommendations
            
               