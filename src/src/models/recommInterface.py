from abc import ABC, abstractmethod


class Recommendation(ABC):
    """
    Interface for testing models using framework.
    """

    def __init__(self, algorithm_config, logger):
        self.algorithm_config = algorithm_config
        self.logger = logger

    @abstractmethod
    def get_n_similar_items(self, item_id, n, params):
        """
        Return list of n similar items to one selected.
        """
        pass

    @abstractmethod
    def get_n_similar_user(self, user_id, n, params):
        """
        Return n similar user to one.
        """
        pass

    @abstractmethod
    def get_n_recommendation(self, query_vector, n, params):
        """
        Recommendation for one selected user.
        """
        pass

    @abstractmethod
    def get_n_recommendation_batch(self, query_vectors, n, params):
        """
        Recommendation for batch users in list.
        """
        pass
