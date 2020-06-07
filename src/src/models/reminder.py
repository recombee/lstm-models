from .recommInterface import Recommendation


class Reminder(Recommendation):
    """
    Basic reminder model which recommend last n items which user interacted.
    """

    def get_n_similar_items(self, item_id, n, params):
        self.logger.warn("Not implemented in reminder")
        return

    def get_n_similar_user(self, user_id, n, params):
        self.logger.warn("Not implemented in reminder")
        return

    def get_n_recommendation(self, query_vector, n, params):
        recommendation = {}
        recomm_items = []
        for i in range(n):
            if i > len(query_vector):
                break
            else:
                recomm_items.append(query_vector[-i]['item_id'])
        recommendation["n={}".format(str(n))] = recomm_items
        return recommendation

    def get_n_recommendation_batch(self, query_vectors, n, params):
        self.logger.warn("Not implemented in reminder")
        return

    def __init__(self, algorithm_config, logger):
        super().__init__(algorithm_config, logger)
