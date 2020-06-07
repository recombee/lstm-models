from .recommInterface import Recommendation


class Popularity(Recommendation):
    """
    Model recommend top-n items by global popularity.
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
            recomm_items.append(self.sort_pop[i][0])
        recommendation["n={}".format(str(n))] = recomm_items
        return recommendation

    def get_n_recommendation_batch(self, query_vectors, n, params):
        self.logger.warn("Not implemented in reminder")
        return

    def __init__(self, algorithm_config, logger, data_storage):
        super().__init__(algorithm_config, logger)
        self.pop = data_storage.item_popularity
        self.sort_pop = [(i, self.pop[i]) for i in self.pop]
        self.sort_pop.sort(key=lambda x: x[1], reverse=True)
