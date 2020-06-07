from .recommInterface import Recommendation
from .itemKnn import ItemKnn


class PrecomputedItemKnn(Recommendation):
    """
    Model which get dict with items similarity and perform item-knn algorithm on precomputed similarity.
    """

    def get_n_recommendation_batch(self, query_vectors, n, params):
        self.logger.warn("Not implemented in pre-item-knn")
        pass

    def get_n_similar_items(self, item_id, n, params):
        if item_id not in self.similar_items:
            return []
        else:
            return self.similar_items[item_id][:n]

    def get_n_similar_user(self, user_id, n, params):
        self.logger.warn("Not implemented in precomputed-item-knn")
        return

    def get_n_recommendation(self, query_vector, n, params):
        self.algorithm_config['param_add'] = ",table=" + self.table_name
        recommendation = ItemKnn.generate_top_n_recommendation_item_knn(
            self.data, query_vector, self.get_n_similar_items, n, self.algorithm_config)
        return recommendation

    def __init__(self, algorithm_config, logger, similar_items, table_name, data):
        super().__init__(algorithm_config, logger)
        self.data = data
        self.similar_items = similar_items
        self.table_name = table_name
