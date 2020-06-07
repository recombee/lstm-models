from .recommInterface import Recommendation
from ..helpers.helpFunctions import parse_str_to_3value_range, d_range, is_close
from .userKnn import UserKnn


class ItemKnn(Recommendation):
    """
    Standard Item-knn algorithm.
    """

    def get_n_similar_items(self, item_id, n, params):
        if item_id in self.cache:
            return self.cache[item_id][:n]
        items = []
        item_vector = self.data.get_train_column(item_id)
        possible = set()
        for row in item_vector:
            if row in self.data.get_train_row_matrix():
                for i in self.data.get_train_row(row):
                    possible.add(i)
        for column in possible:
            items.append([
                column,
                UserKnn.cosine_similarity_fmid(
                    item_vector,
                    self.data.get_train_column_sum(item_id),
                    self.data.get_train_column(column),
                    self.data.get_train_column_sum(column)
                )
            ])
        items.sort(key=lambda x: (x[1]), reverse=True)
        self.cache[item_id] = items
        if len(items) == 0:
            return items
        return items[1:n+1] if is_close(items[0][1], 1.) else items[:n]

    def get_n_recommendation_batch(self, query_vectors, n, params):
        self.logger.warn("Not implemented in user-knn")
        pass

    def get_n_similar_user(self, user_id, n, params):
        self.logger.warn("Not implemented in item-knn")
        return

    def get_n_recommendation(self, query_vector, n, params):
        return ItemKnn.generate_top_n_recommendation_item_knn(
            self.data, query_vector, self.get_n_similar_items, n, self.algorithm_config)

    @staticmethod
    def generate_top_n_recommendation_item_knn(data, query_vector, n_similar_items_method, n, params):
        """
        Method for generation recommendation using user data and similarity function to gen n similar items to one.
        """

        query_vector = {i['item_id']: i['weight'] for i in query_vector}
        recommendation = {}
        k_b, k_m, k_s = parse_str_to_3value_range(params['k'])
        beta = params['beta']

        k_nearest = {}
        for item in query_vector:
            k_nearest[item] = n_similar_items_method(item_id=item, n=int(k_m), params=None)

        for k in d_range(k_b, k_m, k_s):
            recomm_items = ItemKnn.get_item_score_from_nearest(query_vector=query_vector,
                                                               k_near_precomputed=k_nearest,
                                                               k=int(k), n=n)

            recomm_items = [[i, j/pow(data.get_train_item_popularity(i), float(beta))] for i, j in recomm_items]
            recomm_items.sort(key=lambda x: (x[1]), reverse=True)
            recomm_items = [i[0] for i in recomm_items]

            if 'param_add' in params:
                recommendation["k={},beta={}".format(str(k), str(beta)) + params['param_add']] = recomm_items[:n]
            else:
                recommendation["k={},beta={}".format(str(k), str(beta))] = recomm_items[:n]

        return recommendation

    @staticmethod
    def get_item_score_from_nearest(query_vector, k_near_precomputed, k, n):
        recommendation = {}
        for item in query_vector:
            if item in k_near_precomputed:
                k_nearest = k_near_precomputed[item][:k]
                for i in k_nearest:
                    if i[0] in query_vector:
                        continue
                    if i[0] not in recommendation:
                        recommendation[i[0]] = 0
                    recommendation[i[0]] += i[1] * query_vector[item]

        recommendation = [[i[0], i[1]] for i in sorted(recommendation.items(), key=lambda x: x[1])[::-1]]
        return recommendation

    def __init__(self, algorithm_config, logger, data_storage):
        super().__init__(algorithm_config, logger)
        self.data = data_storage
        self.cache = {}
        self.computed_on = 0.
        self.computed_on_cnt = 0
