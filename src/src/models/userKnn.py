import math
from .recommInterface import Recommendation
from ..helpers.helpFunctions import parse_str_to_3value_range, d_range, count_sum_product


class UserKnn(Recommendation):
    """
    Standard User-knn algorithm.
    """

    def __init__(self, algorithm_config, logger, data_storage):
        super().__init__(algorithm_config, logger)
        self.data = data_storage

    def get_n_recommendation_batch(self, query_vectors, n, params):
        self.logger.warn("Not implemented in user-knn")
        pass

    def get_n_similar_items(self, query_vector, n, params):
        self.logger.warn("Not implemented in user-knn")
        return

    def get_n_similar_user(self, query_vector, n, params):
        """
        Fins similar users using cosine similarity.
        """

        if isinstance(query_vector, list):
            query_vector = {i['item_id']: i['weight'] for i in query_vector}
        users = []
        possible = set()
        for column in query_vector:
            if column in self.data.get_train_column_matrix():
                for i in self.data.get_train_column(column):
                    possible.add(i)
        for row in possible:
            users.append([
                row,
                UserKnn.cosine_similarity_fmid(
                    query_vector,
                    count_sum_product(query_vector),
                    self.data.get_train_row(row),
                    self.data.get_train_row_sum(row)
                )
            ])
        users.sort(key=lambda x: (x[1]), reverse=True)
        return users[:n]

    def get_n_recommendation(self, query_vector, n, params):
        """
        Generate recommendation from generated similar users.
        """

        query_vector = {i['item_id']: i['weight'] for i in query_vector}
        recommendation = {}
        k_b, k_m, k_s = parse_str_to_3value_range(self.algorithm_config['k'])
        beta = self.algorithm_config['beta']
        k_nearest = self.get_n_similar_user(query_vector=query_vector, n=int(k_m), params=params)

        for i in d_range(k_b, k_m, k_s):
            recomm_items = self.get_item_score_from_nearest(query_vector=query_vector,
                                                            k_nearest=k_nearest[:int(i)], n=n)

            recomm_items = [
                [_item_id, k/pow(self.data.get_train_item_popularity(_item_id), float(beta))]
                for _item_id, k in recomm_items
            ]
            recomm_items.sort(key=lambda x: (x[1]), reverse=True)
            recomm_items = [_item_pair[0] for _item_pair in recomm_items]
            recommendation["k={},beta={}".format(str(i), str(beta))] = recomm_items[:n]
        return recommendation

    def get_item_score_from_nearest(self, query_vector, k_nearest, n):
        items_and_weight = {}

        for u in k_nearest:
            items = self.data.get_train_row(u[0])
            weight_of_user = u[1]
            for item in items:
                if item in query_vector:
                    continue
                if item not in items_and_weight:
                    items_and_weight[item] = float(items[item]) * weight_of_user
                else:
                    items_and_weight[item] += float(items[item]) * weight_of_user

        recommendation = [[k, items_and_weight[k]] for k in items_and_weight]
        return recommendation

    @staticmethod
    def dot_and_sum(first, second):
        """
        Compute dot product of vectors
        """

        result = 0.0
        for item in first:
            if item in second:
                result += first[item] * second[item]
        return result

    @staticmethod
    def dot_sparse_vectors(first, second):
        """
        Preparation for dot - sparse representation dot short to longer (faster)
        """

        if len(first) < len(second):
            result = UserKnn.dot_and_sum(first, second)
        else:
            result = UserKnn.dot_and_sum(second, first)
        return result

    @staticmethod
    def cosine_similarity_fmid(user_a, sum_a, user_b, sum_b):
        """
        Compute cosine similarity
        """

        return UserKnn.dot_sparse_vectors(user_a, user_b) / math.sqrt(sum_a * sum_b)
