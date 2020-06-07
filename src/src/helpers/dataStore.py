import time
import numpy as np

from ..helpers.constants import *
from ..models.dataSequence import DataSequence
from ..helpers.CSVDataStore import CSVDataStore
from .helpFunctions import is_close, count_sum_product, PercentCounter


def get_users_split(dataset, experiment_name):
    """
    Load users split.
    """

    return CSVDataStore.get_users_splits(
        "../data/{}/{}/data/train".format(dataset, experiment_name),
        "../data/{}/{}/data/valid".format(dataset, experiment_name),
        "../data/{}/{}/data/test".format(dataset, experiment_name))


class DataStore:
    """
    Save and prepare not only the interaction data, manage all data need by models
    Use lazy loading, load data after it is need to save some memory. Not load interaction in reminder...
    """

    def __init__(self, logger, general_config, data_preparation_config, evaluation_config, lstm_config):
        """
        Perform base data load.
        """

        self.logger = logger
        self.n_test_users = evaluation_config['n-test-users']
        self.evaluation_type = evaluation_config['evaluation-type']
        self.max_in_leave_one_out = evaluation_config['max-in-leave-one-out']

        self.lstm_config = lstm_config

        self.train_data_prec = data_preparation_config['data-prec'].split(";")[0]
        self.train_minutes_check = None
        if len(data_preparation_config['data-prec'].split(";")) > 1:
            self.train_minutes_check = float(data_preparation_config['data-prec'].split(";")[1])

        self.test_data_prec = evaluation_config['data-prec'].split(";")[0]
        self.test_minutes_check = None
        if len(evaluation_config['data-prec'].split(";")) > 1:
            self.test_minutes_check = float(evaluation_config['data-prec'].split(";")[1])

        self.min_sequence_length = lstm_config['min-sequence-length']
        self.max_sequence_length = lstm_config['max-sequence-length']
        self.logger.info("Minimum length of sequence: {}".format(self.min_sequence_length))
        self.logger.info("Maximum length of sequence: {}".format(self.max_sequence_length))
        self.min_ratings_per_user = data_preparation_config['min-ratings-per-user']
        self.max_ratings_per_user = data_preparation_config['max-ratings-per-user']
        self.logger.info("Minimum ratings per user: {}".format(self.min_ratings_per_user))
        self.logger.info("Maximum ratings per user: {}".format(self.max_ratings_per_user))
        self.experiment_name = general_config['experiment-name']
        self.dataset = general_config['dataset']

        self.days = data_preparation_config['days']
        self.days = self.days if self.days > 0 else None

        self.logger.info("Getting user split..")
        self.train_users, self.valid_users, self.test_users = get_users_split(self.dataset, self.experiment_name)

        self.number_of_train_users = data_preparation_config['number-of-train-users']
        self.allow_train_users = set()
        self.number_of_train_users = self.number_of_train_users if self.number_of_train_users > 0 else None
        self.number_of_valid_users = data_preparation_config['number-of-valid-users']
        self.allow_valid_users = set()
        self.number_of_valid_users = self.number_of_valid_users if self.number_of_valid_users > 0 else None

        if self.number_of_train_users is not None:
            for i, user in enumerate(self.train_users):
                if i >= self.number_of_train_users:
                    break
                self.allow_train_users.add(user)
            self.train_users = self.train_users.intersection(self.allow_train_users)

        if self.number_of_valid_users is not None:
            for i, user in enumerate(self.valid_users):
                if i >= self.number_of_valid_users:
                    break
                self.allow_valid_users.add(user)
            self.valid_users = self.valid_users.intersection(self.allow_valid_users)

        self.min_weight = data_preparation_config['min-weight']
        self.max_weight = data_preparation_config['max-weight']

        all_size = len(self.train_users) + len(self.valid_users) + len(self.test_users)
        if all_size == 0:
            raise ValueError("Not enough users to test ->> 0")
        logger.info("Data split: train {};{}, valid {};{}, test {};{}".format(
            len(self.train_users)/all_size, len(self.train_users),
            len(self.valid_users)/all_size, len(self.valid_users),
            len(self.test_users)/all_size, len(self.test_users)))

        self.row_data_matrix_train = {}
        self.row_data_matrix_valid = {}
        self.row_data_matrix_test = {}

        popularity = {}

        self.max_timestamp = 0
        self.min_timestamp = time.time()

        interactions_cnt = 0
        t = time.time()
        path = '../data/{}/'.format(self.dataset)
        for i in CSVDataStore.load_raw_data(path + '/data/' + general_config['interaction-data-file']):
            if self.days and t - float(i[2]) > self.days * 86400:
                continue

            if float(i[2]) > self.max_timestamp:
                self.max_timestamp = float(i[2])

            if float(i[2]) < self.min_timestamp:
                self.min_timestamp = float(i[2])

            if i[0] in self.train_users:
                if i[0] not in self.row_data_matrix_train:
                    self.row_data_matrix_train[i[0]] = []
                self.row_data_matrix_train[i[0]].append(
                    {'user_id': i[0], 'item_id': i[1], 'timestamp': i[2], 'type': i[3], 'weight': get_weight(i[3])})
                interactions_cnt += 1
                if i[0] not in popularity:
                    popularity[i[0]] = set()
                popularity[i[0]].add(i[0])

            if i[0] in self.valid_users:
                if i[0] not in self.row_data_matrix_valid:
                    self.row_data_matrix_valid[i[0]] = []
                self.row_data_matrix_valid[i[0]].append(
                    {'user_id': i[0], 'item_id': i[1], 'timestamp': i[2], 'type': i[3], 'weight': get_weight(i[3])})
                interactions_cnt += 1

            if i[0] in self.test_users:
                if i[0] not in self.row_data_matrix_test:
                    self.row_data_matrix_test[i[0]] = []
                self.row_data_matrix_test[i[0]].append(
                    {'user_id': i[0], 'item_id': i[1], 'timestamp': i[2], 'type': i[3], 'weight': get_weight(i[3])})
                interactions_cnt += 1
        self.item_popularity = {x: len(popularity[x]) for x in popularity}
        del popularity

        self.n_items_train = len(self.item_popularity)

        self.logger.info("Number of interactions: {}".format(interactions_cnt))
        self.logger.info("Number of train items: {}".format(self.n_items_train))

        self.embeddings = None
        self.embeddings_file = lstm_config['embeddings']
        self.embedding_length = lstm_config['input-embeddings']
        self.logger.info("Embeddings file: {}".format(self.embeddings_file))
        self.logger.info("Embeddings length: {}".format(self.embedding_length))

        self.train_row_column_matrix = None
        self.train_column_row_matrix = None
        self.train_row_sums = None
        self.train_column_sums = None

        self.bins_dim = lstm_config['bins-exponent']
        self.n_bins = pow(2, self.bins_dim)
        self.bins = []
        self.create_timestamp_binning()

        self.batch_size = lstm_config['batch-size']
        self.add_weight = lstm_config['add-weight']
        self.add_timestamp = lstm_config['add-timestamps']
        self.use_bins = lstm_config['use-bins']
        self.per_seq_bins = lstm_config['per-seq-bins']

        self.model_type = lstm_config['model-type']

    def create_timestamp_binning(self):
        """
        Do binning on timestamp for use context data in recurrent model.
        """

        add = (self.max_timestamp - self.min_timestamp) / self.n_bins
        for i in range(self.n_bins - 1):
            self.bins.append(self.min_timestamp + (i + 1) * add)

    def get_bin_for_timestamp(self, timestamp):
        """
        Return one bin which timestamp bellow.
        """

        bin_number = None
        for i, value in enumerate(self.bins):
            if timestamp < value:
                bin_number = i
                break
        bin_number = self.n_bins - 1 if bin_number is None else bin_number
        return np.array([i for i in np.binary_repr(bin_number, width=self.bins_dim)], np.float)

    def get_bins_dim(self):
        """
        Return bin dimension to choose number of neurons.
        """

        if self.add_timestamp:
            if self.use_bins:
                return self.bins_dim
            else:
                return 1
        else:
            return 0

    def len_embedding_sim(self):
        """
        Return embedding length only without context data.
        """
        return self.embedding_length

    def len_input_embedding(self):
        """
        Complete length of embedding with context data.
        """

        return self.embedding_length + (1 if self.add_weight else 0) + \
               (self.bins_dim if (self.add_timestamp and self.use_bins) else (1 if self.add_timestamp else 0))

    def n_train_columns(self):
        """
        Number of train items.
        """

        if self.n_items_train == 0:
            return 1
        return self.n_items_train

    def get_n_test_users(self):
        """
        Number of test users.
        """

        return len(self.test_users)

    def get_embeddings(self):
        """
        Return all embeddings or load if not already as dict with numpy array.
        """
        if self.embeddings is None:
            self.embeddings = CSVDataStore.load_embeddings(self.dataset, self.experiment_name, self.embeddings_file)
        return self.embeddings

    def create_row_column_matrix(self):
        """
        Create sparse rating matrix representation from sequence store data.
        """

        self.train_row_column_matrix = {}
        self.train_row_sums = {}
        per_print = PercentCounter(len(self.row_data_matrix_train), 0.1, self.logger)
        for user in self.row_data_matrix_train:
            per_print.increment("Create interaction records: ")
            for item in self.row_data_matrix_train[user]:
                if item['weight'] is None or is_close(float(item['weight']), 0., 0.0001):
                    continue
                if user not in self.train_row_column_matrix:
                    self.train_row_column_matrix[user] = {}
                if item['item_id'] not in self.train_row_column_matrix[user]:
                    self.train_row_column_matrix[user][item['item_id']] = 0
                self.train_row_column_matrix[item['user_id']][item['item_id']] = max(
                    self.min_weight,
                    min(self.train_row_column_matrix[item['user_id']][item['item_id']] + float(item['weight']),
                        self.max_weight))
        for row in self.train_row_column_matrix:
            self.train_row_sums[row] = count_sum_product(self.train_row_column_matrix[row])

        self.logger.info("Load rows columns matrix: {}".format(len(self.train_row_column_matrix)))

    def create_column_row_matrix(self):
        """
        Create inverse sparse rating matrix representation from sequence store data.
        This is useful for knn cut off for speed up.
        """

        self.train_column_row_matrix = {}
        self.train_column_sums = {}
        per_print = PercentCounter(len(self.row_data_matrix_train), 0.1, self.logger)
        for user in self.row_data_matrix_train:
            per_print.increment("Create interaction records: ")
            for item in self.row_data_matrix_train[user]:
                if item['weight'] is None or is_close(float(item['weight']), 0., 0.0001):
                    continue
                if item['item_id'] not in self.train_column_row_matrix:
                    self.train_column_row_matrix[item['item_id']] = {}
                if user not in self.train_column_row_matrix[item['item_id']]:
                    self.train_column_row_matrix[item['item_id']][user] = 0
                self.train_column_row_matrix[item['item_id']][item['user_id']] = max(
                    self.min_weight,
                    min(self.train_column_row_matrix[item['item_id']][item['user_id']] + float(item['weight']),
                        self.max_weight))

        for column in self.train_column_row_matrix:
            self.train_column_sums[column] = count_sum_product(self.train_column_row_matrix[column])

        self.logger.info("Load columns row matrix: {}".format(len(self.train_column_row_matrix)))

    def reload_lstm_config(self, lstm_config):
        """
        If testing LSTM models it is useful load config for load model to prepare data structures.
        """

        self.bins_dim = lstm_config['bins-exponent']
        self.add_weight = lstm_config['add-weight']
        self.add_timestamp = lstm_config['add-timestamps']
        self.use_bins = lstm_config['use-bins']
        self.embeddings_table = lstm_config['embeddings']
        self.embedding_length = lstm_config['input-embeddings']
        self.min_sequence_length = lstm_config['min-sequence-length']
        self.max_sequence_length = lstm_config['max-sequence-length']
        self.lstm_config = lstm_config

    def get_train_item_popularity(self, item_id):
        if item_id not in self.item_popularity:
            return 1
        return self.item_popularity[item_id]

    def get_train_row(self, row_id):
        if self.train_row_column_matrix is None:
            self.create_row_column_matrix()
        if row_id not in self.train_row_column_matrix:
            return {}
        return self.train_row_column_matrix[row_id]

    def get_train_column(self, column_id):
        if self.train_column_row_matrix is None:
            self.create_column_row_matrix()
        if column_id not in self.train_column_row_matrix:
            return {}
        return self.train_column_row_matrix[column_id]

    def get_train_value(self, row_id, column_id):
        if self.train_row_column_matrix is None:
            self.create_row_column_matrix()
        if row_id not in self.train_row_column_matrix:
            return None
        if column_id not in self.train_row_column_matrix[row_id]:
            return None
        return self.train_row_column_matrix[row_id][column_id]

    def get_train_row_matrix(self):
        if self.train_row_column_matrix is None:
            self.create_row_column_matrix()
        return self.train_row_column_matrix

    def get_train_column_matrix(self):
        if self.train_column_row_matrix is None:
            self.create_column_row_matrix()
        return self.train_column_row_matrix

    def get_train_column_sum(self, column_id):
        if self.train_column_sums is None:
            self.create_column_row_matrix()
        return self.train_column_sums[column_id]

    def get_train_row_sum(self, row_id):
        if self.train_row_sums is None:
            self.create_row_column_matrix()
        return self.train_row_sums[row_id]

    def get_test_tuples(self):
        """
        Work as generator for testing users. Preprocessed them with method specific in configuration file.
        """

        for i, test_item in enumerate(self.test_users):
            if i >= self.n_test_users:
                break

            if self.evaluation_type == "last-leave-one-out":
                test = self.process_one_user(sorted(self.row_data_matrix_test[test_item],
                                                    key=lambda x: (x["timestamp"])),
                                             self.test_data_prec, self.test_minutes_check)
                for tmp in test:
                    if len(tmp) < self.min_ratings_per_user or len(tmp) > self.max_ratings_per_user:
                        continue
                    yield tmp[:-1], tmp[-1]
            elif self.evaluation_type == "leave-one-out":
                test = self.process_one_user(self.row_data_matrix_test[test_item],
                                             self.test_data_prec, self.test_minutes_check)

                for tmp in test:
                    if len(tmp) < self.min_ratings_per_user or len(tmp) > self.max_ratings_per_user:
                        continue
                    for j in range(len(tmp)):
                        if j >= self.max_in_leave_one_out:
                            break
                        yield [x for num, x in enumerate(tmp) if num != j], tmp[j]
            else:
                raise ValueError("Not supported evaluation type.")
        return None

    def get_train_data_sequences(self):
        """
        Return preprocessed train sequences.
        """

        self.logger.info("{}; Data size before remove: {}".format("train", len(self.row_data_matrix_train)))
        if type(self.row_data_matrix_train) is dict:
            self.logger.info("Train data not process - start prepossessing sequential data.")
            data = []
            for user in list(self.row_data_matrix_train.keys()):
                test_users = self.process_one_user(sorted(self.row_data_matrix_train[user],
                                                          key=lambda x: (x["timestamp"])),
                                                   self.train_data_prec, self.train_minutes_check)
                for tmp in test_users:
                    if self.lstm_config['remove-items-without-embedding']:
                        tmp = [y for y in tmp if y['item_id']]
                    if len(tmp) < self.min_sequence_length:
                        continue
                    data.append(tmp)
                del self.row_data_matrix_train[user]

            self.logger.info("{}; Data size after remove: {}".format("train", len(data)))

            return DataSequence(
                name='train', data=data, logger=self.logger,
                lstm_config=self.lstm_config, min_sequence_length=self.min_sequence_length,
                max_sequence_length=self.max_sequence_length, data_storage=self)

        self.logger.error("Inconsistent state - error in train data processing.")
        raise ValueError("Inconsistent state - error in train data processing.")

    def get_validation_data_sequences(self):
        """
        Return preprocessed validation sequences.
        """

        self.logger.info("{}; Data size before remove: {}".format("valid", len(self.row_data_matrix_valid)))
        if type(self.row_data_matrix_valid) is dict:
            self.logger.info("Validation data not processed - start prepossessing sequential data.")
            data = []
            for user in list(self.row_data_matrix_valid.keys()):
                test_users = self.process_one_user(sorted(self.row_data_matrix_valid[user],
                                                          key=lambda x: (x["timestamp"])),
                                                   self.train_data_prec, self.train_minutes_check)
                for tmp in test_users:
                    if self.lstm_config['remove-items-without-embedding']:
                        tmp = [y for y in tmp if y['item_id']]
                    if len(tmp) < self.min_sequence_length:
                        continue
                    data.append(tmp)
                del self.row_data_matrix_valid[user]

            self.logger.info("{}; Data size after remove: {}".format("valid", len(data)))

            return DataSequence(
                name='valid', data=data, logger=self.logger,
                lstm_config=self.lstm_config, min_sequence_length=self.min_sequence_length,
                max_sequence_length=self.max_sequence_length, data_storage=self)

        self.logger.error("Inconsistent state - error in validation data processing.")
        raise ValueError("Inconsistent state - error in validation data processing.")

    def process_one_user(self, sort_user, process_type, process_interval):
        """
        Do preprocessing on one user sequence.
        """

        temp_user = sort_user.copy()
        if len(temp_user) <= 0:
            return temp_user

        if process_type == 'raw':
            pre_proc_users = [temp_user]

        elif process_type == 'merge':
            pre_proc_users = [merge(temp_user, process_interval)]

        elif process_type == 'end_buy':
            pre_proc_users = [end_with_x(temp_user, BUY)]

        elif process_type == 'end_cart':
            pre_proc_users = [end_with_x(temp_user, CART)]

        elif process_type == 'end_cart_without_click':
            temp_user = remove_x_before_y(temp_user, CLICK, CART)
            pre_proc_users = [end_with_x(temp_user, CART)]

        elif process_type == 'end_buy_without_click':
            temp_user = remove_x_before_y(temp_user, CLICK, BUY)
            pre_proc_users = [end_with_x(temp_user, BUY)]

        elif process_type == 'end_buy_merge':
            temp_user = remove_x_before_y(temp_user, CLICK, BUY)
            pre_proc_users = [end_with_x_merge(temp_user, process_interval, BUY)]

        elif process_type == 'end_cart_merge':
            temp_user = remove_x_before_y(temp_user, CLICK, CART)
            pre_proc_users = [end_with_x_merge(temp_user, process_interval, CART)]

        elif process_type == 'end_buy_sub':
            tmp = [x for x in generate_shortening_sequences_end_with_x(temp_user, BUY, self.min_sequence_length)]
            pre_proc_users = [end_with_x(remove_x_before_y(x, CLICK, BUY), BUY) for x in tmp]

        elif process_type == 'end_buy_sub_merge':
            tmp = [x for x in generate_shortening_sequences_end_with_x(temp_user, BUY, self.min_sequence_length)]
            pre_proc_users = [end_with_x_merge(remove_x_before_y(x, CLICK, BUY), process_interval, BUY) for x in tmp]

        elif process_type == 'end_cart_sub':
            tmp = [x for x in generate_shortening_sequences_end_with_x(temp_user, CART, self.min_sequence_length)]
            pre_proc_users = [end_with_x(remove_x_before_y(x, CLICK, CART), CART) for x in tmp]

        elif process_type == 'end_cart_sub_merge':
            tmp = [x for x in generate_shortening_sequences_end_with_x(temp_user, CART, self.min_sequence_length)]
            pre_proc_users = [end_with_x_merge(remove_x_before_y(x, CLICK, CART), process_interval, CART) for x in tmp]

        else:
            raise ValueError("unsupported data preprocessing")

        return pre_proc_users


def print_user(user):
    print()
    for i in user:
        print('(', i['item_id'], '-', i['timestamp'], '-', i['type'], ")---", sep='')
    print()


def merge(d, interval):
    """
    Merging some interaction in sequence and interval.
    """

    ret = d.copy()
    stable = False
    while not stable:
        stable = True
        indexes = set()
        for i in range(len(ret)-1):
            if ret[i]['item_id'] == ret[i+1]['item_id'] and ret[i]['type'] == ret[i+1]['type'] and \
               (interval is None or ret[i]['timestamp'] - ret[i+1]['timestamp'] < interval*60):
                indexes.add(i)
                stable = False
        ret = remove_from_from_list(ret, indexes)
    return ret


def remove_x_before_y(d, x, y, interval=None):
    """
    Remove interaction x before interaction y if occur.
    """

    indexes = set()
    for i in range(len(d)-1, -1, -1):
        if d[i]['type'] == y:
            j = i-1
            while j >= 0 and d[i]['item_id'] == d[j]['item_id'] and x == d[j]['type'] and \
                    (interval is None or d[i]['timestamp'] - d[j]['timestamp'] < interval*60):
                indexes.add(j)
                j = j - 1
    return remove_from_from_list(d, indexes)


def remove_smaller_before_y(d, y, interval=None):
    """
    If interaction has natural ordering it is possible remove interaction with smaller weight before the bigger one.
    """

    indexes = set()
    for i in range(len(d)-1, -1, -1):
        if d[i]['type'] == y:
            j = i-1
            while j >= 0 and d[i]['item_id'] == d[j]['item_id'] and y > d[j]['type'] and \
                    (interval is None or d[i]['timestamp'] - d[j]['timestamp'] < interval*60):
                indexes.add(j)
                j = j - 1
    return remove_from_from_list(d, indexes)


def end_with_x(d, x):
    """
    Sequence end with x and all not x type at the end remove.
    """

    ret = d.copy()
    indexes = set()
    for i in range(len(ret)-1, -1, -1):
        if ret[i]['type'] == x:
            break
        else:
            indexes.add(i)
    return remove_from_from_list(ret, indexes)


def end_with_x_merge(d, interval, x):
    """
    Sequence end with x and all not x type at the end remove and merge same interaction.
    """

    ret = d.copy()
    ret = merge(ret, interval)
    return end_with_x(ret, x)


def remove_from_from_list(d, indexes):
    # remove indexes from list
    return [x for i, x in enumerate(d) if i not in indexes]


def generate_shortening_sequences_end_with_x(user_vector, x, min_length=3):
    """
    Generate shortening sequences -> this better represent history.
    Use historical sequences which use generate.
    """

    while len(user_vector) >= min_length:
        while user_vector[-1]['type'] != x and len(user_vector) >= min_length:
            user_vector = user_vector[:-1]
        if len(user_vector) < min_length:
            continue
        yield user_vector
        user_vector = user_vector[:-1]
