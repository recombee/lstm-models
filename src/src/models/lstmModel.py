from .recommInterface import Recommendation
import tensorflow as tf
import os
import yaml
import numpy as np
import scipy.spatial as sp

"""# TF2.0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, LearningRateScheduler
from tensorflow.keras.constraints import MaxNorm, NonNeg, MinMaxNorm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("Invalid device or cannot modify virtual devices once initialized.")
    pass
"""

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, GRU
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, LearningRateScheduler
from keras.constraints import MaxNorm, NonNeg, MinMaxNorm
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend

from keras.backend.tensorflow_backend import set_session, get_session

import keras.losses

"""
Definition of parametrized loss function parts and all for the model.
"""


def my_MSE_TIMESTAMP(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_TIMESTAMP(y_true, y_pred):
        return keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, emb_len], [-1, timestamp_len]), y_pred=tf.slice(y_pred, [0, emb_len], [-1, timestamp_len]))
    return MSE_TIMESTAMP


def my_MSE_TIMESTAMP_MANY(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_TIMESTAMP(y_true, y_pred):
        return keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0, emb_len], [-1, -1, timestamp_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len], [-1, -1, timestamp_len]))
    return MSE_TIMESTAMP


def my_MSE_WEIGHT(emb_len=0, timestamp_len=0, weight_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_WEIGHT(y_true, y_pred):
        return keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, emb_len+timestamp_len], [-1, weight_len]), y_pred=tf.slice(y_pred, [0, emb_len+timestamp_len], [-1, weight_len]))
    return MSE_WEIGHT


def my_MSE_WEIGHT_MANY(emb_len=0, timestamp_len=0, weight_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_WEIGHT(y_true, y_pred):
        return keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]))
    return MSE_WEIGHT


def my_COSINE_PROXIMITY(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def COSINE_PROXIMITY(y_true, y_pred):
        return keras.losses.cosine_proximity(y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len]))
    return COSINE_PROXIMITY


def my_COSINE_PROXIMITY_MANY(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def COSINE_PROXIMITY(y_true, y_pred):
        return keras.losses.cosine_proximity(y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len]))
    return COSINE_PROXIMITY


def my_MSE_EMB(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_EMB(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len]))
    return MSE_EMB


def my_MSE_EMB_MANY(emb_len=0, timestamp_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    def MSE_EMB(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len]))
    return MSE_EMB


def my_loss(emb_len=None, timestamp_len=0, weight_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    """
    Definition of loss function for the model.
    It is for the parts of the vectors if use context data.
    wrapping function for pass constant parameters
    """

    def loss(y_true, y_pred):
        if weight_len == 0 and timestamp_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
                mse_weight * keras.losses.mean_squared_error\
                       (y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len]))
        if weight_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
            timestamp_weigh * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, emb_len], [-1, timestamp_len]), y_pred=tf.slice(y_pred, [0, emb_len], [-1, timestamp_len]))
        if timestamp_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
            weight_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, emb_len+timestamp_len], [-1, weight_len]), y_pred=tf.slice(y_pred, [0, emb_len+timestamp_len], [-1, weight_len]))
        return cosine_weight * keras.losses.cosine_proximity(
            y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0], [-1, emb_len]), y_pred=tf.slice(y_pred, [0, 0], [-1, emb_len])) + \
            timestamp_weigh * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, emb_len], [-1, timestamp_len]), y_pred=tf.slice(y_pred, [0, emb_len], [-1, timestamp_len])) + \
            weight_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, emb_len+timestamp_len], [-1, weight_len]), y_pred=tf.slice(y_pred, [0, emb_len+timestamp_len], [-1, weight_len]))
    return loss


def my_loss_MANY(emb_len=None, timestamp_len=0, weight_len=0, cosine_weight=0.9, mse_weight=0.1, timestamp_weigh=0.1, weight_weight=0.1):
    """
    Definition of loss function for the model this is use for run model many-to-many.
    It is for the parts of the vectors if use context data.
    wrapping function for pass constant parameters
    """

    def loss(y_true, y_pred):
        if weight_len == 0 and timestamp_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
                mse_weight * keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len]))
        if weight_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
            timestamp_weigh * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0, emb_len], [-1, -1, timestamp_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len], [-1, -1, timestamp_len]))
        if timestamp_len == 0:
            return cosine_weight * keras.losses.cosine_proximity(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
                y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
            weight_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]))
        return cosine_weight * keras.losses.cosine_proximity(
            y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
               mse_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0, 0], [-1, -1, emb_len]), y_pred=tf.slice(y_pred, [0, 0, 0], [-1, -1, emb_len])) + \
            timestamp_weigh * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0, emb_len], [-1, -1, timestamp_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len], [-1, -1, timestamp_len])) + \
            weight_weight * keras.losses.mean_squared_error(
            y_true=tf.slice(y_true, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]), y_pred=tf.slice(y_pred, [0, 0, emb_len+timestamp_len], [-1, -1, weight_len]))
    return loss


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# TF2.0
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def align_sequence(s, align_size):
    """
    Align sequence to specific size. Use 0 to padding. If is longer it is cut.
    """

    s = s[::-1]
    while len(s) < align_size:
        s.append({'item_id': None})
    return s[:align_size][::-1]


class LSTMModel(Recommendation):
    """
    Class for manage recurrent model load or train model.
    """

    def gen_embeddings_array(self):
        """
        Return preprocessed embeddings array.
        """

        int_to_item = {}
        item_to_int = {}
        train_items_emb = np.zeros((len(self.embeddings), self.data_storage.len_embedding_sim()))
        for idx, item in enumerate(self.embeddings):
            int_to_item[idx] = item
            item_to_int[item] = idx
            train_items_emb[idx] = np.asarray(self.embeddings[item])
        return train_items_emb, int_to_item, item_to_int

    @staticmethod
    def get_b(binning, timestamp, data_storage):
        """
        Return bins for one timestamp as binary numpy array.
        """

        bin_number = None
        for i, value in enumerate(binning):
            if timestamp < value:
                bin_number = i
                break
        bin_number = data_storage.n_bins-1 if bin_number is None else bin_number
        return np.array([i for i in np.binary_repr(bin_number, width=data_storage.bins_dim)], np.float)

    @staticmethod
    def get_emb(item, data_storage, binning):
        """
        Method for get embedding. Check using of context data and create embedding.
        """

        emb = data_storage.get_embeddings()
        e = np.zeros(data_storage.len_input_embedding())
        if item['item_id'] is not None and data_storage.add_weight:
            e[-1] = item['weight']
        if item['item_id'] is not None and data_storage.add_timestamp and data_storage.use_bins:
            e[data_storage.len_embedding_sim():data_storage.len_embedding_sim()+data_storage.get_bins_dim()] = \
                LSTMModel.get_b(binning, item['timestamp'], data_storage)
        if item['item_id'] is not None and data_storage.add_timestamp and not data_storage.use_bins:
            e[-2] = item['timestamp_delta']
        if item['item_id'] is None or item['item_id'] not in emb:
            return False, e
        e[:data_storage.len_embedding_sim()] = emb[item['item_id']]
        return True, e

    def get_n_recommendation_batch(self, query_vectors, n, params):
        """
        It is need return recommendation as list of list with recommendation for users.
        In same order how get it.
        """

        # recommendation array
        recommendation = [{} for _ in range(len(query_vectors))]

        # generate array for prediction
        query_array = np.zeros((len(query_vectors), self.max_seq_length, self.data_storage.len_input_embedding()), dtype=np.float32)

        for i, x in enumerate(query_vectors):
            temp_bin = []
            if self.data_storage.per_seq_bins:

                max_t = max([x['timestamp'] for x in x])
                min_t = min([x['timestamp'] for x in x])
                add = (max_t - min_t) / self.data_storage.n_bins
                for j in range(self.data_storage.n_bins-1):
                    temp_bin.append(min_t + (j+1)*add)

            for j, y in enumerate(align_sequence(x, self.max_seq_length)):
                is_emb, emb = LSTMModel.get_emb(y, self.data_storage, temp_bin)
                query_array[i][j] = emb

        def append_top_n_with_score(prediction, rec, m, coef):
            """Compute recommendation with score for the predict and add n to recommendation"""
            sims = 1 - sp.distance.cdist(prediction[:, :self.data_storage.len_embedding_sim()], self.train_items_embeddings, 'cosine')
            sort_sims = np.argsort(sims, axis=1)
            top_items = []
            for i, one in enumerate(sort_sims):
                for j, u in enumerate(one[-m:][::-1]):
                    if j == 0:
                        top_items.append(self.int_to_item_train[u])
                    if self.int_to_item_train[u] not in rec[i]:
                        rec[i][self.int_to_item_train[u]] = 0
                    rec[i][self.int_to_item_train[u]] += sims[i][u] * coef
            return top_items

        def roll_data_many_to_many(data, last_prediction):
            """Append embedding to the observation of the user."""
            tmp = np.roll(data, -1, axis=1)
            tmp[:, -1, :] = last_prediction[:, -1, :]
            return tmp

        def roll_data_many_to_one(data, last_prediction):
            """Append embedding to the observation of the user."""
            tmp = np.roll(data, -1, axis=1)
            tmp[:, -1, :] = last_prediction[:, :]
            return tmp

        # todo: some heuristic?
        coef = 1

        d = query_array
        for cycle in range(self.algorithm_config['n-rec-cycle']):
            # prediction
            predict = self.model.predict(d, batch_size=2048)

            if self.model_type == 'many-to-many':

                k_from_sequence = 10
                for i, p in enumerate([x for x in range(self.max_seq_length)][::-1][:k_from_sequence]):
                    # todo: coeficient of recommendation item * coeficient
                    # return value is obsolete
                    top_items = append_top_n_with_score(predict[:, p, :], recommendation, self.k, coef)

                d = roll_data_many_to_many(d, predict)

            elif self.model_type == 'many-to-one':
                top_items = append_top_n_with_score(predict[:, :], recommendation, self.k, coef)
                if self.nearest_item_cycle:
                    self.logger.info("Replacing predict embedding with nearest item embedding.")
                    for i in range(len(predict)):
                        predict[i] = self.train_items_embeddings[self.item_to_int_train[top_items[i]]]
                d = roll_data_many_to_one(d, predict)
            else:
                self.logger.error("Not supported model type.")

        for i in range(len(recommendation)):
            for item in recommendation[i]:
                recommendation[i][item] = recommendation[i][item]/pow(self.data_storage.get_train_item_popularity(item), self.beta)

        result = [{"name={},epoch={}".format(
            self.model_name, int(self.load.split("/")[-1].split("-")[1].split(".")[0])
                ):[o[0] for o in sorted([[y, x[y]] for y in x], key=lambda t: t[1], reverse=True)[:n]]} for x in recommendation]

        return result

    def get_n_similar_user(self, user_id, n, params):
        self.logger.warn("Not implemented in user-knn")
        return

    def get_n_recommendation(self, query_vector, n, params):
        return self.get_n_recommendation_batch([query_vector], n, params)[0]

    def get_n_similar_items(self, item_id, n, params):
        self.logger.warn("Not implemented in user-knn")
        return

    def __init__(self, lstm_config, logger, data_storage, general_config, data_preparation_config, train_phase):
        super().__init__(lstm_config, logger)
        self.load = None

        self.add_weight = lstm_config['add-weight']
        self.add_timestamp = lstm_config['add-timestamps']
        self.use_bins = lstm_config['use-bins']

        schema = general_config['dataset']
        self.k = lstm_config['k']
        self.experiment_name = general_config['experiment-name']
        self.embeddings = data_storage.get_embeddings()
        self.beta = lstm_config['beta']
        self.data_storage = data_storage
        self.train_items_embeddings, self.int_to_item_train, self.item_to_int_train = self.gen_embeddings_array()
        self.lstm_config = lstm_config

        model_postfix = lstm_config['model-postfix']
        model_postfix = "" if len(model_postfix) == 0 else "_{}".format(model_postfix)
        self.model_name = "e{}b{}nn{}n{}m{}{}".format(
            lstm_config['input-embeddings'], lstm_config['batch-size'], lstm_config['n-recurent-layers'],
            lstm_config['layer-size'], lstm_config['max-sequence-length'], model_postfix)

        self.batch_size = lstm_config['batch-size']
        self.max_seq_length = lstm_config['max-sequence-length']

        self.layer_size = lstm_config['layer-size']
        self.model_type = lstm_config['model-type']
        self.n_recurent_layers = lstm_config['n-recurent-layers']
        self.logger = logger
        self.nearest_item_cycle = lstm_config['nearest-item-cycle']
        self.learning_rate = lstm_config['learning-rate']
        self.optimizer = lstm_config['optimizer']
        self.activation = lstm_config['activation']
        self.recurrent_activation = lstm_config['recurrent-activation']

        d_constraint, d_b_constraint, lstm_constraint, lstm_b_constraint, lstm_r_constraint = read_weight_constrains(
            lstm_config, logger)

        self.dropout_all = lstm_config['dropout-all']
        self.dropout_rate = lstm_config['dropout-rate']
        self.first_dropout = lstm_config['first-dropout']
        self.first_recurrent_dropout = lstm_config['first-recurrent-dropout']
        self.dropout = lstm_config['dropout']
        self.recurrent_dropout = lstm_config['recurrent-dropout']
        self.dense_dropout = lstm_config['dense-dropout']
        self.dense_dropout_all = lstm_config['dense-dropout-all']

        NEURON_CELL = LSTM if lstm_config['LSTM'] else GRU

        # define of model from parameters
        self.model = Sequential()
        if lstm_config['dropout-all']:
            self.model.add(Dropout(lstm_config['dropout-rate'],
                                   input_shape=(self.max_seq_length, self.data_storage.len_input_embedding())))

        for i in range(self.n_recurent_layers):
            if i == 0:
                if lstm_config['dropout-all']:
                    self.model.add(NEURON_CELL(self.layer_size, activation=self.activation,
                                        dropout=lstm_config['first-dropout'],
                                        recurrent_dropout=lstm_config['first-recurrent-dropout'],
                                        recurrent_activation=self.recurrent_activation,
                                        return_sequences=False if (i == self.n_recurent_layers-1 and self.model_type == 'many-to-one') else True,
                                        kernel_constraint=lstm_constraint,
                                        recurrent_constraint=lstm_r_constraint,
                                        bias_constraint=lstm_b_constraint))
                else:
                    self.model.add(NEURON_CELL(self.layer_size, activation=self.activation,
                                        dropout=lstm_config['first-dropout'],
                                        recurrent_dropout=lstm_config['first-recurrent-dropout'],
                                        recurrent_activation=self.recurrent_activation,
                                        input_shape=(self.max_seq_length, self.data_storage.len_input_embedding()),
                                        return_sequences=False if (i == self.n_recurent_layers-1 and self.model_type == 'many-to-one') else True,
                                        kernel_constraint=lstm_constraint,
                                        recurrent_constraint=lstm_r_constraint,
                                        bias_constraint=lstm_b_constraint))
                continue
            if i == self.n_recurent_layers-1:
                if self.model_type == 'many-to-many':
                    self.model.add(
                            NEURON_CELL(self.layer_size, activation=self.activation, recurrent_activation=self.recurrent_activation,
                                 return_sequences=True, dropout=lstm_config['dropout'],
                                 recurrent_dropout=lstm_config['recurrent-dropout'],
                                 kernel_constraint=lstm_constraint,
                                 recurrent_constraint=lstm_r_constraint,
                                 bias_constraint=lstm_b_constraint))
                if self.model_type == 'many-to-one':
                    self.model.add(
                        NEURON_CELL(self.layer_size, activation=self.activation, recurrent_activation=self.recurrent_activation,
                             return_sequences=False, dropout=lstm_config['dropout'],
                             recurrent_dropout=lstm_config['recurrent-dropout'],
                             kernel_constraint=lstm_constraint,
                             recurrent_constraint=lstm_r_constraint,
                             bias_constraint=lstm_b_constraint))
                continue
            self.model.add(
                NEURON_CELL(self.layer_size, activation=self.activation, recurrent_activation=self.recurrent_activation,
                     return_sequences=True, dropout=lstm_config['dropout'],
                     recurrent_dropout=lstm_config['recurrent-dropout'],
                     kernel_constraint=lstm_constraint,
                     recurrent_constraint=lstm_r_constraint,
                     bias_constraint=lstm_b_constraint))

        if self.model_type == 'many-to-many':
            if lstm_config['dense-dropout-all']:
                self.model.add(TimeDistributed(Dropout(lstm_config['dense-dropout'],
                                   input_shape=(self.max_seq_length, self.data_storage.len_input_embedding()))))
            self.model.add(TimeDistributed(BatchNormalization()))
            self.model.add(TimeDistributed(Dense(self.data_storage.len_input_embedding(), kernel_constraint=d_constraint, bias_constraint=d_b_constraint)))
        elif self.model_type == 'many-to-one':
            if lstm_config['dense-dropout-all']:
                self.model.add(Dropout(lstm_config['dense-dropout'],
                                   input_shape=(self.max_seq_length, self.data_storage.len_input_embedding())))
            self.model.add(BatchNormalization())
            self.model.add(Dense(self.data_storage.len_input_embedding(), kernel_constraint=d_constraint, bias_constraint=d_b_constraint))
        else:
            raise ValueError("unsupported model-type")

        # create dirs for save checkpoints for next evaluation
        if not os.path.exists("../data/{}/{}/checkpoints/{}/{}".format(
                schema, self.experiment_name, self.model_type, self.model_name)):
            os.makedirs("../data/{}/{}/checkpoints/{}/{}".format(
                schema, self.experiment_name, self.model_type, self.model_name))
        if not os.path.exists("../data/{}/{}/logs/{}/{}".format(
                schema, self.experiment_name, self.model_type, self.model_name)):
            os.makedirs("../data/{}/{}/logs/{}/{}".format(
                schema, self.experiment_name, self.model_type, self.model_name))
        checkpoint_path = "../data/{}/{}/checkpoints/{}/{}/cp-{}.ckpt".format(
            schema, self.experiment_name, self.model_type, self.model_name, "{epoch:04d}")

        self.schedule = get_lr_scheduler(lstm_config, self.learning_rate, self.logger)

        self.callbacks = [
            MyLogger(logger, self.schedule, self.learning_rate, lstm_config, "../data/{}/{}/checkpoints/{}/{}/{}".format(
                schema, self.experiment_name, self.model_type, self.model_name, 'run_config.yml'), train_phase),
            ModelCheckpoint(filepath=checkpoint_path, monitor='cosine', verbose=1, save_best_only=False,
                            save_weights_only=False, mode='auto', period=1),
            TensorBoard(log_dir='../data/{}/{}/logs/{}/{}'.format(
                schema, self.experiment_name, self.model_type, self.model_name), write_graph=True),
            CSVLogger('../data/{}/{}/logs/{}/{}/run_data_t.csv'.format(
                schema, self.experiment_name, self.model_type, self.model_name), append=True,
                      separator=',')
        ]

        if self.schedule is not None:
            self.callbacks.append(LearningRateScheduler(self.schedule))

        # create loss function and metric to monitor
        if self.model_type == 'many-to-many':
            self.my_loss = my_loss_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=lstm_config['cosine-weight'],
                                   mse_weight=lstm_config['mse-weight'],
                                   timestamp_weigh=lstm_config['timestamp-weight'],
                                   weight_weight=lstm_config['weight-weight'])
            self.my_cosine_proximity = my_COSINE_PROXIMITY_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_emb = my_MSE_EMB_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_timestamp = my_MSE_TIMESTAMP_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_weight = my_MSE_WEIGHT_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim())

        else:
            self.my_loss = my_loss(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=lstm_config['cosine-weight'],
                                   mse_weight=lstm_config['mse-weight'],
                                   timestamp_weigh=lstm_config['timestamp-weight'],
                                   weight_weight=lstm_config['weight-weight'])
            self.my_cosine_proximity = my_COSINE_PROXIMITY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_emb = my_MSE_EMB(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_timestamp = my_MSE_TIMESTAMP(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            self.my_mse_weight = my_MSE_WEIGHT(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim())

    def compile(self):
        """
        Compile model with specific optimizer and monitor metric to monitor.
        """

        if self.optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == 'momentum':
            optimizer = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=False)
        elif self.optimizer == 'SGD':
            optimizer = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0, nesterov=False)
        else:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # tf==2.1
        # self.model.compile(optimizer=optimizer, loss=tf.keras.losses.CosineSimilarity(axis=1), metrics=['mse'])
        if self.data_storage.get_bins_dim() == 0 and self.data_storage.add_weight == 0:
            self.model.compile(
                optimizer=optimizer, loss=self.my_loss,
                metrics=['mse', 'cosine', self.my_cosine_proximity, self.my_mse_emb])
        if self.data_storage.get_bins_dim() == 0:
            self.model.compile(
                optimizer=optimizer, loss=self.my_loss,
                metrics=['mse', 'cosine', self.my_cosine_proximity, self.my_mse_emb, self.my_mse_weight])
        if self.data_storage.add_weight == 0:
            self.model.compile(
                optimizer=optimizer, loss=self.my_loss,
                metrics=['mse', 'cosine', self.my_cosine_proximity, self.my_mse_emb, self.my_mse_timestamp])
        if self.data_storage.get_bins_dim() != 0 and self.data_storage.add_weight != 0:
            self.model.compile(
                optimizer=optimizer, loss=self.my_loss,
                metrics=['mse', 'cosine', self.my_cosine_proximity, self.my_mse_emb, self.my_mse_timestamp, self.my_mse_weight])

    def fit(self, x, y, val_x, val_y, epochs=100, batch_size=32):
        """
        Train created or loaded LSTM network. Obsolete not usable for large data.
        """
        return self.model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size, verbose=1,
                              callbacks=self.callbacks)

    def fit_generator(self, generator, validation_data, use_multiprocessing, shuffle, epochs=100, workers=8):
        """
        Use for train LSTM or GRU using data sequence generator which prepare data for training.
        """
        return self.model.fit_generator(generator=generator, validation_data=validation_data,
                                        use_multiprocessing=use_multiprocessing, shuffle=shuffle, epochs=epochs,
                                        verbose=1, callbacks=self.callbacks, workers=workers)

    def summary(self):
        """
        Print info about model.
        """

        self.logger.info("Input embedding: {}".format(self.data_storage.len_input_embedding()))
        self.logger.info("Max sequence length: {}".format(self.max_seq_length))
        self.logger.info("Model name: {}".format(self.model_name))
        self.logger.info("Recurrent layer size: {}".format(self.layer_size))
        self.logger.info("Number of recurrent layers: {}".format(self.n_recurent_layers))
        self.logger.info("Learning rate: {}".format(self.learning_rate))
        self.logger.info("Optimizer: {}".format(self.optimizer))
        self.logger.info("Activation function: {}".format(self.activation))
        self.logger.info("Recurrent activation function: {}".format(self.recurrent_activation))
        self.logger.info("dropout_all: {}".format(self.dropout_all))
        self.logger.info("dropout_rate: {}".format(self.dropout_rate))
        self.logger.info("first_dropout: {}".format(self.first_dropout))
        self.logger.info("first_recurrent_dropout: {}".format(self.first_recurrent_dropout))
        self.logger.info("dropout: {}".format(self.dropout))
        self.logger.info("recurrent_dropout: {}".format(self.recurrent_dropout))
        self.logger.info("dense_dropout: {}".format(self.dense_dropout))
        self.logger.info("dense_dropout_all: {}".format(self.dense_dropout_all))
        return self.model.summary()

    def predict(self, x):
        """
        Perform prediction using recurrent model.
        """

        return self.model.predict(x, verbose=0)

    def clear_ses(self):
        """
        Clear and create new session. Useful for GPU computation.
        TF do some leaks :D restart session solve problems for several days running continual testing.
        """

        global sess
        backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    def load_model(self, file):
        """
        Load save checkpoint to continue training or testing prediction. Need load loss function too if not use in predict.
        """

        if self.model_type == 'many-to-many':
            dependencies = {
                'my_loss': my_loss_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=self.lstm_config['cosine-weight'],
                                   mse_weight=self.lstm_config['mse-weight'],
                                   timestamp_weigh=self.lstm_config['timestamp-weight'],
                                   weight_weight=self.lstm_config['weight-weight']),
                'loss': my_loss_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=self.lstm_config['cosine-weight'],
                                   mse_weight=self.lstm_config['mse-weight'],
                                   timestamp_weigh=self.lstm_config['timestamp-weight'],
                                   weight_weight=self.lstm_config['weight-weight']),
                'my_COSINE_PROXIMITY': my_COSINE_PROXIMITY_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_EMB': my_MSE_EMB_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_EMB': my_MSE_EMB_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_WEIGHT': my_MSE_WEIGHT_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_WEIGHT': my_MSE_WEIGHT_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_TIMESTAMP': my_MSE_TIMESTAMP_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_TIMESTAMP': my_MSE_TIMESTAMP_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'COSINE_PROXIMITY': my_COSINE_PROXIMITY_MANY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            }
        else:
            dependencies = {
                'my_loss': my_loss(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=self.lstm_config['cosine-weight'],
                                   mse_weight=self.lstm_config['mse-weight'],
                                   timestamp_weigh=self.lstm_config['timestamp-weight'],
                                   weight_weight=self.lstm_config['weight-weight']),
                'loss': my_loss(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim(),
                                   weight_len=1 if self.data_storage.add_weight else 0,
                                   cosine_weight=self.lstm_config['cosine-weight'],
                                   mse_weight=self.lstm_config['mse-weight'],
                                   timestamp_weigh=self.lstm_config['timestamp-weight'],
                                   weight_weight=self.lstm_config['weight-weight']),
                'my_COSINE_PROXIMITY': my_COSINE_PROXIMITY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_EMB': my_MSE_EMB(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_EMB': my_MSE_EMB(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_WEIGHT': my_MSE_WEIGHT(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_WEIGHT': my_MSE_WEIGHT(emb_len=self.data_storage.len_embedding_sim(),
                                               weight_len=1 if self.data_storage.add_weight else 0,
                                               timestamp_len=self.data_storage.get_bins_dim()),
                'my_MSE_TIMESTAMP': my_MSE_TIMESTAMP(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'MSE_TIMESTAMP': my_MSE_TIMESTAMP(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim()),
                'COSINE_PROXIMITY': my_COSINE_PROXIMITY(emb_len=self.data_storage.len_embedding_sim(),
                                   timestamp_len=self.data_storage.get_bins_dim())
            }
        self.load = file
        self.model = load_model(file, custom_objects=dependencies)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)


def get_lr_scheduler(lstm_config, learning_rate, logger):
    """
    Return learning rate scheduler for lstm training
    """

    scheduler = lstm_config['ls-scheduler']
    if scheduler == "step":
        return StepDecay(logger, init_alpha=learning_rate, factor=lstm_config['factor'],
                         drop_every=lstm_config['drop-every'])
    elif scheduler == "poly":
        return PolynomialDecay(logger, max_epochs=lstm_config['epochs'], init_alpha=learning_rate,
                               power=lstm_config['power'])
    else:
        return None


class StepDecay:
    """
    StepDecay learning rate
    """


    def __init__(self, logger, init_alpha=0.01, factor=0.25, drop_every=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every
        self.logger = logger
        logger.info("Use Step LR decay with "
                    "parameters: factor: {}, init_alpha: {}, drop_every: {}".format(self.factor,
                                                                                    self.init_alpha,
                                                                                    self.drop_every))

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        self.logger.info("Learning rate: {}. Epoch: {}".format(float(alpha), epoch))
        return float(alpha)


class PolynomialDecay:
    """
    PolynomialDecay learning rate
    """

    def __init__(self, logger, max_epochs=100, init_alpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.max_epochs = max_epochs
        self.init_alpha = init_alpha
        self.power = power
        self.logger = logger
        logger.info("Use polynomial LR decay with "
                    "parameters: max_epochs: {}, init_alpha: {}, power: {}".format(self.max_epochs,
                                                                                   self.init_alpha,
                                                                                   self.power))

    def get_value(self):
        ...

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_alpha * decay
        self.logger.info("Learning rate: {}. Epoch: {}".format(float(alpha), epoch))
        return float(alpha)


class MyLogger(Callback):
    """
    Logger for save run data to csv.
    """

    def __init__(self, logger, schedule, learning_rate, lstm_config, file, train_phase=False):
        super().__init__()
        if train_phase:
            with open(file, 'w') as configfile:
                yaml.dump(lstm_config, configfile, default_flow_style=False)
        self.logger = logger
        self.schedule = schedule
        self.learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        else:
            learning_rate = self.learning_rate
            if self.schedule is not None:
                learning_rate = self.schedule(epoch)
            self.logger.info(logs)
            logs["learning-rate"] = learning_rate
            logs["n-epoch"] = epoch


def read_weight_constrains(lstm_config, logger):
    """
    Preprocess weights norms from config file.
    """

    return return_norm("dense-l-constraint", lstm_config, lstm_config['dl-min'], lstm_config['dl-max'], logger), \
        return_norm("dense-l-bias-constraint", lstm_config, lstm_config['dlb-min'], lstm_config['dlb-max'], logger), \
        return_norm("lstm-l-constraint", lstm_config, lstm_config['lstm-min'], lstm_config['lstm-max'], logger), \
        return_norm("lstm-l-bias-constraint", lstm_config,
                    lstm_config['lstmb-min'], lstm_config['lstmb-max'], logger), \
        return_norm("lstm-l-recurrent-constraint",
                    lstm_config,lstm_config['lstmr-min'],
                    lstm_config['lstmr-max'], logger)


def return_norm(name, lstm_config, minimum, maximum, logger):
    """
    Return Norm object to norm weight of neural network.
    """

    log_name = name
    name = lstm_config[name.lower()]
    if name == 'maxnorm':
        logger.info("In {} use {} constraint with max={}".format(log_name, name, maximum))
        return MaxNorm(maximum)
    if name == 'nonnegnorm':
        logger.info("In {} use {} constraint ".format(log_name, name))
        return NonNeg()
    if name == 'minmaxnorm':
        logger.info("In {} use {} constraint with min={} and max={}".format(log_name, name, minimum, maximum))
        return MinMaxNorm(minimum, maximum)
    else:
        logger.info("None constraint in {}.".format(log_name))
        return None
