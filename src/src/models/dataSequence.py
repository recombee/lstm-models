import json

import numpy as np

# TF2.0
# from tensorflow.keras.utils import Sequence
from keras.utils import Sequence


class DataSequence(Sequence):
    """
    Splits batches to idx. For use CPU to preprocessing batch for GPU.
    """

    def __init__(self, data, logger, name, min_sequence_length, max_sequence_length, lstm_config, data_storage):
        super().__init__()
        self.name = name
        self.data_storage = data_storage
        self.cache_size_max = lstm_config['cache-size']*(10**6)
        self.cache_size = 0
        self.cache = {}
        self.cache_full = False

        self.min_seq_length = min_sequence_length
        self.max_seq_length = max_sequence_length
        self.logger = logger
        self.data = data

        if self.data_storage.per_seq_bins:
            self.bins_max = []
            self.bins_min = []
            self.bins = []
            for i in self.data:
                temp_bin = []
                max_t = max([x['timestamp'] for x in i])
                min_t = min([x['timestamp'] for x in i])
                add = (max_t - min_t) / self.data_storage.n_bins
                for j in range(self.data_storage.n_bins-1):
                    temp_bin.append(min_t + (j+1)*add)
                self.bins.append(temp_bin)

    def get_bin(self, timestamp, index):
        if self.data_storage.per_seq_bins:
            bin_number = None
            for i, value in enumerate(self.bins[index]):
                if timestamp < value:
                    bin_number = i
                    break
            bin_number = self.data_storage.n_bins-1 if bin_number is None else bin_number
            return np.array([i for i in np.binary_repr(bin_number, width=self.data_storage.bins_dim)], np.float)
        else:
            return self.data_storage.get_bin_for_timestamp(timestamp)

    def __len__(self):
        """
        return number of batches
        """
        return int(np.ceil(len(self.data) / float(self.data_storage.batch_size)))

    def __getitem__(self, idx):
        """
        Get batch sequences and create numpy array with it witch embeddings
        """
        if idx in self.cache:
            return self.cache[idx]
        data_seq = self.data[idx * self.data_storage.batch_size:(idx + 1) * self.data_storage.batch_size]
        x, y = replace_only_embedding(sequences=data_seq, max_length=self.max_seq_length,
                                      model_type=self.data_storage.model_type, data_storage=self.data_storage,
                                      first=idx * self.data_storage.batch_size, get_bin=self.get_bin)

        if not self.cache_full and idx not in self.cache and self.cache_size + x.nbytes + y.nbytes < self.cache_size_max:
            self.cache_size += (x.nbytes + y.nbytes)
            self.cache[idx] = (x, y)
            self.logger.info("{}; Data cache use {} %. {} % is in cache.".format(
                self.name, round((self.cache_size/self.cache_size_max)*100),
                round((len(self.cache)/self.__len__())*100)))
        return x, y


def load_user_sequences_from_file(file_name):
    """
    Load sequences of items form file. Sequence of item is sort by time. One user is one sequence.
    """

    with open(file_name, "r") as reader:
        return [json.loads(x) for x in reader.read().splitlines()]


def align_sequence(s, align_size):
    """
    Align sequence to specific size. Use 0 to padding. If is longer it is cut.
    """

    s = s[::-1]
    while len(s) < align_size:
        s.append({'item_id': None})
    return s[:align_size][::-1]


def get_embedding(item, data_storage, index, get_bin):
    """
    Return embedding for the item. If is 0 return vector of 0.
    """

    emb = data_storage.get_embeddings()
    e = np.zeros(data_storage.len_input_embedding())
    if item['item_id'] is not None and data_storage.add_weight:
        e[-1] = item['weight']
    if item['item_id'] is not None and data_storage.add_timestamp and data_storage.use_bins:
        e[data_storage.len_embedding_sim():data_storage.len_embedding_sim()+data_storage.get_bins_dim()] = \
            get_bin(item['timestamp'], index)
    if item['item_id'] is not None and data_storage.add_timestamp and not data_storage.use_bins:
        e[-2] = item['timestamp_delta']
    if item['item_id'] is None or item['item_id'] not in emb:
        return False, e
    e[:data_storage.len_embedding_sim()] = emb[item['item_id']]
    return True, e


def replace_only_embedding(sequences, max_length, model_type, data_storage, first, get_bin):
    """
    Create 3D numpy array from sequences of itemsID to sequences of embeddings.
    """

    if model_type == 'many-to-many':
        ar = np.zeros((len(sequences), max_length+1, data_storage.len_input_embedding()), dtype=np.float32)
        for i, x in enumerate(sequences):
            for j, y in enumerate(align_sequence(x, max_length+1)):
                is_emb, emb = get_embedding(y, data_storage, first+i, get_bin)
                ar[i][j] = emb
        in_array = ar[:, :-1, :]
        out_array = ar[:, 1:, :]
        del ar
        return in_array, out_array
    if model_type == 'many-to-one':
        in_array = np.zeros((len(sequences), max_length, data_storage.len_input_embedding()), dtype=np.float32)
        out_array = np.zeros((len(sequences), data_storage.len_input_embedding()), dtype=np.float32)
        for i, x in enumerate(sequences):
            is_emb, emb = get_embedding(x[-1], data_storage, first+i, get_bin)
            out_array[i] = emb
            for j, y in enumerate(align_sequence(x[:-1], max_length)):
                is_emb, emb = get_embedding(y, data_storage, first+i, get_bin)
                in_array[i][j] = emb
        return in_array, out_array
    raise ValueError("unsupported model type")
