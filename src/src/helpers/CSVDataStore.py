import json
import csv
import os
import re

from .helpFunctions import PercentCounter


class CSVDataStore:
    """
    Alternative storage for use csv files.
    Can be replace with another with same interface (PostgreSQL, Cassandra, etc..)
    """

    @staticmethod
    def save_users(file_name, data):
        """Save users to file. One row one userID."""
        with open(file_name, 'w', newline='') as writer:
            writer.write("userid" + "\n")
            for user in data:
                writer.write(str(user) + "\n")

    @staticmethod
    def get_users_splits(train_file, valid_file, test_file):
        with open(train_file, "r") as f:
            train = set(f.read().splitlines()[1:])
        with open(valid_file, "r") as f:
            valid = set(f.read().splitlines()[1:])
        with open(test_file, "r") as f:
            test = set(f.read().splitlines()[1:])
        return train, valid, test

    @staticmethod
    def load_users(file_name):
        users = set()
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                users.add(row[0])
        return users

    @staticmethod
    def load_raw_data(file_name):
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                yield row

    @staticmethod
    def load_embeddings(experiment, experiment_name, embeddings_file):
        # format as dict(itemid, embedding)
        with open("../data/{}/{}/embeddings/{}".format(experiment, experiment_name, embeddings_file), 'r') as f:
            return {y.split(',')[0]:[float(x) for x in y.split(',')[1:]] for y in f.readlines()}

    @staticmethod
    def load_similar_items(dataset, experiment_name, file_name):
        # format as dict(itemid, list((itemid, similarity)))
        with open("../data/{}/{}/similar_items/{}".format(dataset, experiment_name, file_name), 'r') as f:
            return {row.split(",")[0]: [(x['id'], x['rating']) for x in sorted(json.loads(",".join(row.split(",")[1:]).strip()), key=lambda x: x['rating'], reverse=True)] for row in f.readlines()}

    @staticmethod
    def save_similar_items(dataset, experiment_name, file_name, similar_items, logger):
        per_print = PercentCounter(len(similar_items), 0.05, logger)

        if not os.path.exists("../data/{}/{}/similar_items".format(dataset, experiment_name)):
            os.makedirs("../data/{}/{}/similar_items".format(dataset, experiment_name))

        with open("../data/{}/{}/similar_items/{}".format(dataset, experiment_name, file_name), 'w') as f:
            for idx, sim in enumerate(similar_items):
                f.write(str(sim) + ',' + json.dumps(similar_items[sim]))
                f.write('\n')
                per_print.increment("{} items written".format(idx))

        logger.info("{} items written".format(len(similar_items)))

    @staticmethod
    def get_all_precomp_with_prefix(dataset, experiment_name, prefix):
        my_path = '../data/{}/{}/similar_items'.format(dataset, experiment_name)
        onlyfiles = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f)) and re.match(prefix, f)]
        return onlyfiles

    @staticmethod
    def store_items_embeddings(dataset, experiment_name, file_name, embeddings_array, item_index, logger):
        per_print = PercentCounter(len(embeddings_array), 0.05, logger)

        if not os.path.exists("../data/{}/{}/embeddings".format(dataset, experiment_name)):
            os.makedirs("../data/{}/{}/embeddings".format(dataset, experiment_name))

        with open("../data/{}/{}/embeddings/{}".format(dataset, experiment_name, file_name), 'w') as f:
            for idx, factors in enumerate(embeddings_array):
                f.write(item_index[idx] + ',' + ','.join([str(x) for x in factors]))
                f.write('\n')

                per_print.increment("{} items written".format(idx))

        logger.info("{} items written".format(len(embeddings_array)))
