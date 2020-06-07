from .helpers.helpFunctions import init
from .helpers.helpFunctions import MemMon
from .helpers.helpFunctions import BlockTimer
from .helpers.CSVDataStore import CSVDataStore
from .helpers.helpFunctions import PercentCounter

import os
import random


def split_and_save_data(config, log):
    """
    Select subset from data set and split it to train, validation and testing and create sequence of items for users.
    """
    general_config = config['general']
    data_preparation_config = config['data-preparation']

    memory_monitor = MemMon(log, general_config['max-memory-mb'], 10)
    memory_monitor.run()

    dataset = general_config['dataset']
    experiment_name = general_config['experiment-name']

    path = '../data/{}/'.format(dataset)

    min_ratings_per_user = data_preparation_config['min-ratings-per-user']
    max_ratings_per_user = data_preparation_config['max-ratings-per-user']
    users = {}

    for i in CSVDataStore.load_raw_data(path + '/data/' + general_config['interaction-data-file']):
        if i[0] not in users:
            users[i[0]] = set()
        users[i[0]].add(i[1])

    train_p = data_preparation_config['train-p']
    valid_p = data_preparation_config['valid-p']
    test_p = data_preparation_config['test-p']

    train = set()
    valid = set()
    test = set()

    log.info("Selecting interacting users: min int: {}, max int: {}, train: {}, test: {}, valid: {}".format(
        min_ratings_per_user, max_ratings_per_user, train_p, test_p, valid_p
    ))

    log.info("Number of users: {}".format(len(users)))
    keep_users_p = data_preparation_config['num-users-take']

    train_p = keep_users_p * train_p
    valid_p = keep_users_p * valid_p + train_p
    test_p = keep_users_p * test_p + valid_p

    log.info("Probability to train user: {}".format(train_p))
    log.info("Probability to valid user: {}".format(valid_p))
    log.info("Probability to test user: {}".format(test_p))

    per = PercentCounter(len(users), 0.05, log)
    for user in users:
        per.increment("Users done")
        if len(users[user]) < min_ratings_per_user or len(users[user]) > max_ratings_per_user:
            continue
        rnd = random.uniform(0, 1)
        if rnd <= train_p:
            train.add(user)
            continue
        if rnd <= valid_p:
            valid.add(user)
            continue
        if rnd <= test_p:
            test.add(user)
            continue

    log.info("Number of train users: {}".format(len(train)))
    log.info("Number of valid users: {}".format(len(valid)))
    log.info("Number of test users: {}".format(len(test)))
    log.info("Percent of train users: {}".format(len(train)/(len(train)+len(valid)+len(test))))
    log.info("Number of valid users: {}".format(len(valid)/(len(train)+len(valid)+len(test))))
    log.info("Number of test users: {}".format(len(test)/(len(train)+len(valid)+len(test))))

    with BlockTimer("", "save users to file: ", log):
        log.info("saving to file...")
        log.info("Saving train users...")
        CSVDataStore.save_users(get_file_name_with_existing_path(dataset, experiment_name, "train"), train)
        log.info("Saving validation users...")
        CSVDataStore.save_users(get_file_name_with_existing_path(dataset, experiment_name, "valid"), valid)
        log.info("Saving test users...")
        CSVDataStore.save_users(get_file_name_with_existing_path(dataset, experiment_name, "test"), test)
    log.info("saved")

    log.info("DONE users splits")


def get_file_name_with_existing_path( schema, experiment_name, part_of_data):
    """
    Return file name from configuration
    """

    if not os.path.exists("../data/{}/{}/data".format(schema, experiment_name)):
        os.makedirs("../data/{}/{}/data".format(schema, experiment_name))
    return "../data/{}/{}/data/{}".format(schema, experiment_name, part_of_data)


run_config, logger = init("Training data generator")

split_and_save_data(run_config, logger)

logger.info("DONE data preparation")
