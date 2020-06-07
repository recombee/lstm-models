from .helpers.helpFunctions import init, WriteCSVData, PercentCounter
from .models.precomputedItemKnn import PrecomputedItemKnn
from .models.evaluation import evaluate_model
from .helpers.helpFunctions import MemMon
from .helpers.CSVDataStore import CSVDataStore
from .helpers.dataStore import DataStore
from .models.lstmModel import LSTMModel
from .models.reminder import Reminder
from .models.userKnn import UserKnn
from .models.itemKnn import ItemKnn
from .models.popularityModel import Popularity

import gc
import os.path
import yaml
import numpy as np

from scipy import spatial

experiment_config, logger = init("Testing recommendation")

evaluation_config = experiment_config['evaluation']
general_config = experiment_config['general']
data_preparation_config = experiment_config['data-preparation']
user_knn_config = experiment_config['user-knn']
item_knn_config = experiment_config['item-knn']
lstm_config = experiment_config['lstm']
epoch_numbers = lstm_config['model-load-epoch-range'].split(";")
precomputed_similar_items_config = experiment_config['precomputed-similar-items']
similarity_items_rating_config = experiment_config['similarity-items-rating']

memory_monitor = MemMon(logger, general_config['max-memory-mb'], 10)
memory_monitor.run()

algorithms = [i.strip() for i in evaluation_config['algorithms'].split(',')]

experiment_name = general_config['experiment-name']
dataset = general_config['dataset']

logger.info("Loading data for experiment")
data_storage = DataStore(logger=logger, general_config=general_config, data_preparation_config=data_preparation_config,
                         evaluation_config=evaluation_config, lstm_config=lstm_config)

result_path_prefix = "../data/{}/{}/{}/{}/{}".format(
    dataset, experiment_name, 'results', evaluation_config['evaluation-type'], evaluation_config['data-prec'])
if not os.path.exists(result_path_prefix):
    os.makedirs(result_path_prefix)

model = None

logger.info("data load successfully")

for alg in algorithms:

    if alg == 'user-knn':
        logger.info("Testing user-knn...")
        model = UserKnn(user_knn_config, logger, data_storage)
        evaluate_model(csv_writer=WriteCSVData("{}/uknn.csv".format(result_path_prefix)), model=model,
                       data_storage=data_storage, evaluation_config=evaluation_config, batch=False, logger=logger)

    elif alg == 'item-knn':
        logger.info("Testing item-knn...")
        model = ItemKnn(item_knn_config, logger, data_storage)
        evaluate_model(csv_writer=WriteCSVData("{}/iknn.csv".format(result_path_prefix)), model=model,
                       data_storage=data_storage, evaluation_config=evaluation_config, batch=False, logger=logger)

    elif alg == "reminder":
        logger.info("Testing reminder...")
        model = Reminder(algorithm_config=None, logger=logger)
        evaluate_model(csv_writer=WriteCSVData("{}/reminders.csv".format(result_path_prefix)), model=model,
                       data_storage=data_storage, evaluation_config=evaluation_config, batch=False, logger=logger)

    elif alg == "popularity":
        logger.info("Testing popularity...")
        model = Popularity(algorithm_config=None, logger=logger, data_storage=data_storage)
        evaluate_model(csv_writer=WriteCSVData("{}/popularity.csv".format(result_path_prefix)), model=model,
                       data_storage=data_storage, evaluation_config=evaluation_config, batch=False, logger=logger)

    elif alg == "pre-item-knn":
        file_names_prefix = precomputed_similar_items_config['similar-items-files-regex']
        files = CSVDataStore.get_all_precomp_with_prefix(dataset, experiment_name, file_names_prefix)
        logger.info("Files:".format(files))
        for file in files:
            result_file = "{}/piknn_{}.csv".format(result_path_prefix, file)
            if os.path.isfile(result_file + ".csv"):
                logger.info("Precomputed-item-knn evaluate result file exist. Not compute again.")
                continue
            logger.info("Testing precomputed item-knn on table {} ...".format(file))
            model = PrecomputedItemKnn(precomputed_similar_items_config, logger,
                                       CSVDataStore.load_similar_items(dataset, experiment_name, file), file, data_storage)
            evaluate_model(csv_writer=WriteCSVData(result_file), model=model, data_storage=data_storage,
                           evaluation_config=evaluation_config, batch=False, logger=logger)

    elif alg == "similarity-items-rating":
        logger.info("Computing similar items rating")
        model = ItemKnn(None, logger, data_storage)
        similar_items = {}
        per = PercentCounter(data_storage.n_train_columns(), 0.05, logger)
        for item in data_storage.get_train_column_matrix().keys():
            per.increment("Computed similar items: ")
            similar_items[item] = [{"id": x[0], "rating": x[1]} for x in model.get_n_similar_items(
                item, similarity_items_rating_config['max-n'], None)]

        CSVDataStore.save_similar_items(dataset, experiment_name,
                                        similarity_items_rating_config['similarity-rating-file'], similar_items, logger)
        logger.info("Similar items save to table.")

    elif alg == "embeddings-similarity-items-rating":
        logger.info("Computing embeddings similar items rating")
        emb = CSVDataStore.load_embeddings(dataset, experiment_name, lstm_config['embeddings'])

        ids = list(emb.keys())
        arr = np.zeros(shape=(len(ids), lstm_config['input-embeddings']), dtype=float)
        print(arr.shape)
        for i, item_id in enumerate(ids):
            arr[i] = np.array(emb[item_id], dtype=float)

        result = 1 - spatial.distance.cdist(arr, arr, 'cosine')
        indexes = np.argsort(result, axis=1)
        similar_items = {ids[index]:
                    [
                        {'id': int(ix), 'rating': float(result[index][ix])}
                        for cnt, ix in enumerate(row[::-1][1:similarity_items_rating_config['max-n']+1])
                    ]
                for index, row in enumerate(indexes)}
        CSVDataStore.save_similar_items(dataset, experiment_name, "similars" + lstm_config['embeddings'], similar_items, logger)

    elif alg == "lstm":
        model = LSTMModel(lstm_config=lstm_config, logger=logger, data_storage=data_storage,
                          general_config=general_config, data_preparation_config=data_preparation_config, train_phase=False)
        model.compile()

        model_postfix = lstm_config['model-postfix']
        model_postfix = "" if len(model_postfix) == 0 else "_{}".format(model_postfix)
        model_type = lstm_config['model-type']

        for lstm_name in lstm_config['models-to-test'].split(";"):
            
            with open("../data/{}/{}/checkpoints/{}/{}/{}".format(dataset, experiment_name, model_type, lstm_name, "run_config.yml"), 'r') as stream:
                try:
                    lstm_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                    exit(-5)
            
            data_storage.reload_lstm_config(lstm_config)
            model_path = "../data/{}/{}/checkpoints/{}/{}/{}".format(dataset, experiment_name, model_type, lstm_name, "cp-{}.ckpt")
            result_file = "{}/{}_{}.csv".format(result_path_prefix, model_type, lstm_name)

            logger.info("Evaluating model {}. Results save in {}.".format(lstm_name, result_file))

            logger.info("Epoch numbers: {}".format(epoch_numbers))
            csv_writer = WriteCSVData(result_file)
            for i in range(*tuple([int(x) for x in epoch_numbers])):
                if i == 0:
                    continue
                logger.info('Evaluate {} epoch.'.format(i))
                if os.path.isfile(model_path.format("{0:0=4d}".format(i))):
                    model.clear_ses()
                    gc.collect()
                    model.load_model(model_path.format("{0:0=4d}".format(i)))
                    
                    evaluate_model(csv_writer=csv_writer, model=model, data_storage=data_storage,
                                   evaluation_config=evaluation_config, batch=lstm_config['test-batch-evaluate'],
                                   logger=logger)
                else:
                    logger.info('File {} not exists.'.format(model_path.format("{0:0=4d}".format(i))))
            gc.collect()

    else:
        logger.warning("Algorithm not exist")
