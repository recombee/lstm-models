from scipy.sparse import csr_matrix
from .helpers.helpFunctions import init
from .helpers.dataStore import DataStore
from .helpers.helpFunctions import MemMon
from .helpers.CSVDataStore import CSVDataStore
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from .helpers.dataClean import reduction, remove_robots

import copy
import os

# disable OPENBLAS parallelization. It is use parallelization in implicit
os.environ["OPENBLAS_NUM_THREADS"] = "1"

experiment_config, logger = init("Embeddings generator")

evaluation_config = experiment_config['evaluation']
general_config = experiment_config['general']
data_preparation_config = experiment_config['data-preparation']
lstm_config = experiment_config['lstm']
factorization_config = experiment_config['factorization']
factorization_parametrizations = factorization_config['parametrizations']
dataset = general_config['dataset']
experiment_name = general_config['experiment-name']


memory_monitor = MemMon(logger, general_config['max-memory-mb'], 10)
memory_monitor.run()

data_storage = DataStore(logger=logger, general_config=general_config, data_preparation_config=data_preparation_config,
                         evaluation_config=evaluation_config, lstm_config=lstm_config)

users = data_storage.get_train_row_matrix()
items = data_storage.get_train_column_matrix()
logger.info("Number of users: {}".format(len(users)))
logger.info("Number of items: {}".format(len(items)))

logger.info("Generating embeddings using factorization")

for fac_pam in factorization_parametrizations:

    u = copy.deepcopy(users)
    i = copy.deepcopy(items)
    u, i = remove_robots(u, i, logger, 0.25)
    u, i = reduction(u, i, logger, pruning=fac_pam['pruning'])

    logger.info("Building data matrix")
    int_to_user = list(u)
    int_to_item = list(i)
    user_to_int = {user_id: idx for idx, user_id in enumerate(int_to_user)}
    item_to_int = {item_id: idx for idx, item_id in enumerate(int_to_item)}

    ratings = [(user_to_int[user_id], item_to_int[item_id], u[user_id][item_id])
                   for user_id in u for item_id in u[user_id]]
    data = None
    if len(ratings) > 0:
        data = csr_matrix(([float(r[2]) for r in ratings], ([r[1] for r in ratings], [r[0] for r in ratings])))
        del ratings
    else:
        logger.error("No data in interaction matrix.")
    del u
    del i

    logger.info("Running for {}".format(fac_pam))

    model = AlternatingLeastSquares(
        factors=fac_pam['factors'], regularization=fac_pam['regularization'], num_threads=fac_pam['num-threads'],
        iterations=fac_pam['iterations'], use_gpu=fac_pam['use-gpu'])

    if fac_pam['use-bm25']:
        logger.info("Applying BM25 normalization with B: {} and bm25_multiplication: {}".format(fac_pam['bm25B'],
                                                                                                fac_pam['bm25M']))
        print(data.shape)
        data = bm25_weight(data, B=fac_pam['bm25B']) * fac_pam['bm25M']
    else:
        logger.info("NOT applying BM25 normalization")

    logger.info("Fitting model")
    model.fit(data)

    CSVDataStore.store_items_embeddings(dataset=dataset, experiment_name=experiment_name,
                                        file_name="embeddings_als_{}_f_{}_l_{}_p_{}_B_{}_M_{}.emb".format(
                                            general_config['experiment-name'], str(fac_pam['factors']),
                                            str(fac_pam['regularization']).replace(".", "_"), str(fac_pam['pruning']),
                                            str(round(fac_pam['bm25B'], 3)).replace(".", "_"),
                                            str(round(fac_pam['bm25M'], 3)).replace(".", "_")),
                                        embeddings_array=model.item_factors, item_index=int_to_item, logger=logger)
    logger.info("Embeddings save to file.")
