from .helpers.helpFunctions import init
from .models.lstmModel import LSTMModel
from .helpers.dataStore import DataStore
from .helpers.helpFunctions import MemMon

import tensorflow as tf


"""
Create or load LSTM model.
"""
experiment_config, logger = init("Training LSTM network")

evaluation_config = experiment_config['evaluation']
general_config = experiment_config['general']
data_preparation_config = experiment_config['data-preparation']
lstm_config = experiment_config['lstm']

memory_monitor = MemMon(logger, general_config['max-memory-mb'], 10)
memory_monitor.run()

logger.info("TensorFlow version:{}".format(tf.version.VERSION))
logger.info("Keras version:{}".format(tf.keras.__version__))

logger.info("Loading data for experiment...")
data_storage = DataStore(logger=logger, general_config=general_config, data_preparation_config=data_preparation_config,
                         evaluation_config=evaluation_config, lstm_config=lstm_config)
logger.info("Loading data for experiment finished")

logger.info("Creating lstm model for experiment...")
model = LSTMModel(lstm_config=lstm_config, logger=logger, data_storage=data_storage, general_config=general_config,
                  data_preparation_config=data_preparation_config, train_phase=True)
model.compile()
model_path = lstm_config['model-to-train']
if model_path.lower() == 'none':
    logger.info("It will be create new model.")
else:
    logger.info("Load model {}".format(model_path))
    model.load_model(model_path)
logger.info("Model load finished.")


logger.info(model.summary())
logger.info("Start training...")
epochs = lstm_config['epochs']
model.fit_generator(generator=data_storage.get_train_data_sequences(),
                    validation_data=data_storage.get_validation_data_sequences(),
                    epochs=epochs, use_multiprocessing=False, shuffle=True)
