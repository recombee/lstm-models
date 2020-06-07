import logging


def setup_logger(file_name, terminal, level):
    """
    Create logger - log file or terminal of both
    """
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = terminal

    if file_name is not None:

        # create a file handler
        handler = logging.FileHandler(file_name)
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

    logger.info('Logging successfully set.')

    return logger
