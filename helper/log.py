import logging
import os


def get_logger(output_folder, name, debug=True):
    logger = logging.getLogger(name=name)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(
        output_folder, 'logs', name + '.log.csv'))
    formatter = logging.Formatter(
        '%(asctime)s,%(name)s,%(levelname)s,%(message)s')
    if debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
