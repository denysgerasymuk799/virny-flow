"""Alpine Meadow logging utilities."""

import os
import logging
import logging.config

import yaml


def setup_logger(output_file=None, logging_config=None):
    """
    Set up logger
    :param output_file:
    :param logging_config:
    :return:
    """

    # logging_config must be a dictionary object specifying the configuration
    # for the loggers to be used in auto-sklearn.
    if logging_config is not None:
        if output_file is not None:
            logging_config['handlers']['file_handler']['filename'] = output_file
        logging.config.dictConfig(logging_config)
    else:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r') as fh:
            logging_config = yaml.safe_load(fh)
        if output_file is not None:
            logging_config['handlers']['file_handler'] = {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'mode': 'w',
                'filename': output_file
            }
            logging_config['root']['handlers'].append('file_handler')
        logging.config.dictConfig(logging_config)


def get_logger(name, debug=False):
    logger = logging.getLogger(name)
    if debug:
        logger.info('Debugging mode...')
        logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)
    return logger


# silence loggers
logging.getLogger('smac').setLevel(logging.CRITICAL)
