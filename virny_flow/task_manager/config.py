import asyncio
import logging

from virny_flow.core.utils.custom_logger import CustomHandler
from virny_flow.configs.constants import DEBUG_MODE

kafka_loop = asyncio.get_event_loop()

# Prepare own helper class objects
logger = logging.getLogger('root')
if DEBUG_MODE:
    logger.setLevel('DEBUG')
else:
    logger.setLevel('INFO')
    logging.disable(logging.DEBUG)
logger.addHandler(CustomHandler())
