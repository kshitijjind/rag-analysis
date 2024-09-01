import logging
import config

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(config.LOG_LEVEL)

# Create a stream handler and set its level
stream_handler = logging.StreamHandler()
stream_handler.setLevel(config.LOG_LEVEL)

# Create a formatter and add it to the stream handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# Add the stream handler to the logger
logger.addHandler(stream_handler)