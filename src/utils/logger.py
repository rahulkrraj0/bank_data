# src/utils/logger.py

import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    """
    Set up logging configuration to log messages to a file named 'app.log' in the 'logs' directory.
    
    from src.utils.logger import logger
    logger.info("Data ingestion started")
    """


    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()