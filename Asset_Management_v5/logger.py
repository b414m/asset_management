import logging
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Logger:
    def get_logger(self, log_dir='logs', log_name=None, show_in_console=False):
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'{timestamp}_priorization_pipeline.log'

        log_path = os.path.join(log_dir, log_name)
        if not logger.handlers:
            fh = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            if show_in_console:
                ch = logging.StreamHandler()
                ch.setFormatter(formatter)
                logger.addHandler(ch)

        return logger
