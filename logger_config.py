import logging
from datetime import datetime

def setup_logger():
    class ColoredFormatter(logging.Formatter):
        light_magenta = "\x1b[35;20m" # Subdued pink
        green = "\x1b[32;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        white = "\x1b[37;20m"

        FORMATS = {
            logging.DEBUG: green,
            logging.INFO: green,
            logging.WARNING: yellow,
            logging.ERROR: red,
            logging.CRITICAL: bold_red
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(f"{self.light_magenta}%(asctime)s{self.reset} - {log_fmt}%(levelname)s{self.reset} - {self.white}%(message)s{self.reset}", "%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"application_{timestamp}.log"

    # Create a file handler with the timestamp-based name
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(ColoredFormatter())

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

    logging.basicConfig(level=logging.INFO, handlers=[handler, file_handler])

    return logging.getLogger()
