import logging


class Protocol:
    def __init__(
        self, priority="DEBUG", log_dir: str = "logs", log_file: str = "protocol.log"
    ):
        # make logs directory
        import os

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        # set up logging
        if priority == "DEBUG":
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
        elif priority == "INFO":
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
        elif priority == "WARNING":
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
        elif priority == "ERROR":
            logging.basicConfig(
                level=logging.ERROR,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
        elif priority == "CRITICAL":
            logging.basicConfig(
                level=logging.CRITICAL,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
        else:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
                ],
            )
            logging.warning("Priority {} not defined. Default: DEBUG".format(priority))

    def debug(self, message):
        logging.debug(message)

    def info(self, message):
        logging.info(message)

    def warning(self, message):
        logging.warning(message)

    def error(self, message):
        logging.error(message)

    def critical(self, message):
        logging.critical(message)
