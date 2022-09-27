import logging


def custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ''
    log_format = logging.Formatter(format_string)
    
    # Creating and adding the console handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)
    
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
