# logger.py
import logging
from colorama import Fore, Style, init

# Инициализация colorama для корректного вывода в Windows / PyCharm
init(autoreset=True)

class ColorFormatter(logging.Formatter):
    """Добавляет цвета в зависимости от уровня логирования."""
    LEVEL_COLORS = {
        logging.DEBUG: Style.DIM + Fore.LIGHTBLACK_EX,        # серый
        logging.INFO: Fore.WHITE,                             # белый (по умолчанию)
        logging.WARNING: Fore.YELLOW,                         # жёлтый
        logging.ERROR: Fore.LIGHTRED_EX,                      # красный
        logging.CRITICAL: Style.BRIGHT + Fore.RED,             # ярко-красный
    }

    def format(self, record):
        log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
        formatter = logging.Formatter(log_fmt)
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        return color + formatter.format(record) + Style.RESET_ALL


def get_logger(name="AITrader"):
    """Создаёт логгер с цветным форматированием."""
    logger = logging.getLogger(name)

    # предотвращаем дублирование хэндлеров
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# глобальный логгер для использования по всему проекту
logger = get_logger()