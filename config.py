# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_ENV = os.getenv("BYBIT_ENV", "sandbox").lower()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
TIMEFRAME = os.getenv("TIMEFRAME", "1")
CANDLES = int(os.getenv("CANDLES", "50"))
CONCURRENT_TASKS = int(os.getenv("CONCURRENT_TASKS", "20"))
PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "0.01"))
VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", "2.0"))
RISK_REWARD = float(os.getenv("RISK_REWARD", "2.0"))
RETRAIN_INTERVAL = int(os.getenv("RETRAIN_INTERVAL", "3600"))
LOG_PATH = os.getenv("LOG_PATH", "logs/signals_log.csv")
BYBIT_DRY_RUN = os.getenv("BYBIT_DRY_RUN", "True").lower() in ("true", "1", "yes")