import os

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_ENV = os.getenv("BYBIT_ENV", "mainnet").lower()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID", "")

TIMEFRAME = os.getenv("TIMEFRAME", "1")
CANDLES = int(os.getenv("CANDLES", "50"))
CONCURRENT_TASKS = int(os.getenv("CONCURRENT_TASKS", "20"))
PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "0.01"))
VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", "8.0"))
RISK_REWARD = float(os.getenv("RISK_REWARD", "2.0"))
RETRAIN_INTERVAL = int(os.getenv("RETRAIN_INTERVAL", "3600"))
LOG_PATH = os.getenv("LOG_PATH", "logs/signals_log.csv")
MODEL_DIR = os.getenv("MODEL_DIR", ".")
BYBIT_DRY_RUN = os.getenv("BYBIT_DRY_RUN", "True").lower() in ("true", "1", "yes")

RSI_PUMP_THRESHOLD = float(os.getenv("RSI_PUMP_THRESHOLD", "80"))
SENTIMENT_EUPHORIA_THRESHOLD = float(os.getenv("SENTIMENT_EUPHORIA_THRESHOLD", "70"))
PROFILE_WINDOW = int(os.getenv("PROFILE_WINDOW", "80"))
PROFILE_BINS = int(os.getenv("PROFILE_BINS", "24"))
KELTNER_MULTIPLIER = float(os.getenv("KELTNER_MULTIPLIER", "1.5"))
SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "https://api.alternative.me/fng/")

MIN_AI_PROB_TO_ORDER = float(os.getenv("MIN_AI_PROB_TO_ORDER", "0.60"))
TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", "logs/trades_log.csv")

RISK_ACCOUNT_EQUITY_USDT = float(os.getenv("RISK_ACCOUNT_EQUITY_USDT", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
RISK_MAX_CONCURRENT_POSITIONS = int(os.getenv("RISK_MAX_CONCURRENT_POSITIONS", "3"))
RISK_DAILY_STOP_LOSS_USDT = float(os.getenv("RISK_DAILY_STOP_LOSS_USDT", "100"))
RISK_MAX_CONSECUTIVE_LOSSES = int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "3"))
RISK_CIRCUIT_BREAKER_MINUTES = int(os.getenv("RISK_CIRCUIT_BREAKER_MINUTES", "30"))
RISK_MIN_QTY = float(os.getenv("RISK_MIN_QTY", "0.001"))
RISK_MAX_QTY = float(os.getenv("RISK_MAX_QTY", "100"))

SENTIMENT_REFRESH_SEC = int(os.getenv("SENTIMENT_REFRESH_SEC", "60"))
SENTIMENT_MAX_AGE_SEC = int(os.getenv("SENTIMENT_MAX_AGE_SEC", "300"))

METRICS_PATH = os.getenv("METRICS_PATH", "logs/metrics.json")

EXPECTED_FEATURES = 11
