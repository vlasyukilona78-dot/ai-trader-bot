# codex_trainer.py (final version)
import os
import shutil
import time
from datetime import datetime
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
from logger import logger
from config import *

# === SETTINGS ===
LOGS_PATH = LOG_PATH if LOG_PATH else "logs/signals_log.csv"
BACKUP_DIR = "backups"
EXPECTED_FEATURES = 11  # main.py формирует 11 признаков

# === INIT LOGGER ===
logger.setLevel("INFO")
logger.addHandler(__import__("logging").StreamHandler())


# === HELPERS ===
def backup_model_files(save_dir="."):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for fname in ["ai_model_win.pkl", "ai_model_horizon.pkl", "scaler.pkl"]:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            dst = os.path.join(BACKUP_DIR, f"{ts}_{fname}")
            shutil.copy2(fpath, dst)
            logger.info(f"🔁 Backup: {fname} -> backups/{ts}_{fname}")


def safe_read_csv(path):
    import csv
    fixed_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_cols:
            r += [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        fixed_lines.append(r)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(fixed_lines)
    logger.info(f"🧹 CSV отремонтирован: {len(fixed_lines)} строк, {max_cols} столбцов")
    return pd.read_csv(path)


def update_dataset():
    if not os.path.exists(LOGS_PATH):
        logger.warning(f"⚠️ Не найден лог: {LOGS_PATH}")
        return pd.DataFrame()
    try:
        df = safe_read_csv(LOGS_PATH)
    except Exception as e:
        logger.exception(f"Ошибка при чтении CSV: {e}")
        return pd.DataFrame()

    # Приводим типы
    for col in ["entry", "tp", "sl", "rsi", "vol_change", "ai_prob", "ai_horizon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["entry", "tp", "sl"])
    df["ai_prob"] = df["ai_prob"].fillna(0.5)
    df["ai_horizon"] = df["ai_horizon"].fillna(8.48)

    logger.info(f"✅ Загружено {len(df)} строк после очистки")
    return df


def prepare_features(df):
    df = df.copy()

    # создаём безопасно все нужные колонки
    for col in ["rsi", "vol_change", "ema20", "ema50", "atr", "close"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["rsi_5mean"] = df["rsi"].rolling(5, min_periods=1).mean()
    df["rsi_20mean"] = df["rsi"].rolling(20, min_periods=1).mean()
    df["price_change"] = df["close"].pct_change().fillna(0.0)
    df["ema_diff_norm"] = (df["ema20"] - df["ema50"]) / (df["close"] + 1e-9)
    df["atr_norm"] = df["atr"] / (df["close"] + 1e-9)

    # целевые переменные
    if "profit" in df.columns:
        df["y_win"] = (pd.to_numeric(df["profit"], errors="coerce") > 0).astype(int)
    elif "ai_prob" in df.columns:
        df["y_win"] = (pd.to_numeric(df["ai_prob"], errors="coerce") > 0.6).astype(int)
    else:
        df["y_win"] = pd.Series([0] * len(df))

    if "duration" in df.columns:
        df["y_horizon"] = pd.to_numeric(df["duration"], errors="coerce").fillna(8.48)
    else:
        df["y_horizon"] = 8.48

    features = [
        "rsi", "rsi_5mean", "rsi_20mean", "price_change", "vol_change",
        "ema20", "ema50", "atr", "close", "ema_diff_norm", "atr_norm"
    ]

    X = df[features].fillna(0.0)
    y = df["y_win"].astype(int)
    y_h = df["y_horizon"].astype(float)

    # Проверка количества признаков
    if X.shape[1] != EXPECTED_FEATURES:
        logger.warning(
            f"⚠️ Несовпадение признаков! main.py ожидает {EXPECTED_FEATURES}, "
            f"а подготовлено {X.shape[1]}"
        )

    return X, y, y_h


def train_models(save_dir="."):
    df = update_dataset()
    if df.empty:
        logger.warning("⚠️ Нет данных для обучения")
        print("⚠️ Нет данных для обучения, проверь logs/signals_log.csv")
        return False

    X, y, y_h = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Классификатор
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    model_win = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    )
    model_win.fit(X_train, y_train)
    preds = model_win.predict(X_test)
    print("\n📊 Отчёт классификатора:\n", classification_report(y_test, preds, digits=3))

    # Регрессор горизонта
    try:
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
            X_scaled, y_h, test_size=0.2, random_state=42
        )
        model_horizon = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model_horizon.fit(X_train_h, y_train_h)
        pred_h = model_horizon.predict(X_test_h)
        print(f"📈 Horizon R² = {r2_score(y_test_h, pred_h):.3f}")
    except Exception:
        model_horizon = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model_horizon.fit(X_scaled, y_h)

    os.makedirs(save_dir, exist_ok=True)
    backup_model_files(save_dir)

    joblib.dump(model_win, os.path.join(save_dir, "ai_model_win.pkl"))
    joblib.dump(model_horizon, os.path.join(save_dir, "ai_model_horizon.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    logger.info(f"💾 Модели сохранены в {save_dir}")
    print("\n✅ Обучение завершено успешно!")
    return True


def train_if_needed(retrain_interval=RETRAIN_INTERVAL, force=False, save_dir="."):
    try:
        model_path = os.path.join(save_dir, "ai_model_win.pkl")
        if not force and os.path.exists(model_path):
            age = time.time() - os.path.getmtime(model_path)
            if age < retrain_interval:
                logger.info(f"🕒 Модель свежая (возраст {age:.1f} сек) — переобучение не требуется")
                return False
        return train_models(save_dir=save_dir)
    except Exception as e:
        logger.exception(f"Ошибка train_if_needed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Запуск обучения моделей...")
    train_models()
    print("✅ Обучение завершено.")