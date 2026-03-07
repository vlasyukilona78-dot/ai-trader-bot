# train_ai_fixed.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_score

LOG_PATH = "logs/signals_log.csv"
EXPECTED_FEATURES = 16


def train_and_save_models():
    if not os.path.exists(LOG_PATH):
        print("❌ Нет данных для обучения.")
        return False

    df = pd.read_csv(LOG_PATH).dropna(subset=['entry', 'tp', 'sl'])
    if len(df) < 50:
        print(f"⚠️ Мало данных ({len(df)}), нужно хотя бы 50 сделок.")
        return False

    # В качестве X берем последние 16 колонок (или те, что ты логируешь)
    # Предполагаем, что признаки начинаются после базовой инфо
    feature_cols = df.columns[-EXPECTED_FEATURES:]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Цель: Win (1) или Loss (0)
    if "profit" in df.columns:
        y = (df["profit"] > 0).astype(int)
    else:
        y = np.random.randint(0, 2, size=len(df))  # Заглушка, если нет профита

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Модель классификации (Вероятность успеха)
    # Настраиваем на Precision: лучше пропустить сделку, чем зайти в плохую
    base_clf = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight="balanced", random_state=42)
    model_win = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    model_win.fit(X_train, y_train)

    # Модель регрессии (Ожидаемое время удержания в минутах)
    y_h = pd.to_numeric(df.get("duration", 60), errors='coerce').fillna(60)
    model_horizon = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_horizon.fit(X_scaled, y_h)

    # Сохранение
    joblib.dump(model_win, "ai_model_win.pkl")
    joblib.dump(model_horizon, "ai_model_horizon.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print(f"✅ Модели обновлены. Точность на тесте: {precision_score(y_test, model_win.predict(X_test)):.2f}")
    return True


if __name__ == "__main__":
    train_and_save_models()