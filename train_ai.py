import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# === 1. Загрузка данных ===
print("📊 Загружаем датасет...")
df = pd.read_excel("teacher1_enriched.xlsx")
# Убираем пробелы в названиях и заполняем пропуски
df.columns = df.columns.str.strip()
df = df.fillna(df.median(numeric_only=True))

# Проверяем пустые значения
print("Пустые значения по столбцам:")
print(df.isna().sum())


# === 2. Подготовка данных ===
features = [
    "RSI_5m", "RSI_1h", "RSI_4h", "MFI", "BB_Over%", "VWAP_Dist%", "Vol_Decay",
    "Shadow_Ratio", "POC_Dist%", "Velocity", "ATR", "EMA20", "EMA50", "RSI", "VWAP"
]

df = df.fillna(df.median(numeric_only=True))
df = df.dropna(subset=["Win", "Best_Horizon"])
X = df[features]
y_win = df["Win"]
y_horizon = df["Best_Horizon"]

scaler = StandardScaler()
# Заполняем NaN перед масштабированием (чтобы не было предупреждений)
X = X.fillna(0)
X_scaled = scaler.fit_transform(X)

# Удаляем признаки с нулевой дисперсией (не несут информации)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=1e-5)
X_scaled = selector.fit_transform(X_scaled)

print(f"✅ Осталось признаков после фильтрации: {X_scaled.shape[1]}")


# === 3. Разделение на обучающую и тестовую выборки ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_win, test_size=0.2, random_state=42)

# === 4. Обучение модели успеха ===
print("🧠 Обучаем модель вероятности успеха...")
model_win = RandomForestClassifier(n_estimators=200, random_state=42)
model_win.fit(X_train, y_train)

# === 5. Оценка точности ===
pred_win = model_win.predict(X_test)
print("📈 Отчёт по метке Win:")
print(classification_report(y_test, pred_win))
# === Важность признаков ===
import matplotlib.pyplot as plt

importance = model_win.feature_importances_
feature_names = [f for f, keep in zip(features, selector.get_support()) if keep]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance)
plt.xlabel("Важность признака")
plt.ylabel("Показатель")
plt.title("🧠 Важность признаков для метки Win")
plt.tight_layout()
plt.show()


# === 6. Обучение модели времени отработки ===
print("🕒 Обучаем модель времени отработки...")
model_horizon = RandomForestClassifier(n_estimators=200, random_state=42)
model_horizon.fit(X_scaled, y_horizon)

# === 7. Сохранение моделей ===
joblib.dump(model_win, "ai_model_win.pkl")
joblib.dump(model_horizon, "ai_model_horizon.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Модели сохранены:")
print(" - ai_model_win.pkl")
print(" - ai_model_horizon.pkl")
print(" - scaler.pkl")
