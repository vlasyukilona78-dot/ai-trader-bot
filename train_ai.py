import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import joblib

# === 1. Загрузка данных ===
print("📊 Загружаем датасет...")
df = pd.read_excel("teacher1_enriched.xlsx")

# Очистка
df.columns = df.columns.str.strip()
df = df.fillna(df.median(numeric_only=True))

# Проверяем пустые значения
print("Пустые значения по столбцам:")
print(df.isna().sum())

# === 2. Фильтрация под стратегию "shorting the pump" ===
# Оставляем только short-сценарии
if "Signal" in df.columns:
    before = len(df)
    df = df[df["Signal"].str.upper() == "SHORT"]
    after = len(df)
    print(f"🎯 Фильтрация: оставлено только SHORT {after}/{before} записей.")
else:
    print("⚠️ Столбец 'Signal' не найден — обучаемся на всём датасете.")

# === 3. Подготовка данных ===
features = [
    "RSI_5m", "RSI_1h", "RSI_4h", "MFI", "BB_Over%", "VWAP_Dist%", "Vol_Decay",
    "Shadow_Ratio", "POC_Dist%", "Velocity", "ATR", "EMA20", "EMA50", "RSI", "VWAP"
]

df = df.fillna(df.median(numeric_only=True))
df = df.dropna(subset=["Win", "Best_Horizon"])
X = df[features]
y_win = df["Win"]
y_horizon = df["Best_Horizon"]

# === 4. Масштабирование и фильтрация ===
scaler = StandardScaler()
X = X.fillna(0)
X_scaled = scaler.fit_transform(X)

selector = VarianceThreshold(threshold=1e-5)
X_scaled = selector.fit_transform(X_scaled)

print(f"✅ Осталось признаков после фильтрации: {X_scaled.shape[1]}")

# === 5. Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_win, test_size=0.2, random_state=42)

# === 6. Обучение модели вероятности успеха ===
print("🧠 Обучаем модель вероятности успеха (SHORT)...")
model_win = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight="balanced_subsample"  # помогает при дисбалансе данных
)
model_win.fit(X_train, y_train)

# === 7. Оценка точности ===
pred_win = model_win.predict(X_test)
print("📈 Отчёт по метке Win:")
print(classification_report(y_test, pred_win))

# Важность признаков
importance = model_win.feature_importances_
feature_names = [f for f, keep in zip(features, selector.get_support()) if keep]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance, color="lightcoral")
plt.xlabel("Важность признака")
plt.title("🧠 Важность признаков для Win (SHORT стратегия)")
plt.tight_layout()
plt.show()

# === 8. Обучение модели горизонта ===
print("🕒 Обучаем модель времени отработки...")
model_horizon = RandomForestClassifier(n_estimators=200, random_state=42)
model_horizon.fit(X_scaled, y_horizon)

# === 9. Сохранение моделей ===
joblib.dump(model_win, "ai_model_win.pkl")
joblib.dump(model_horizon, "ai_model_horizon.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Модели сохранены:")
print(" - ai_model_win.pkl")
print(" - ai_model_horizon.pkl")
print(" - scaler.pkl")