import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 🚀 Генерируем условный набор данных 1000x11
# (только чтобы scaler понял размерность и диапазоны)
# Если есть реальные обучающие данные — вставь их сюда вместо X
X = np.random.rand(1000, 11) * np.array([
    100,  # RSI
    100,  # RSI_5mean
    100,  # RSI_20mean
    0.02, # price_change
    3.0,  # vol_change
    50000, # EMA20
    50000, # EMA50
    1000,  # ATR
    50000, # Close
    1.0,   # EMA_diff_norm
    0.05   # ATR_norm
])

# ⚙️ Обучаем StandardScaler под эти 11 признаков
scaler = StandardScaler()
scaler.fit(X)

# 💾 Сохраняем в scaler.pkl
joblib.dump(scaler, "scaler.pkl")

print("✅ Новый scaler.pkl обучен на 11 признаках и готов к использованию.")