# retrain_scaler_16.py
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Создаем синтетические данные для 16 признаков, чтобы задать структуру скалера
# Соответствует порядку в compute_indicators:
# 1.RSI, 2.VolRatio, 3.DistBB, 4.OBV_Slope, 5.VWAP, 6.POC, 7.VAH, 8.Sentiment
# 9.PriceChg, 10.Trend5m, 11.ATR, 12.EMA_Diff, 13.VolChg, 14.Range, 15.RSI_Mom, 16.BB_Count
X = np.random.rand(100, 16)

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler.pkl инициализирован на 16 признаков.")