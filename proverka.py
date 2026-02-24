# proverka.py — проверка согласованности scaler / моделей (без предупреждений)
import joblib
import numpy as np
import pandas as pd
import warnings

print("Loading models...")
model_win = joblib.load("ai_model_win.pkl")
model_horizon = joblib.load("ai_model_horizon.pkl")
scaler = joblib.load("scaler.pkl")

print("MODEL_WIN:", type(model_win), type(model_win).__name__)
print("MODEL_HORIZON:", type(model_horizon), type(model_horizon).__name__)
print("SCALER:", type(scaler))

n_features = getattr(scaler, "n_features_in_", None)
print("scaler.n_features_in_:", n_features)

# создаём fake vector нужной длины
if n_features is None:
    # если scaler не хранит feature_names_in_, используем 11 (дефолт)
    n_features = 11

fake_array = np.random.random((1, n_features))
print("fake_features shape:", fake_array.shape)

# Если scaler был обучен с именами (feature_names_in_), передаём DataFrame с такими именами
if hasattr(scaler, "feature_names_in_"):
    cols = list(scaler.feature_names_in_)
    # безопасно: если размерность не совпадает — подстраиваем
    if len(cols) != fake_array.shape[1]:
        print("⚠️ Количество имен признаков в scaler.feature_names_in_ не совпадает с длиной fake vector.")
        print("feature_names_in_ length:", len(cols), "fake vector length:", fake_array.shape[1])
        # подстроим: либо усечём, либо дополним искусственными именами
        if len(cols) > fake_array.shape[1]:
            cols = cols[: fake_array.shape[1]]
        else:
            extra = [f"f{i}" for i in range(len(cols), fake_array.shape[1])]
            cols = cols + extra
    fake_df = pd.DataFrame(fake_array, columns=cols)
    X_for_transform = fake_df
else:
    # обычный numpy массив — scaler.transform примет его без warning
    X_for_transform = fake_array

# опционально подавляем warning в этой проверке (если хочется чистый вывод)
with warnings.catch_warnings():
    warnings.simplefilter("default")
    try:
        scaled = scaler.transform(X_for_transform)
        print("Scaled OK:", scaled)
    except Exception as e:
        print("Scaler transform failed:", e)
        raise

# предсказания
try:
    win_pred = model_win.predict(scaled)
    horizon_pred = model_horizon.predict(scaled)
    print("WIN pred:", win_pred, "HORIZON pred:", horizon_pred)
except Exception as e:
    print("Prediction failed:", e)