# check_dataset.py — диагностика данных для обучения
import os
import pandas as pd

LOG_PATH = "logs/signals_log.csv"

print("=== Проверка датасета ===")
if not os.path.exists(LOG_PATH):
    print(f"❌ Файл {LOG_PATH} не найден!")
    raise SystemExit

try:
    df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
except Exception as e:
    print("❌ Ошибка чтения CSV:", e)
    raise SystemExit

print(f"✅ Прочитано {len(df)} строк, {len(df.columns)} столбцов")
print("Столбцы:", list(df.columns))
print("-" * 50)

# Проверим, какие столбцы заполнены
for col in df.columns:
    na_count = df[col].isna().sum()
    print(f"{col:<15} | пустых: {na_count:<5} | заполнено: {len(df)-na_count}")

# Минимальные обязательные поля
required = {"entry", "tp", "sl"}
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"\n⚠️ Не хватает столбцов: {missing}")
else:
    empty = [c for c in required if df[c].dropna().empty]
    if empty:
        print(f"\n⚠️ Полностью пустые столбцы: {empty}")
    else:
        print("\n✅ Все ключевые столбцы (entry, tp, sl) присутствуют и заполнены")

print("-" * 50)
if len(df) < 10:
    print("⚠️ Мало строк для обучения (менее 10) — тренер ничего не обучит.")
else:
    print("✅ Данных достаточно для тренировки.")