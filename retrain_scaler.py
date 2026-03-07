import pandas as pd

print("🔄 Обновляем датасет teacher1_enriched.xlsx из signals_history.xlsx...")

try:
    base = pd.read_excel("teacher1_enriched.xlsx")
    new = pd.read_excel("signals_history.xlsx")

    # удаляем неполные или явно ошибочные записи
    new = new.dropna(subset=["Win", "Symbol"])
    new = new[new["Profit"].abs() < 50_000]

    # добавляем отсутствующие столбцы
    for col in new.columns:
        if col not in base.columns:
            base[col] = None

    # объединяем по уникальным ключам Symbol + Time
    merged = pd.concat([base, new], ignore_index=True).drop_duplicates(subset=["Symbol", "Time"])

    merged.to_excel("teacher1_enriched.xlsx", index=False)
    print(f"✅ Датасет обновлён. Всего {len(merged)} записей.")
except Exception as e:
    print("⚠️ Ошибка обновления датасета:", e)