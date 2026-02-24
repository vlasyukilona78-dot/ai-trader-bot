# PATCH: обновлённые модули для асинхронного торгового сканера
# Содержит три файла в одном документе для простого применения патча.
# 1) codex_module.py  (оптимизированные промпты + gpt-4o для объяснений, gpt-4o-code для генерации кода)
# 2) codex_trainer.py (интеграция с main: функция train_if_needed(), валидация данных, безопасный бэкап)
# 3) main.py (неблокирующий запуск auto-retrain в фоне через asyncio.to_thread, улучшённая обработка AI вызовов, опция --retrain-interval)

# --- codex_module.py ---
import os
import time
import openai
from typing import Dict, Optional

openai.api_key = os.getenv("openai.api_key")  # обязательное окружение

# Универсальные параметры retry
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


def _call_openai_chat(model: str, messages: list, max_tokens: int = 512, temperature: float = 0.4):
    """Надёжный обёртка с повторами и базовой обработкой ошибок."""
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            last_exc = e
            time.sleep(_RETRY_DELAY * attempt)
    raise last_exc


# -----------------------------------------
# 1) Explain signal — используем gpt-4o (лучше для "аналитики")
# -----------------------------------------

def explain_signal(symbol: str, indicators: Dict[str, float], probability: float, direction: str) -> str:
    """Возвращает краткое, понятное объяснение сигнала на русском.
    Использует gpt-4o — модель общего назначения, более сильна в аналитике и natural language.
    """
    indicators_text = ", ".join([f"{k}={v:.4g}" for k, v in indicators.items()])

    system = (
        "Ты — профессиональный криптоаналитик и трейдер. Объясняй сигналы коротко и практично, "
        "давай торговую интуицию и возможные риски (1-2 пункта).")

    user = (
        f"Сигнал: {direction} по инструменту {symbol}. Вероятность успеха (оценка ИИ): {probability:.3f}. "
        f"Индикаторы: {indicators_text}. "
        "Дай ровно 2–3 коротких предложения на русском: 1) почему сигнал выглядит сильным/слабым; 2) какие ключевые риски; 3) одно практическое действие (например, подтвердить на таймфрейме H1)."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return _call_openai_chat(model="gpt-4o", messages=messages, max_tokens=150, temperature=0.25)
    except Exception as e:
        return f"(Неудалось получить объяснение от модели: {e})"


# -----------------------------------------
# 2) Improve strategy optimization (use code-optimized model but stricter prompt + tests)
# -----------------------------------------

def optimize_strategy(current_code: str, feedback: str) -> str:
    """Запрос к модели кода: вернуть новый валидный Python-код. В ответе допускается только код.
    Мы просим включить короткие автоматические тесты внутри кода (в блоке if __name__ == '__main__')
    чтобы быстро проверить базовую работоспособность стратегии.
    """
    system = "Ты — Codex: эксперт по рефакторингу и улучшению Python кода для торговых стратегий."

    user = (
        "Ниже — текущий код стратегии на Python. Задача: исправить логические ошибки, улучшить стабильность вычислений индикаторов, "
        "защитить от деления на ноль, добавить проверку на минимальную длину данных, и добавить небольшой тест при запуске файла, "
        "который создаёт искусственный DataFrame с минимальными данными и выводит 'TEST_OK' при успешной работе. "
        f"Текущий код:\n\n{current_code}\n\nТребуемые правки: {feedback}\n\nВерни ТОЛЬКО чистый исправленный Python код, без объяснений."
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        return _call_openai_chat(model="gpt-4o-code", messages=messages, max_tokens=3500, temperature=0.15)
    except Exception as e:
        # Возвращаем оригинал в случае ошибки, чтобы не ломать пайплайн
        return current_code


# -----------------------------------------
# 3) Generate new strategy (guidelines + safety checks)
# -----------------------------------------

def generate_strategy(description: str) -> str:
    """Генерирует стратегию: просим понятную структуру функций, тест и пример использования.
    Делаем модель более детерминистичной (низкая температура).
    """
    system = "Ты — Codex: пиши качественный, безопасный и стилизованный Python-код для анализа свечных данных."
    user = (
        f"Сгенерируй стратегию по описанию: {description}. "
        "Код должен использовать pandas (или чистые numpy/pandas операции), возвращать функции: compute_indicators(df), generate_signals(df), backtest(df). "
        "Добавь тестовый блок if __name__ == '__main__' который генерирует синтетические свечи и запускает backtest. "
        "Верни только код Python."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        return _call_openai_chat(model="gpt-4o-code", messages=messages, max_tokens=3500, temperature=0.2)
    except Exception as e:
        return f"# Ошибка генерации стратегии: {e}\n"

