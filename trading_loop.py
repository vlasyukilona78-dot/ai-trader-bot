import asyncio
import time
import traceback


class TradingLoop:
    """
    Торговый цикл — мониторит открытые позиции, закрывает по условиям,
    и уведомляет об изменениях через callback.
    """

    def __init__(self, bybit_client):
        self.bybit_client = bybit_client
        self.active_trades = {}  # order_id -> dict
        self._running = True
        print("✅ TradingLoop инициализирован")

    def _get_positions_safe(self):
        """Универсальный способ получить открытые позиции."""
        try:
            if hasattr(self.bybit_client, "get_open_positions"):
                return self.bybit_client.get_open_positions()
            elif hasattr(self.bybit_client, "fetch_open_positions"):
                return self.bybit_client.fetch_open_positions()
            else:
                print("⚠️ BybitClient не имеет метода получения позиций")
                return []
        except Exception as e:
            print("⚠️ Ошибка получения позиций:", e)
            return []

    def register_trade(self, order_id, symbol, direction, entry_price, qty):
        """Регистрирует новую сделку для отслеживания."""
        self.active_trades[order_id] = {
            "symbol": symbol,
            "side": direction,
            "entry_price": entry_price,
            "amount": qty,
            "opened_at": time.time(),
            "closed": False,
            "profit": 0.0
        }
        print(f"📥 Зарегистрирована сделка {symbol} ({direction}) по цене {entry_price}")

    async def monitor_loop(self, on_trade_closed_callback=None):
        """Главный цикл мониторинга открытых позиций."""
        print("📡 monitor_loop started")
        while self._running:
            try:
                positions = self._get_positions_safe()

                # если Bybit не вернул ничего — подождём
                if not positions:
                    await asyncio.sleep(10)
                    continue

                for pos in positions:
                    try:
                        symbol = pos.get("symbol")
                        side = pos.get("side")
                        size = float(pos.get("size", 0))
                        entry = float(pos.get("entryPrice", 0))
                        pnl = float(pos.get("unrealisedPnl", 0))

                        # поиск активной сделки
                        trade = next((t for t in self.active_trades.values() if t["symbol"] == symbol), None)
                        if not trade:
                            continue

                        trade["profit"] = pnl

                        # если позиция закрыта
                        if size <= 0 and not trade["closed"]:
                            trade["closed"] = True
                            trade["closed_at"] = time.time()
                            trade["duration"] = trade["closed_at"] - trade["opened_at"]

                            print(f"💰 Сделка {symbol} закрыта | PnL={pnl:.4f}")
                            if on_trade_closed_callback:
                                result = {
                                    "symbol": symbol,
                                    "side": side,
                                    "profit": pnl,
                                    "reason": "Position closed",
                                    "closed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "duration": trade["duration"]
                                }
                                on_trade_closed_callback(result)

                    except Exception as inner_e:
                        print("⚠️ Ошибка обработки позиции:", inner_e)
                        traceback.print_exc()

                await asyncio.sleep(15)

            except Exception as e:
                print("Error in monitor_loop:", e)
                traceback.print_exc()
                await asyncio.sleep(10)

    def stop(self):
        """Останавливает торговый цикл."""
        self._running = False
        print("🛑 TradingLoop остановлен")