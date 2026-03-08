import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")
import asyncio
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chart_generator import make_candlestick_chart
from config import CHAT_ID
from logger import logger


class TradingLoop:
    """Monitors open trades and records closures for later retraining."""

    def __init__(self, bybit_client, notifier=None, chart_dir: str = "charts"):
        self.bybit_client = bybit_client
        self.notifier = notifier
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.active_trades: dict[str, dict[str, Any]] = {}
        self._running = True
        logger.info("TradingLoop initialized")

    def _get_positions_safe(self) -> list[dict[str, Any]]:
        try:
            if hasattr(self.bybit_client, "get_open_positions"):
                positions = self.bybit_client.get_open_positions()
            elif hasattr(self.bybit_client, "fetch_open_positions"):
                positions = self.bybit_client.fetch_open_positions()
            else:
                logger.warning("BybitClient does not provide a positions method")
                return []
        except Exception as exc:
            logger.error("Failed to load open positions: %s", exc)
            return []

        if isinstance(positions, dict):
            result = positions.get("result", {})
            if isinstance(result, dict):
                items = result.get("list", [])
                return items if isinstance(items, list) else []
            return []
        return positions if isinstance(positions, list) else []

    def register_trade(self, order_id, symbol, direction, entry_price, qty):
        self.active_trades[order_id] = {
            "symbol": symbol,
            "side": direction,
            "entry_price": entry_price,
            "amount": qty,
            "opened_at": time.time(),
            "closed": False,
            "profit": 0.0,
        }
        logger.info("Registered trade %s (%s) at %s", symbol, direction, entry_price)

    async def monitor_loop(self, on_trade_closed_callback=None):
        logger.info("monitor_loop started")
        while self._running:
            try:
                positions = self._get_positions_safe()
                if not positions:
                    await asyncio.sleep(10)
                    continue

                for pos in positions:
                    try:
                        symbol = pos.get("symbol")
                        side = pos.get("side")
                        size = float(pos.get("size", 0) or 0)
                        pnl = float(pos.get("unrealisedPnl", 0) or 0)

                        trade = next(
                            (item for item in self.active_trades.values() if item["symbol"] == symbol),
                            None,
                        )
                        if not trade:
                            continue

                        trade["profit"] = pnl

                        if size <= 0 and not trade["closed"]:
                            trade["closed"] = True
                            trade["closed_at"] = time.time()
                            trade["duration"] = trade["closed_at"] - trade["opened_at"]
                            logger.info("Trade %s closed | PnL=%.4f", symbol, pnl)

                            chart_path = self._build_chart(symbol, trade)
                            self._notify_trade_closed(trade, pnl, chart_path)

                            result = {
                                "symbol": symbol,
                                "side": side,
                                "profit": pnl,
                                "reason": "Position closed",
                                "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "duration": trade["duration"],
                            }
                            log_trade_to_history(
                                result,
                                indicators={
                                    "RSI_5m": pos.get("RSI_5m"),
                                    "MFI": pos.get("MFI"),
                                    "VWAP_Dist%": pos.get("VWAP_Dist%"),
                                },
                            )
                            if on_trade_closed_callback:
                                on_trade_closed_callback(result)

                    except Exception as exc:
                        logger.error("Position processing failed: %s", exc)
                        traceback.print_exc()

                await asyncio.sleep(15)

            except Exception as exc:
                logger.error("Error in monitor_loop: %s", exc)
                traceback.print_exc()
                await asyncio.sleep(10)

    def _build_chart(self, symbol: str, trade: dict[str, Any]) -> Path | None:
        try:
            df = self._get_candles(symbol)
            if df.empty:
                return None
            chart_path = self.chart_dir / f"{symbol}_{int(trade['closed_at'])}.png"
            make_candlestick_chart(
                symbol=symbol,
                df=df.copy(),
                signal=trade["side"],
                entry=trade["entry_price"],
                filename=str(chart_path),
            )
            return chart_path
        except Exception as exc:
            logger.error("Failed to build notification chart: %s", exc)
            return None

    def _get_candles(self, symbol: str) -> pd.DataFrame:
        try:
            payload = self.bybit_client.get_klines(symbol=symbol, interval="5", limit=200)
            rows = self._extract_kline_rows(payload)
            if not rows:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(
                rows,
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
            return df.dropna().sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            logger.error("Failed to load candles for %s: %s", symbol, exc)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def _extract_kline_rows(self, payload: Any) -> list[list[Any]]:
        if isinstance(payload, dict):
            result = payload.get("result", {})
            if isinstance(result, dict) and isinstance(result.get("list"), list):
                return result["list"]
            if isinstance(payload.get("list"), list):
                return payload["list"]
            return []
        return payload if isinstance(payload, list) else []

    def _notify_trade_closed(self, trade: dict[str, Any], pnl: float, chart_path: Path | None):
        if not self.notifier:
            return

        payload = {
            "chat_id": CHAT_ID,
            "message": (
                f"Trade closed: {trade['symbol']}\n"
                f"Direction: {trade['side']}\n"
                f"PnL: {pnl:.4f}\n"
                f"Duration: {trade['duration']:.1f}s"
            ),
            "chart_path": str(chart_path) if chart_path else None,
        }
        try:
            self.notifier(payload)
        except Exception as exc:
            logger.error("Notifier failed: %s", exc)

    def stop(self):
        self._running = False
        logger.warning("TradingLoop stopped")


def log_trade_to_history(result: dict, indicators: dict | None = None):
    """Append a closed trade to the local training history."""
    try:
        log_path = "signals_history.xlsx"
        df_new = pd.DataFrame(
            [
                {
                    "Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": result.get("symbol"),
                    "Direction": result.get("side"),
                    "Profit": result.get("profit"),
                    "Duration": result.get("duration"),
                    "Win": 1 if result.get("profit", 0) > 0 else 0,
                    **(indicators or {}),
                }
            ]
        )
        try:
            df = pd.read_excel(log_path)
            df = pd.concat([df, df_new], ignore_index=True)
        except FileNotFoundError:
            df = df_new
        df.to_excel(log_path, index=False)
        logger.info("Trade appended to %s", log_path)
    except Exception as exc:
        logger.error("Failed to append trade history: %s", exc)

