from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_leverage: float = 3.0
    max_concurrent_positions: int = 2
    max_symbol_exposure_pct: float = 0.25
    max_total_notional_pct: float = 0.60
    cooldown_after_stop_sec: int = 900
    halt_after_consecutive_losses: int = 4
    min_liquidation_buffer_pct: float = 0.01
    execution_cost_buffer_bps: float = 6.0
    require_stop_loss: bool = True
    pyramiding_enabled: bool = False



def load_risk_limits_from_env() -> RiskLimits:
    return RiskLimits(
        max_risk_per_trade_pct=float(os.getenv("RISK_MAX_RISK_PER_TRADE_PCT", "0.01")),
        max_daily_loss_pct=float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", "0.03")),
        max_leverage=float(os.getenv("RISK_MAX_LEVERAGE", "3.0")),
        max_concurrent_positions=int(os.getenv("RISK_MAX_CONCURRENT_POSITIONS", "2")),
        max_symbol_exposure_pct=float(os.getenv("RISK_MAX_SYMBOL_EXPOSURE_PCT", "0.25")),
        max_total_notional_pct=float(os.getenv("RISK_MAX_TOTAL_NOTIONAL_PCT", "0.60")),
        cooldown_after_stop_sec=int(os.getenv("RISK_COOLDOWN_AFTER_STOP_SEC", "900")),
        halt_after_consecutive_losses=int(os.getenv("RISK_HALT_AFTER_CONSECUTIVE_LOSSES", "4")),
        min_liquidation_buffer_pct=float(os.getenv("RISK_MIN_LIQ_BUFFER_PCT", "0.01")),
        execution_cost_buffer_bps=float(os.getenv("RISK_EXECUTION_COST_BUFFER_BPS", "6.0")),
        require_stop_loss=os.getenv("RISK_REQUIRE_STOP_LOSS", "true").lower() in ("1", "true", "yes"),
        pyramiding_enabled=os.getenv("RISK_PYRAMIDING_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
