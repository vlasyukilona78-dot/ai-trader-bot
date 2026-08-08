from __future__ import annotations

from trading.signals.volatility_context import VolatilityContext, VolatilityContextConfig


def _floor_after_worker_order(order: tuple[str, str]) -> tuple[float, int]:
    context = VolatilityContext(
        VolatilityContextConfig(
            percentile=0.8,
            max_age_sec=10.0,
            min_observations=1,
            fallback_floor=0.5,
        )
    )
    values = {"LOW": 0.1, "HIGH": 0.9}
    context.start_sweep(now=100.0)
    for worker_index, symbol in enumerate(order):
        # These are worker completion clocks, not market-observation clocks. Both
        # values belong to the sweep that began at t=100.
        context.observe(symbol, values[symbol], now=100.0 + worker_index * 9.0)

    context.start_sweep(now=110.5)
    return context.floor(), context.observed_symbols


def test_one_sweep_uses_one_timestamp_near_the_ttl_boundary() -> None:
    forward = _floor_after_worker_order(("LOW", "HIGH"))
    reverse = _floor_after_worker_order(("HIGH", "LOW"))

    # At t=110.5 the complete t=100 sweep expires as one unit. Previously the
    # last worker survived, so reversing worker order changed the next floor from
    # 0.9 to 0.1.
    assert forward == reverse == (0.5, 0)
