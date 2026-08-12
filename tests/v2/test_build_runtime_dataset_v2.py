"""The parallel dataset builder must be as deterministic as the serial one.

Spreading symbols across processes is the one speed lever that does not touch
behaviour-locked strategy code, but it introduces the same hazard the rest of
this project spent several review rounds removing: a result that depends on which
worker happened to finish first. These tests pin the properties that make the
parallel build trustworthy rather than merely faster.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ai.build_runtime_dataset import (
    SymbolResult,
    build_dataset,
    complete_symbols,
    concatenate,
)
import ai.build_runtime_dataset as builder


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        for interval in ("Min60", "Hour4", "Min5"):
            (root / f"{symbol}_{interval}.csv").write_text("time\n1\n", encoding="utf-8")
    # A symbol missing its 5m history cannot be labelled and must be skipped.
    (root / "PARTIALUSDT_Min60.csv").write_text("time\n1\n", encoding="utf-8")
    (root / "PARTIALUSDT_Hour4.csv").write_text("time\n1\n", encoding="utf-8")
    return root


def test_only_symbols_with_every_required_timeframe_are_selected(cache) -> None:
    assert complete_symbols(cache) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


def _fake_worker(payload):
    """Return rows whose content identifies the symbol, ignoring real history."""

    symbol, _cache, output, _horizon = payload
    from pathlib import Path

    rows = [{"symbol": symbol, "ts": index} for index in range(2)]
    shard = Path(output) / f"{symbol}.jsonl"
    shard.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return SymbolResult(symbol=symbol, rows=len(rows), events=len(rows))


def test_completion_order_does_not_change_the_dataset(tmp_path, monkeypatch, cache) -> None:
    """The hazard parallelism introduces: results ordered by whoever finished."""

    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]

    def run(reverse: bool):
        output = tmp_path / ("rev" if reverse else "fwd")
        destination = tmp_path / f"{'rev' if reverse else 'fwd'}.csv"

        class _Pool:
            """Yields futures in a deliberately different order each run."""

            def __init__(self, max_workers=None):
                self.jobs = []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, payload):
                class _Future:
                    def __init__(self, value):
                        self._value = value

                    def result(self):
                        return self._value

                future = _Future(fn(payload))
                self.jobs.append(future)
                return future

        pool_holder = {}

        def _pool_factory(max_workers=None):
            pool = _Pool()
            pool_holder["pool"] = pool
            return pool

        monkeypatch.setattr(builder, "ProcessPoolExecutor", _pool_factory)
        monkeypatch.setattr(builder, "_worker", _fake_worker)
        monkeypatch.setattr(
            builder,
            "as_completed",
            lambda futures: reversed(list(futures)) if reverse else list(futures),
        )
        results = build_dataset(symbols, cache=cache, output=output, workers=2)
        concatenate(symbols, output=output, destination=destination)
        return [r.symbol for r in results], destination.read_text(encoding="utf-8")

    forward_order, forward_csv = run(reverse=False)
    reverse_order, reverse_csv = run(reverse=True)

    assert forward_order == symbols
    assert reverse_order == symbols
    assert forward_csv == reverse_csv


def test_a_failing_symbol_does_not_end_the_run(tmp_path, monkeypatch, cache) -> None:
    def _explode(payload):
        symbol = payload[0]
        if symbol == "BBBUSDT":
            return SymbolResult(symbol=symbol, rows=0, events=0, error="ValueError")
        return _fake_worker(payload)

    monkeypatch.setattr(builder, "_worker", _explode)
    results = build_dataset(
        ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
        cache=cache,
        output=tmp_path / "shards",
        workers=1,
    )

    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["BBBUSDT"].error == "ValueError"
    assert by_symbol["AAAUSDT"].rows == 2
    assert by_symbol["CCCUSDT"].rows == 2


def test_resume_skips_symbols_that_already_have_a_shard(tmp_path, monkeypatch, cache) -> None:
    output = tmp_path / "shards"
    output.mkdir()
    (output / "AAAUSDT.jsonl").write_text(
        json.dumps({"symbol": "AAAUSDT", "ts": 1}) + "\n", encoding="utf-8"
    )
    called: list[str] = []

    def _record(payload):
        called.append(payload[0])
        return _fake_worker(payload)

    monkeypatch.setattr(builder, "_worker", _record)
    results = build_dataset(
        ["AAAUSDT", "BBBUSDT"], cache=cache, output=output, workers=1
    )

    assert called == ["BBBUSDT"]  # the finished symbol is not recomputed
    assert {r.symbol: r.rows for r in results} == {"AAAUSDT": 1, "BBBUSDT": 2}


def test_a_partial_shard_is_never_mistaken_for_a_finished_one(tmp_path, monkeypatch, cache) -> None:
    """The worker renames into place, so an interrupted write leaves no shard."""

    output = tmp_path / "shards"
    output.mkdir()
    (output / "AAAUSDT.jsonl.partial").write_text('{"symbol":"AAAUSDT"}\n', encoding="utf-8")

    called: list[str] = []

    def _record(payload):
        called.append(payload[0])
        return _fake_worker(payload)

    monkeypatch.setattr(builder, "_worker", _record)
    build_dataset(["AAAUSDT"], cache=cache, output=output, workers=1)

    assert called == ["AAAUSDT"]


def test_a_single_worker_run_uses_no_process_pool(tmp_path, monkeypatch, cache) -> None:
    """Debuggability: one worker must not hide a traceback behind pickling."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("a single-worker run must not spawn a pool")

    monkeypatch.setattr(builder, "ProcessPoolExecutor", _forbidden)
    monkeypatch.setattr(builder, "_worker", _fake_worker)

    results = build_dataset(
        ["AAAUSDT"], cache=cache, output=tmp_path / "shards", workers=1
    )
    assert [r.symbol for r in results] == ["AAAUSDT"]


def test_concatenate_preserves_the_requested_symbol_order(tmp_path) -> None:
    output = tmp_path / "shards"
    output.mkdir()
    for symbol in ("AAAUSDT", "BBBUSDT"):
        (output / f"{symbol}.jsonl").write_text(
            json.dumps({"symbol": symbol, "ts": 1}) + "\n", encoding="utf-8"
        )

    destination = tmp_path / "joined.csv"
    total = concatenate(["BBBUSDT", "AAAUSDT"], output=output, destination=destination)

    assert total == 2
    assert list(pd.read_csv(destination)["symbol"]) == ["BBBUSDT", "AAAUSDT"]


def test_no_shards_produces_no_file_and_reports_zero(tmp_path) -> None:
    output = tmp_path / "shards"
    output.mkdir()
    destination = tmp_path / "joined.csv"

    assert concatenate(["AAAUSDT"], output=output, destination=destination) == 0
    assert not destination.exists()
