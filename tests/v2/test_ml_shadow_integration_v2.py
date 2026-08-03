from types import SimpleNamespace

from app.main import _attach_ml_shadow_prediction
from trading.signals.signal_types import IntentAction, StrategyIntent


class _FakeService:
    def __init__(self):
        self.artifacts = SimpleNamespace(feature_names=["rsi", "atr"], version="test-v1")
        self.calls = []

    def predict(self, features):
        self.calls.append(features)
        return SimpleNamespace(
            probability=0.73,
            horizon=11.0,
            model_enabled=True,
            reason="ok",
        )


def _features(values):
    return SimpleNamespace(row=SimpleNamespace(values=values))


def test_ml_shadow_prediction_is_advisory_and_uses_exact_model_schema():
    service = _FakeService()
    intent = StrategyIntent(
        symbol="BTCUSDT",
        action=IntentAction.SHORT_ENTRY,
        reason="layered_short_entry",
        metadata={"original": True},
    )

    result = _attach_ml_shadow_prediction(
        intent=intent,
        features=_features({"rsi": 71.0, "atr": 0.8, "new_runtime_feature": 1.0}),
        service=service,
    )

    assert result.action == IntentAction.SHORT_ENTRY
    assert result.reason == "layered_short_entry"
    assert service.calls == [{"rsi": 71.0, "atr": 0.8}]
    assert result.metadata["original"] is True
    assert result.metadata["ml_shadow_enabled"] is True
    assert result.metadata["ml_shadow_governed"] is False
    assert result.metadata["ml_shadow_probability"] == 0.73
    assert result.metadata["ml_shadow_horizon"] == 11.0
    assert result.metadata["ml_shadow_version"] == "test-v1"
    assert result.metadata["ml_shadow_reason"] == "ok:ungoverned_artifact"


def test_ml_shadow_disables_prediction_when_model_feature_is_missing():
    service = _FakeService()
    intent = StrategyIntent(
        symbol="BTCUSDT",
        action=IntentAction.HOLD,
        reason="no_signal_layer2",
        metadata={
            "layer_trace": {
                "layers": {
                    "layer1_pump_detection": {"passed": True},
                }
            }
        },
    )

    result = _attach_ml_shadow_prediction(
        intent=intent,
        features=_features({"rsi": 71.0}),
        service=service,
    )

    assert service.calls == []
    assert result.metadata["ml_shadow_enabled"] is False
    assert result.metadata["ml_shadow_reason"] == "feature_parity_missing"


def test_ml_shadow_skips_non_pump_hold():
    service = _FakeService()
    intent = StrategyIntent(
        symbol="BTCUSDT",
        action=IntentAction.HOLD,
        reason="no_signal_layer1",
        metadata={"layer_trace": {"layers": {}}},
    )

    result = _attach_ml_shadow_prediction(
        intent=intent,
        features=_features({"rsi": 50.0, "atr": 0.8}),
        service=service,
    )

    assert service.calls == []
    assert "ml_shadow_enabled" not in result.metadata
