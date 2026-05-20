from __future__ import annotations


ENTRY_GATE_VERSION = "entry_gate_v2_live_continuation_microstructure_1"
STRATEGY_RUNTIME_VERSION = "layered_v2_entry_gate_live_continuation_dedupe_ultra_v2_microstructure_1"
FEATURE_PIPELINE_VERSION = "feature_pipeline_v2_mtf_bound_1"
EXECUTION_GUARD_VERSION = "execution_guard_v2"
ULTRA_V2_VERSION = "ultra_v2_culmination_scenarios_1"


def runtime_versions() -> dict[str, str]:
    return {
        "strategy_runtime": STRATEGY_RUNTIME_VERSION,
        "entry_gate": ENTRY_GATE_VERSION,
        "feature_pipeline": FEATURE_PIPELINE_VERSION,
        "execution_guard": EXECUTION_GUARD_VERSION,
        "ultra_v2": ULTRA_V2_VERSION,
    }
