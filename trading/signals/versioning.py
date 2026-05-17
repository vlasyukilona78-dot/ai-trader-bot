from __future__ import annotations


ENTRY_GATE_VERSION = "entry_gate_v2_mtf_1"
STRATEGY_RUNTIME_VERSION = "layered_v2_entry_gate_mtf_2"
FEATURE_PIPELINE_VERSION = "feature_pipeline_v2_mtf_bound_1"
EXECUTION_GUARD_VERSION = "execution_guard_v2"


def runtime_versions() -> dict[str, str]:
    return {
        "strategy_runtime": STRATEGY_RUNTIME_VERSION,
        "entry_gate": ENTRY_GATE_VERSION,
        "feature_pipeline": FEATURE_PIPELINE_VERSION,
        "execution_guard": EXECUTION_GUARD_VERSION,
    }
