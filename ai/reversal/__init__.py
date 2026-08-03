"""Causal reversal-research contracts for the MEXC signals-only path."""

from .feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_SPECS,
    FeatureRole,
    FeatureSpec,
    RuntimeStatus,
    build_runtime_feature_snapshot,
    configuration_hash,
    feature_contract_hash,
    feature_registry_hash,
    model_feature_names,
    model_feature_specs,
)

__all__ = [
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_SPECS",
    "FeatureRole",
    "FeatureSpec",
    "RuntimeStatus",
    "build_runtime_feature_snapshot",
    "configuration_hash",
    "feature_contract_hash",
    "feature_registry_hash",
    "model_feature_names",
    "model_feature_specs",
]
